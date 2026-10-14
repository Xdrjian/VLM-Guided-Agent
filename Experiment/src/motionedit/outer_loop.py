"""第 4 层 · 闭环：外层循环主体（workflow.md §4）。

本模块**不依赖 pytorch3d、不依赖任何网络库**：渲染与求解都以注入点（协议对象）
的形式接入。因此它可以在没有 GPU、没有 API Key 的机器上被完整单测，
真正跑实验时只需把两个注入点换成实机实现。

一轮的固定流程::

    LLM 给参数 → 求解器生成动作 → 特征提取 → 渲染(多机位宫格 + 留底视频)
        → VLM 同场对比打分 → History 判定接受/回滚 → 终止检查

第 0 轮是特例：不调 LLM、不调求解器，直接用 A_ref 渲染 + 评判（§4.4）。

手动模式下，任何一次模型调用都会抛 :class:`PendingManualInput` 使流程挂起；
所有中间产物已落盘，人工回填答案文件后重跑即可从断点续跑（幂等）。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .clients import ModelClient, PendingManualInput
from .history import History, StopReason, TerminationConfig
from .params import GaitParams, DEFAULT_PARAMS
from .prompts import (GridSpec, VideoSpec, LLM_SYSTEM, VLM_SYSTEM, extract_json,
                      render_llm_prompt, render_vlm_prompt,
                      render_vlm_prompt_round0)
from .records import EvidenceStore, RawResponse, RoundRecord, VLMScore

__all__ = [
    "RenderOutput",
    "SolveOutput",
    "Renderer",
    "SolveFn",
    "LoopConfig",
    "OuterLoop",
]


# --------------------------------------------------------------------------- #
@dataclass
class RenderOutput:
    """一次渲染的产物。

    ``grids`` 的顺序必须与 :attr:`GridSpec.cameras` 一致，
    因为 VLM prompt 里是按这个顺序声明机位的。
    """

    grids: List[Path] = field(default_factory=list)
    #: 留底视频（三机位横向拼接）。给人看，也可作为 VLM 输入形态的备选试验
    video: Optional[Path] = None
    #: 同内容的 gif（部分模型只吃 gif/静图序列）
    gif: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveOutput:
    """一次求解的产物。"""

    #: 生成动作的落盘路径（pkl）
    motion_path: Optional[Path] = None
    #: 与求解器任务项同源的生物力学步态特征
    features: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    warnings: List[str] = field(default_factory=list)
    #: 求解器执行失效判定（§4.4）：参数指令与实际输出显著背离
    exec_failure: bool = False


class Renderer(Protocol):
    """注入点 1：多机位相位抽帧渲染 + 留底视频。需 pytorch3d，仅 Linux 机可用。"""

    def __call__(self, motion_path: Optional[Path], out_dir: Path,
                 media: Any, tag: str) -> RenderOutput: ...


class SolveFn(Protocol):
    """注入点 2：给定一组参数，跑求解器并提特征。"""

    def __call__(self, params: GaitParams, out_dir: Path) -> SolveOutput: ...


# --------------------------------------------------------------------------- #
@dataclass
class LoopConfig:
    run_dir: Path
    #: 目标重量，全程不变
    target_weight_kg: float = 30.0
    #: 提交给 VLM 的素材规格。A0 探针实测后定为 :class:`VideoSpec`：
    #: 宫格图按步态相位归一化抽帧，时间信息被抹掉，模型无从判断快慢。
    #: 宫格图仍照常渲染，但只作论文插图，不再提交给 VLM。
    media: Any = field(default_factory=VideoSpec)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    #: 同一组图重复问几次取中位数。噪声摸底阶段设 3，摸清后按结论调整
    vlm_repeats: int = 1
    #: 是否在提交时随机交换左右列，用于量化"位置偏好偏差"（审稿人会问）
    swap_columns_probe: bool = False
    #: A_ref 的动作文件路径（第 0 轮直接渲染它，不经求解器）
    reference_motion: Optional[Path] = None


# --------------------------------------------------------------------------- #
class OuterLoop:
    """外层循环。用法::

        loop = OuterLoop(cfg, vlm_factory, llm_factory, renderer, solve_fn)
        loop.run()      # 手动模式下会在需要人工介入处挂起并打印指引
    """

    def __init__(self, cfg: LoopConfig,
                 vlm_factory: Callable[[int, EvidenceStore, int], ModelClient],
                 llm_factory: Callable[[int, EvidenceStore], ModelClient],
                 renderer: Optional[Renderer] = None,
                 solve_fn: Optional[SolveFn] = None):
        self.cfg = cfg
        self.store = EvidenceStore(cfg.run_dir)
        self.vlm_factory = vlm_factory
        self.llm_factory = llm_factory
        self.renderer = renderer
        self.solve_fn = solve_fn

        hist_path = self.store.run_dir / "history.json"
        if hist_path.exists():
            self.history = History.load(hist_path, cfg.termination)
        else:
            self.history = History(cfg.termination)

    # ------------------------------------------------------------ 主流程 -- #
    def run(self, max_steps: Optional[int] = None) -> History:
        """连续推进直到终止或需要人工介入。"""
        self._save_run_meta()
        steps = 0
        while self.history.should_stop() is None:
            if max_steps is not None and steps >= max_steps:
                break
            try:
                self.step()
            except PendingManualInput as e:
                self._persist()
                print(e)
                return self.history
            steps += 1
        self._persist()
        return self.history

    def step(self) -> RoundRecord:
        """推进一轮。可重复调用；已完成的子步骤会复用磁盘产物（幂等）。"""
        idx = self.history.next_round_index
        rdir = self.store.ensure(idx)

        if idx == 0:
            record = self._round_zero(idx, rdir)
        else:
            record = self._round_n(idx, rdir)

        self.history.add(record)
        self.store.save_record(record)
        self._post_round_checks(record)
        self._persist()
        return record

    # ----------------------------------------------------------- 第 0 轮 -- #
    def _round_zero(self, idx: int, rdir: Path) -> RoundRecord:
        """A_ref 直接评判：不调 LLM、不调求解器（§4.4）。

        跳过求解器的理由：恒等参数下任务项目标曲线就是 A_ref，
        求解器理论最优解即 A_ref 本身，跑一遍只是多花一次调用、不带来信息。
        """
        render = self._render(self.cfg.reference_motion, rdir, tag="reference")
        features = self._reference_features()

        prompt = render_vlm_prompt_round0(self.cfg.target_weight_kg, self.cfg.media)
        score, resps = self._ask_vlm(idx, prompt, self.cfg.media.select(render))

        rec = RoundRecord(
            round_index=idx, params=DEFAULT_PARAMS, features=features,
            score=score, is_reference=True, solver_status="skipped(reference)",
            artifacts=self._artifacts(render, None),
        )
        rec.artifacts["vlm_calls"] = str(len(resps))
        return rec

    # ----------------------------------------------------------- 第 N 轮 -- #
    def _round_n(self, idx: int, rdir: Path) -> RoundRecord:
        params, reasoning = self._ask_llm(idx)

        if self.solve_fn is None:
            raise RuntimeError("未注入 solve_fn，无法生成候选动作。")
        solved = self.solve_fn(params, rdir)

        if solved.exec_failure:
            # 求解器执行失效：独立于 Score 收敛之外的异常终止（§4.4）
            self.history.force_stop(
                StopReason.SOLVER_FAILURE,
                f"第 {idx} 轮参数指令与实际输出显著背离：{'; '.join(solved.warnings)}")
            return RoundRecord(
                round_index=idx, params=params, features=solved.features,
                score=None, llm_reasoning=reasoning,
                solver_status=solved.status, warnings=list(solved.warnings),
            )

        render = self._render(solved.motion_path, rdir, tag=f"round{idx:03d}")
        best = self.history.best
        prompt = render_vlm_prompt(idx, self.cfg.target_weight_kg, best, self.cfg.media)
        images = self._compose_pairs(best, render)
        score, resps = self._ask_vlm(idx, prompt, images)

        rec = RoundRecord(
            round_index=idx, params=params, features=solved.features,
            score=score, llm_reasoning=reasoning,
            solver_status=solved.status, warnings=list(solved.warnings),
            artifacts=self._artifacts(render, solved),
        )
        rec.artifacts["vlm_calls"] = str(len(resps))
        return rec

    # -------------------------------------------------------------- 子步 -- #
    def _render(self, motion_path: Optional[Path], rdir: Path, tag: str) -> RenderOutput:
        if self.renderer is None:
            raise RuntimeError(
                "未注入 renderer。渲染依赖 pytorch3d，只能在配置好的 Linux 机器上运行。")
        return self.renderer(motion_path, rdir, self.cfg.media, tag)

    def _compose_pairs(self, best: Optional[RoundRecord],
                       cand: RenderOutput) -> List[Path]:
        """拼出提交顺序：先历史最佳解的素材，再本轮候选的。

        第一轮尚无历史最佳时退化为只提交候选（此时 prompt 的锚点段也为空）。
        具体取哪几份文件由素材规格决定，不在此处硬编码——否则 prompt 正文和
        实际附件会错位，而且不报任何错。
        """
        right = self.cfg.media.select(cand)
        if best is None:
            return right
        left = [self.store.run_dir / p for p in self._media_paths(best)]
        missing = [p for p in left if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"历史最佳解（第 {best.round_index} 轮）的素材缺失，无法同场对比：{missing}")
        return left + right

    def _media_paths(self, rec: RoundRecord) -> List[str]:
        raw = rec.artifacts.get(self.cfg.media.artifact_key, "")
        return [s for s in raw.split("|") if s]

    def _artifacts(self, render: RenderOutput,
                   solved: Optional[SolveOutput]) -> Dict[str, str]:
        art = {"grids": "|".join(self.store.rel(p) for p in render.grids)}
        if render.video:
            art["video"] = self.store.rel(render.video)
        if render.gif:
            art["gif"] = self.store.rel(render.gif)
        if solved is not None and solved.motion_path is not None:
            art["motion"] = self.store.rel(solved.motion_path)
        return art

    # --------------------------------------------------------- 模型调用 -- #
    def _ask_vlm(self, idx: int, prompt: str,
                 images: Sequence[Path]) -> tuple:
        """调用 VLM（可重复多次做噪声摸底），返回 (聚合分数, 所有原始返回)。"""
        self.store.save_prompt(idx, "vlm", prompt)
        scores: List[VLMScore] = []
        resps: List[RawResponse] = []
        for k in range(max(1, self.cfg.vlm_repeats)):
            client = self.vlm_factory(idx, self.store, k)
            resp = client.call(VLM_SYSTEM, prompt, images)
            suffix = "vlm" if self.cfg.vlm_repeats == 1 else f"vlm_rep{k}"
            self.store.save_response(idx, suffix, resp)
            resps.append(resp)
            score = VLMScore.from_dict(extract_json(resp.text))
            problems = score.validate()
            if problems:
                raise ValueError(f"第 {idx} 轮 VLM 打分越界：{problems}（原文已落盘）")
            scores.append(score)
        agg = self._aggregate(scores)
        if len(scores) > 1:
            self.store.write_json(
                self.store.round_dir(idx) / "vlm_score_repeats.json",
                {"repeats": [s.to_dict() for s in scores],
                 "aggregated": agg.to_dict(),
                 "spread": self._spread(scores)})
        return agg, resps

    @staticmethod
    def _aggregate(scores: List[VLMScore]) -> VLMScore:
        """多次打分取**中位数**（而非均值）：对单次离群打分更鲁棒。"""
        if len(scores) == 1:
            return scores[0]
        fields_ = ("weight", "time", "space", "flow", "naturalness")
        vals = {f: int(round(statistics.median(getattr(s, f) for s in scores)))
                for f in fields_}
        # 评语取中位数最接近的那一次，保持语义与分数自洽
        pivot = min(scores, key=lambda s: sum(abs(getattr(s, f) - vals[f]) for f in fields_))
        return VLMScore(comments=dict(pivot.comments), **vals)

    @staticmethod
    def _spread(scores: List[VLMScore]) -> Dict[str, float]:
        """各维度的极差，用于噪声摸底：若极差 ≥2，必须启用多次投票并上调 patience。"""
        fields_ = ("weight", "time", "space", "flow", "naturalness")
        return {f: float(max(getattr(s, f) for s in scores)
                         - min(getattr(s, f) for s in scores)) for f in fields_}

    def _ask_llm(self, idx: int) -> tuple:
        best = self.history.best
        recent = self.history.recent(2)
        last = self.history.records[-1] if self.history.records else None
        prompt = render_llm_prompt(
            idx, self.cfg.target_weight_kg, best, recent,
            last_features=(last.features if last and not last.is_reference else None),
            solver_warnings=(last.warnings if last else ()),
        )
        self.store.save_prompt(idx, "llm", prompt)
        client = self.llm_factory(idx, self.store)
        resp = client.call(LLM_SYSTEM, prompt)
        self.store.save_response(idx, "llm", resp)
        data = extract_json(resp.text)
        params = GaitParams.from_dict(data).clipped()
        return params, str(data.get("reasoning", ""))

    # ------------------------------------------------------------ 检查 --- #
    def _post_round_checks(self, record: RoundRecord) -> None:
        if record.is_reference:
            warns = self.history.check_reference_health()
            if warns:
                record.warnings.extend(warns)
                self.store.save_record(record)
                for w in warns:
                    print(w)
                # 不自动修分、不静默继续：交由人工判断后手动放行
                self.history.force_stop(StopReason.VLM_UNRELIABLE, warns[0])

    def _reference_features(self) -> Dict[str, float]:
        """A_ref 的生物力学特征。由调用方通过 ``reference_features`` 属性预置。"""
        return dict(getattr(self, "reference_features", {}) or {})

    # ------------------------------------------------------------ 持久化 -- #
    def _save_run_meta(self) -> None:
        meta = {
            "target_weight_kg": self.cfg.target_weight_kg,
            "media": self.cfg.media.to_dict(),
            "vlm_repeats": self.cfg.vlm_repeats,
            "swap_columns_probe": self.cfg.swap_columns_probe,
            "reference_motion": str(self.cfg.reference_motion or ""),
            "termination": self.history.to_dict()["termination"],
        }
        self.store.save_run_meta(meta)

    def _persist(self) -> None:
        self.history.save(self.store.run_dir / "history.json")
