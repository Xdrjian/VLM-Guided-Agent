"""第 4 层 · 闭环：历史管理与终止条件（workflow.md §4.1 / §4.3 / §4.5）。

核心不变量（务必不要在别处绕过 History 直接改 records）：

1. **调参基底永远是"当前已接受的历史最佳解"**，而不是紧邻的上一轮。
   被回滚的尝试只作为反面参考影响方向判断，不作为参数延拓起点。
2. **VLM 打分的对比锚点同样是历史最佳解**，与调参基底保持同一参照系，
   避免 VLM 评语中的"相比上一轮是否改善"被 LLM 误读（§4.2 末段）。
3. 回滚只丢弃动作产物，**记录本身永远保留**（§4.5）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .records import EvidenceStore, RoundRecord

__all__ = ["StopReason", "TerminationConfig", "History"]


# --------------------------------------------------------------------------- #
class StopReason:
    """终止原因常量。``None`` 表示尚未终止。"""

    NO_IMPROVEMENT = "no_improvement"     # Score 连续若干轮未超过历史最佳
    SCORE_TARGET = "score_target"         # Score 达到预设上限
    MAX_ROUNDS = "max_rounds"             # 兜底保护
    SOLVER_FAILURE = "solver_failure"     # 求解器执行失效（§4.4）
    VLM_UNRELIABLE = "vlm_unreliable"     # 第0轮健康检查未通过（§4.4）


@dataclass
class TerminationConfig:
    """终止阈值。

    workflow.md §4.1 明确指出这三项"当前无法预先确定，留待第一次实验实测后设定"。
    这里给出的是**占位默认值**，A1 阶段噪声摸底完成后必须回头重设：
    ``patience`` 必须大于 VLM 打分噪声所能造成的虚假停滞长度，
    ``score_target`` 必须高于噪声上限，否则会被一次运气好的打分提前触发。
    """

    #: 连续多少轮未超过历史最佳即判定收敛  🔴 待实测
    patience: int = 4
    #: 归一化加权总分达到该值即停止（VLMScore.total 的量纲，[0,1]） 🔴 待实测
    score_target: float = 0.90
    #: 最大轮数兜底  🔴 待实测
    max_rounds: int = 20
    #: 第 0 轮自然度健康检查下限（原始分，1-10 档）。低于即告警（§4.4）
    ref_naturalness_floor: int = 8
    #: 新解需超过历史最佳多少才算"提升"。设为 0 即严格大于；
    #: 噪声摸底后应设为略大于噪声标准差，避免把噪声当改进
    improvement_margin: float = 0.0


# --------------------------------------------------------------------------- #
class History:
    """轮次记录的唯一权威容器。"""

    def __init__(self, cfg: Optional[TerminationConfig] = None):
        self.cfg = cfg or TerminationConfig()
        self.records: List[RoundRecord] = []
        self._best_index: Optional[int] = None
        self._stall: int = 0
        self.stop_reason: Optional[str] = None
        self.stop_detail: str = ""

    # ------------------------------------------------------------ 查询 -- #
    def __len__(self) -> int:
        return len(self.records)

    @property
    def next_round_index(self) -> int:
        return len(self.records)

    @property
    def best(self) -> Optional[RoundRecord]:
        """当前已接受的历史最佳解。第 0 轮（A_ref）也可以是最佳解。"""
        if self._best_index is None:
            return None
        return self.records[self._best_index]

    @property
    def reference(self) -> Optional[RoundRecord]:
        """第 0 轮记录（A_ref 本身）。"""
        for r in self.records:
            if r.is_reference:
                return r
        return None

    def recent(self, n: int = 2, exclude_best: bool = True) -> List[RoundRecord]:
        """最近 n 轮记录（含被回滚的失败尝试），按轮次升序。

        ``exclude_best=True`` 时剔除历史最佳本身——它会以"锚点"身份
        单独出现在 prompt 里，不应在"最近记录"中重复占用上下文。
        """
        best = self.best
        pool = [r for r in self.records
                if not r.is_reference and not (exclude_best and best is not None
                                               and r.round_index == best.round_index)]
        return pool[-n:] if n > 0 else []

    def stall_count(self) -> int:
        return self._stall

    # ------------------------------------------------------------ 写入 -- #
    def add(self, record: RoundRecord) -> RoundRecord:
        """登记一轮结果，并就地判定接受 / 回滚。

        返回同一个 ``record`` 对象（``accepted`` / ``rolled_back`` 已被回写）。
        """
        if record.round_index != len(self.records):
            raise ValueError(
                f"轮次编号不连续：期望 {len(self.records)}，收到 {record.round_index}"
            )
        self.records.append(record)

        if record.score is None:
            # 未评分（例如求解器失败）→ 既不接受也不参与停滞计数
            record.accepted = False
            record.rolled_back = True
            return record

        best = self.best
        if best is None:
            record.accepted = True
            self._best_index = record.round_index
            self._stall = 0
        elif record.total_score > best.total_score + self.cfg.improvement_margin:
            record.accepted = True
            self._best_index = record.round_index
            self._stall = 0
        else:
            record.accepted = False
            record.rolled_back = True
            self._stall += 1
        return record

    # ------------------------------------------------------ 终止与健康 -- #
    def check_reference_health(self) -> List[str]:
        """第 0 轮健康检查（§4.4）。

        A_ref 是动捕真值，自然度理应是所有候选中最高的。若偏低，
        大概率不是动作不自然，而是 **VLM prompt / 渲染管线 / 模型能力** 有问题。
        此处只产出告警，**不自动修正分数、不静默继续**。
        """
        ref = self.reference
        if ref is None or ref.score is None:
            return []
        warnings: List[str] = []
        nat = ref.score.naturalness
        if nat < self.cfg.ref_naturalness_floor:
            warnings.append(
                f"[VLM 可靠性告警] 第0轮 A_ref（动捕真值）自然度仅 {nat}/10，"
                f"低于阈值 {self.cfg.ref_naturalness_floor}。这通常指向 VLM prompt 设计、"
                f"渲染质量或模型能力问题，而非 A_ref 本身不自然。"
                f"请人工核查后再决定是否继续。"
            )
        return warnings

    def should_stop(self) -> Optional[str]:
        """返回终止原因；``None`` 表示继续。"""
        if self.stop_reason is not None:
            return self.stop_reason
        best = self.best
        if best is not None and best.total_score >= self.cfg.score_target:
            self._stop(StopReason.SCORE_TARGET,
                       f"历史最佳总分 {best.total_score:.4f} ≥ 目标 {self.cfg.score_target}")
        elif self._stall >= self.cfg.patience:
            self._stop(StopReason.NO_IMPROVEMENT,
                       f"连续 {self._stall} 轮未超过历史最佳（patience={self.cfg.patience}）")
        elif len(self.records) >= self.cfg.max_rounds:
            self._stop(StopReason.MAX_ROUNDS,
                       f"已达最大轮数 {self.cfg.max_rounds}")
        return self.stop_reason

    def force_stop(self, reason: str, detail: str = "") -> None:
        self._stop(reason, detail)

    def _stop(self, reason: str, detail: str) -> None:
        self.stop_reason = reason
        self.stop_detail = detail

    # ------------------------------------------------------------ 持久化 #
    def to_dict(self) -> Dict:
        return {
            "records": [r.to_dict() for r in self.records],
            "best_index": self._best_index,
            "stall": self._stall,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "termination": {
                "patience": self.cfg.patience,
                "score_target": self.cfg.score_target,
                "max_rounds": self.cfg.max_rounds,
                "ref_naturalness_floor": self.cfg.ref_naturalness_floor,
                "improvement_margin": self.cfg.improvement_margin,
            },
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str, cfg: Optional[TerminationConfig] = None) -> "History":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        term = d.get("termination") or {}
        h = cls(cfg or TerminationConfig(**{k: v for k, v in term.items()
                                            if k in TerminationConfig.__dataclass_fields__}))
        h.records = [RoundRecord.from_dict(r) for r in d.get("records", [])]
        h._best_index = d.get("best_index")
        h._stall = int(d.get("stall", 0))
        h.stop_reason = d.get("stop_reason")
        h.stop_detail = d.get("stop_detail", "")
        return h

    @classmethod
    def from_store(cls, store: EvidenceStore,
                   cfg: Optional[TerminationConfig] = None) -> "History":
        """从证据包目录重建历史（history.json 损坏时的兜底路径）。"""
        h = cls(cfg)
        for idx in store.list_rounds():
            rec = store.load_record(idx)
            # 重放 add 的判定逻辑，保证 best/stall 与记录自洽
            h.records.append(rec)
            if rec.accepted:
                h._best_index = rec.round_index
                h._stall = 0
            elif rec.score is not None:
                h._stall += 1
        return h

    # ------------------------------------------------------------ 报告 -- #
    def summary(self) -> str:
        lines = [f"历史轮次：{len(self.records)}  停滞：{self._stall}/{self.cfg.patience}"]
        for r in self.records:
            lines.append("  " + r.summary())
        best = self.best
        lines.append(f"当前最佳：{'round %03d' % best.round_index if best else '无'}"
                     + (f"  total={best.total_score:.4f}" if best else ""))
        if self.stop_reason:
            lines.append(f"终止：{self.stop_reason} — {self.stop_detail}")
        return "\n".join(lines)
