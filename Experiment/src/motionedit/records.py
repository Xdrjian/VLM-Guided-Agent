"""第 4 层 · 闭环：单轮记录与证据包（Evidence Package）。

设计动机（2026-09 讨论结论）：
网页端 LLM/VLM 存在隐藏系统提示、图像二次压缩、输出后处理、同名模型静默更新等
不可控因素，**不具备科学可复现性**。因此正式实验一律走 API，并要求每一轮把
"输入 → 请求参数 → 原始返回 → 解析结果" 全量落盘，作为论文可复现性声明的依据。

证据包目录结构（每轮一个子目录）::

    <run_dir>/
        run_meta.json              # 本次 run 的全局配置与环境指纹
        history.json               # 所有轮次记录的汇总（可重建）
        rounds/
            round_000/
                params.json            # 本轮 7 参数（绝对值）
                features.json          # 求解器输出的生物力学特征
                vlm_prompt.txt         # 提交给 VLM 的 system + user 原文
                vlm_request.json       # 模型版本号 / temperature / seed / 图像清单
                vlm_response.json      # API 原始返回体全文（含 model 回显、usage）
                vlm_score.json         # 解析后的结构化打分
                llm_prompt.txt         # 提交给 LLM 的原文
                llm_request.json
                llm_response.json      # 原始返回体
                record.json            # 本轮汇总记录
                images/                # 提交给 VLM 的原始图像（未经任何二次压缩）
                video/                 # 留底视频（给人看，不一定提交给 VLM）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .params import GaitParams

__all__ = [
    "SCORE_RANGES",
    "DEFAULT_SCORE_WEIGHTS",
    "VLMScore",
    "RawResponse",
    "RoundRecord",
    "EvidenceStore",
]


#: 各评判维度的合法分档（workflow.md §4.2）。
#: 主观性强的 Effort 维度用窄分档（1-5），相对客观的自然度用宽分档（1-10）。
SCORE_RANGES: Dict[str, tuple] = {
    "weight": (1, 5),
    "time": (1, 5),
    "space": (1, 5),
    "flow": (1, 5),
    "naturalness": (1, 10),
}

#: 加权总分的权重（workflow.md §4.2）：
#: Weight 权重显著大于其余三个 Effort 维度；自然度权重 = 四项 Effort 之和。
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "weight": 4.0,
    "time": 1.0,
    "space": 1.0,
    "flow": 1.0,
    "naturalness": 7.0,
}

SCORE_FIELDS = tuple(SCORE_RANGES.keys())


# --------------------------------------------------------------------------- #
@dataclass
class VLMScore:
    """VLM 对一个候选解的分项打分。

    注意：加权总分 :meth:`total` **仅用于外层循环的收敛判断**，
    绝不传给 LLM（workflow.md §4.2）——传给 LLM 的永远是分项 + 评语。
    """

    weight: int
    time: int
    space: int
    flow: int
    naturalness: int
    comments: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def scores(self) -> Dict[str, int]:
        return {k: int(getattr(self, k)) for k in SCORE_FIELDS}

    def validate(self) -> List[str]:
        """返回越界项的描述列表；空列表表示合法。"""
        problems: List[str] = []
        for k, (lo, hi) in SCORE_RANGES.items():
            v = getattr(self, k)
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                problems.append(f"{k}={v!r} 超出合法区间 [{lo}, {hi}]")
        return problems

    def normalized(self) -> Dict[str, float]:
        """把各维度线性归一化到 [0, 1]，消除 1-5 与 1-10 分档的量纲差异。"""
        out = {}
        for k, (lo, hi) in SCORE_RANGES.items():
            v = float(getattr(self, k))
            out[k] = (v - lo) / float(hi - lo)
        return out

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """加权总分，归一化到 [0, 1]。仅供内部收敛判断使用。"""
        w = weights or DEFAULT_SCORE_WEIGHTS
        norm = self.normalized()
        num = sum(w[k] * norm[k] for k in SCORE_FIELDS)
        den = sum(w[k] for k in SCORE_FIELDS)
        return num / den if den > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = self.scores()
        d["comments"] = dict(self.comments)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLMScore":
        """从 VLM 返回的 JSON 构造，容忍多余字段与字符串型数字。"""
        vals = {}
        for k in SCORE_FIELDS:
            if k not in d:
                raise KeyError(f"VLM 返回缺少必需字段 '{k}'：{sorted(d.keys())}")
            vals[k] = int(round(float(d[k])))
        comments = d.get("comments") or {}
        if not isinstance(comments, dict):
            comments = {"_raw": str(comments)}
        return cls(comments={str(k): str(v) for k, v in comments.items()}, **vals)

    def summary(self) -> str:
        s = self.scores()
        return (
            f"W={s['weight']}/5 T={s['time']}/5 S={s['space']}/5 "
            f"F={s['flow']}/5 Nat={s['naturalness']}/10 "
            f"| total={self.total():.4f}"
        )


# --------------------------------------------------------------------------- #
@dataclass
class RawResponse:
    """一次模型调用的完整证据。

    ``text`` / ``body`` 必须是**未经任何加工的原文**；解析失败时也要落盘，
    否则无法事后复盘 "是模型没按格式输出，还是我们解析错了"。
    """

    text: str
    body: Dict[str, Any] = field(default_factory=dict)
    #: API 返回体里回显的实际模型标识——用于检测"繁忙降级"，必须记录
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # "api" | "manual" | "replay"
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
@dataclass
class RoundRecord:
    """一轮外层循环的完整记录。

    第 0 轮是特殊轮次：``params`` 为恒等参数、不经过求解器、
    ``is_reference=True``（workflow.md §4.4）。
    """

    round_index: int
    params: GaitParams
    features: Dict[str, float] = field(default_factory=dict)
    score: Optional[VLMScore] = None
    #: 本轮是否被接受为新的历史最佳（由 History 判定后回写）
    accepted: bool = False
    #: 被回滚的轮次：动作产物丢弃，但记录保留作为"反面参考"（§4.5）
    rolled_back: bool = False
    #: 第 0 轮标记：A_ref 原始动捕数据，非求解器输出
    is_reference: bool = False
    #: LLM 给出本轮参数时的理由原文
    llm_reasoning: str = ""
    #: 求解器执行状态（converged / rom_violation / exec_failure 等）
    solver_status: str = ""
    #: 各类产物的相对路径（图像、视频、动作 pkl）
    artifacts: Dict[str, str] = field(default_factory=dict)
    #: 健康检查告警（如第0轮自然度过低、求解器执行失效）
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    @property
    def total_score(self) -> float:
        return self.score.total() if self.score is not None else float("-inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "params": self.params.to_dict(),
            "features": dict(self.features),
            "score": self.score.to_dict() if self.score else None,
            "total_score": self.total_score if self.score else None,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "is_reference": self.is_reference,
            "llm_reasoning": self.llm_reasoning,
            "solver_status": self.solver_status,
            "artifacts": dict(self.artifacts),
            "warnings": list(self.warnings),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoundRecord":
        score = VLMScore.from_dict(d["score"]) if d.get("score") else None
        return cls(
            round_index=int(d["round_index"]),
            params=GaitParams.from_dict(d["params"]),
            features=dict(d.get("features") or {}),
            score=score,
            accepted=bool(d.get("accepted", False)),
            rolled_back=bool(d.get("rolled_back", False)),
            is_reference=bool(d.get("is_reference", False)),
            llm_reasoning=str(d.get("llm_reasoning", "")),
            solver_status=str(d.get("solver_status", "")),
            artifacts=dict(d.get("artifacts") or {}),
            warnings=list(d.get("warnings") or []),
            timestamp=float(d.get("timestamp", 0.0)),
        )

    def summary(self) -> str:
        tag = "REF" if self.is_reference else ("BEST" if self.accepted else
                                              ("ROLLED-BACK" if self.rolled_back else "cand"))
        sc = self.score.summary() if self.score else "(未评分)"
        return f"[round {self.round_index:03d} {tag}] {sc}"


# --------------------------------------------------------------------------- #
class EvidenceStore:
    """证据包读写。所有路径均由本类统一约定，避免散落在各处的字符串拼接。"""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.rounds_dir = self.run_dir / "rounds"

    # -------------------------------------------------------------- 路径 -- #
    def round_dir(self, index: int) -> Path:
        return self.rounds_dir / f"round_{index:03d}"

    def images_dir(self, index: int) -> Path:
        return self.round_dir(index) / "images"

    def video_dir(self, index: int) -> Path:
        return self.round_dir(index) / "video"

    def ensure(self, index: int) -> Path:
        d = self.round_dir(index)
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "video").mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path | str) -> str:
        """转为相对 run_dir 的 posix 路径，保证记录跨机器可读。"""
        p = Path(path)
        try:
            return p.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    # -------------------------------------------------------------- IO --- #
    @staticmethod
    def write_json(path: Path, obj: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def write_text(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # ------------------------------------------------------------ 语义层 -- #
    def save_run_meta(self, meta: Dict[str, Any]) -> Path:
        return self.write_json(self.run_dir / "run_meta.json", meta)

    def save_prompt(self, index: int, kind: str, prompt: str) -> Path:
        """``kind`` ∈ {"vlm", "llm"}。落盘的是 system+user 拼接后的完整原文。"""
        self.ensure(index)
        return self.write_text(self.round_dir(index) / f"{kind}_prompt.txt", prompt)

    def save_response(self, index: int, kind: str, resp: RawResponse) -> Path:
        self.ensure(index)
        self.write_json(self.round_dir(index) / f"{kind}_request.json", resp.request)
        return self.write_json(self.round_dir(index) / f"{kind}_response.json", resp.to_dict())

    def save_record(self, rec: RoundRecord) -> Path:
        self.ensure(rec.round_index)
        d = self.round_dir(rec.round_index)
        self.write_json(d / "params.json", rec.params.to_dict())
        self.write_json(d / "features.json", rec.features)
        if rec.score is not None:
            self.write_json(d / "vlm_score.json", rec.score.to_dict())
        return self.write_json(d / "record.json", rec.to_dict())

    def load_record(self, index: int) -> RoundRecord:
        return RoundRecord.from_dict(self.read_json(self.round_dir(index) / "record.json"))

    def list_rounds(self) -> List[int]:
        if not self.rounds_dir.exists():
            return []
        out = []
        for p in self.rounds_dir.iterdir():
            if p.is_dir() and p.name.startswith("round_") and (p / "record.json").exists():
                try:
                    out.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(out)
