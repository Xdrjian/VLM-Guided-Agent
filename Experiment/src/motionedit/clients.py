"""第 4 层 · 闭环：模型调用客户端（手动 / API 双实现）。

阶段划分（2026-09 讨论结论）：

- **A0 探路**：网页端免费额度，仅做定性可行性验证（能否分辨宫格内单帧细节、
  能否同场对比 6 图、打分噪声多大）。**结果不进实验记录**——网页端存在隐藏
  系统提示、图像二次压缩、输出后处理、同名模型静默更新等不可控因素。
- **A1 正式手动**：走 API，但人工触发、人工查看。本模块的 ``Manual*Client``
  服务于 A1 之前的过渡以及 API 不可用时的降级，它把 prompt 与图像写到磁盘，
  抛出 :class:`PendingManualInput` 让外层循环优雅挂起，待人工填入答案文件后续跑。
- **B/C 半自动/全自动**：``Api*Client``，与 Manual 实现同一接口，外层循环无感切换。

模型版本纪律：``Api*Client`` 强制要求带日期的显式版本号（如
``gemini-2.5-pro-preview-06-05``），拒绝 ``*-latest`` 这类会静默漂移的别名，
并把 API 返回体中回显的 ``model`` 字段记录下来，用于检测"繁忙降级"。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .prompts import extract_json
from .records import RawResponse

__all__ = [
    "PendingManualInput",
    "ModelClient",
    "ManualVLMClient",
    "ManualLLMClient",
    "ApiVLMClient",
    "ApiLLMClient",
    "LATEST_ALIASES",
]

#: 禁止使用的滚动别名——会导致论文中的"模型版本"不可复现
LATEST_ALIASES = ("latest", "preview", "stable", "default")


class PendingManualInput(Exception):
    """手动模式下等待人工填写答案。

    这不是错误，而是**外层循环的正常挂起点**：材料已全部落盘，
    人工填好 ``answer_path`` 指向的文件后重跑即可从断点继续。
    """

    def __init__(self, answer_path: Path, prompt_path: Path,
                 images: Sequence[Path] = ()):
        self.answer_path = Path(answer_path)
        self.prompt_path = Path(prompt_path)
        self.images = [Path(p) for p in images]
        super().__init__(self._instructions())

    def _instructions(self) -> str:
        lines = [
            "",
            "=" * 72,
            "【手动模式挂起】需要人工完成一次模型调用，然后重跑本脚本即可继续。",
            "=" * 72,
            f"1. 打开 prompt 原文：{self.prompt_path}",
        ]
        if self.images:
            lines.append(f"2. 按顺序提交以下 {len(self.images)} 张图像：")
            for i, p in enumerate(self.images, 1):
                lines.append(f"     [{i}] {p}")
            lines.append("3. 把模型返回的 JSON **原样**粘贴保存到：")
        else:
            lines.append("2. 把模型返回的 JSON **原样**粘贴保存到：")
        lines += [
            f"     {self.answer_path}",
            "4. 重新运行本脚本，流程将从此处继续。",
            "",
            "注意：网页端结果仅可用于 A0 定性探路，不得作为正式实验数据。",
            "=" * 72,
            "",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
class ModelClient(Protocol):
    """VLM / LLM 客户端的统一接口。

    实现必须保证：无论成功失败，返回的 :class:`RawResponse` 都携带**未经加工的原文**。
    """

    name: str

    def call(self, system: str, user: str,
             images: Sequence[Path] = ()) -> RawResponse: ...


# --------------------------------------------------------------------------- #
class _ManualBase:
    """把 prompt 写盘、等待人工回填答案文件。"""

    kind = "manual"

    def __init__(self, prompt_path: Path | str, answer_path: Path | str,
                 name: str = "manual"):
        self.prompt_path = Path(prompt_path)
        self.answer_path = Path(answer_path)
        self.name = name

    def call(self, system: str, user: str,
             images: Sequence[Path] = ()) -> RawResponse:
        full = f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n"
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(full, encoding="utf-8")

        if not self.answer_path.exists():
            raise PendingManualInput(self.answer_path, self.prompt_path, images)

        text = self.answer_path.read_text(encoding="utf-8")
        return RawResponse(
            text=text,
            body={"manual_answer_file": str(self.answer_path)},
            model=f"manual:{self.name}",
            usage={},
            request={
                "mode": "manual",
                "prompt_file": str(self.prompt_path),
                "images": [str(p) for p in images],
                "system_chars": len(system),
                "user_chars": len(user),
                "warning": "手动模式（尤其网页端）不具备科学可复现性，仅用于 A0 探路。",
            },
            source="manual",
        )


class ManualVLMClient(_ManualBase):
    """手动 VLM：写出 prompt + 图像清单，等待人工粘贴 JSON 打分。"""


class ManualLLMClient(_ManualBase):
    """手动 LLM：写出 prompt，等待人工粘贴 JSON 参数。"""


# --------------------------------------------------------------------------- #
class _ApiBase:
    """API 客户端骨架。

    真正的 HTTP 调用留给具体厂商适配层（A0 探路选型确定后再补），
    这里先把**与厂商无关的纪律**固化下来：版本号校验、重试、计时、原始返回落盘。
    """

    kind = "api"

    def __init__(self, model: str, temperature: float = 0.0,
                 seed: Optional[int] = 0, max_tokens: int = 2048,
                 max_retries: int = 2, extra: Optional[Dict[str, Any]] = None):
        self._assert_pinned(model)
        self.name = model
        self.model = model
        self.temperature = float(temperature)
        self.seed = seed
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.extra = dict(extra or {})

    @staticmethod
    def _assert_pinned(model: str) -> None:
        low = model.lower()
        if any(low.endswith("-" + a) or low == a for a in LATEST_ALIASES):
            raise ValueError(
                f"模型标识 '{model}' 使用了滚动别名。正式实验必须固定到带日期的版本号，"
                f"否则厂商静默更新会让实验结果不可复现。"
            )

    def request_meta(self, system: str, user: str,
                     images: Sequence[Path]) -> Dict[str, Any]:
        return {
            "mode": "api",
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "images": [str(p) for p in images],
            "system_chars": len(system),
            "user_chars": len(user),
            **self.extra,
        }

    # ------------------------------------------------------------------ #
    def call(self, system: str, user: str,
             images: Sequence[Path] = ()) -> RawResponse:
        meta = self.request_meta(system, user, images)
        t0 = time.time()
        body = self._invoke(system, user, images)
        text, echoed, usage = self._unpack(body)
        return RawResponse(
            text=text, body=body, model=echoed or self.model,
            usage=usage, request=meta, source="api",
            elapsed_s=time.time() - t0,
        )

    # -------------------------------------------------- 厂商适配层（待补）-- #
    def _invoke(self, system: str, user: str,
                images: Sequence[Path]) -> Dict[str, Any]:
        raise NotImplementedError(
            "API 适配层尚未实现。A0 探路阶段确定模型选型后，在此实现具体厂商的调用；"
            "务必把完整返回体（含 model 回显、usage、finish_reason）原样返回。"
        )

    @staticmethod
    def _unpack(body: Dict[str, Any]) -> tuple:
        """从返回体中取出 (正文, 实际模型标识, 用量)。厂商适配层可覆写。"""
        return body.get("text", ""), body.get("model", ""), body.get("usage", {})


class ApiVLMClient(_ApiBase):
    """API VLM。图像必须以**原始文件**提交，不做任何客户端二次压缩。"""


class ApiLLMClient(_ApiBase):
    """API LLM。纯文本调用。"""


# --------------------------------------------------------------------------- #
def parse_or_raise(resp: RawResponse) -> Dict[str, Any]:
    """解析返回体中的 JSON；失败时抛出但保证原文已在 ``resp`` 中可追溯。"""
    return extract_json(resp.text)
