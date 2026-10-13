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
    "GeminiVLMClient",
    "GeminiLLMClient",
    "GEMINI_VIDEO_FPS",
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
                 media: Sequence[Path] = ()):
        self.answer_path = Path(answer_path)
        self.prompt_path = Path(prompt_path)
        self.media = [Path(p) for p in media]
        super().__init__(self._instructions())

    def _instructions(self) -> str:
        lines = [
            "",
            "=" * 72,
            "【手动模式挂起】需要人工完成一次模型调用，然后重跑本脚本即可继续。",
            "=" * 72,
            f"1. 打开 prompt 原文：{self.prompt_path}",
        ]
        if self.media:
            lines.append(f"2. 按顺序提交以下 {len(self.media)} 份素材：")
            for i, p in enumerate(self.media, 1):
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
             media: Sequence[Path] = ()) -> RawResponse: ...


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
             media: Sequence[Path] = ()) -> RawResponse:
        full = f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n"
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(full, encoding="utf-8")

        if not self.answer_path.exists():
            raise PendingManualInput(self.answer_path, self.prompt_path, media)

        text = self.answer_path.read_text(encoding="utf-8")
        return RawResponse(
            text=text,
            body={"manual_answer_file": str(self.answer_path)},
            model=f"manual:{self.name}",
            usage={},
            request={
                "mode": "manual",
                "prompt_file": str(self.prompt_path),
                "media": [str(p) for p in media],
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
                     media: Sequence[Path]) -> Dict[str, Any]:
        return {
            "mode": "api",
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "media": [str(p) for p in media],
            "system_chars": len(system),
            "user_chars": len(user),
            **self.extra,
        }

    # ------------------------------------------------------------------ #
    def call(self, system: str, user: str,
             media: Sequence[Path] = ()) -> RawResponse:
        meta = self.request_meta(system, user, media)
        t0 = time.time()
        body = self._invoke(system, user, media)
        text, echoed, usage = self._unpack(body)
        return RawResponse(
            text=text, body=body, model=echoed or self.model,
            usage=usage, request=meta, source="api",
            elapsed_s=time.time() - t0,
        )

    # -------------------------------------------------- 厂商适配层（待补）-- #
    def _invoke(self, system: str, user: str,
                media: Sequence[Path]) -> Dict[str, Any]:
        raise NotImplementedError(
            "API 适配层尚未实现。A0 探路阶段确定模型选型后，在此实现具体厂商的调用；"
            "务必把完整返回体（含 model 回显、usage、finish_reason）原样返回。"
        )

    @staticmethod
    def _unpack(body: Dict[str, Any]) -> tuple:
        """从返回体中取出 (正文, 实际模型标识, 用量)。厂商适配层可覆写。"""
        return body.get("text", ""), body.get("model", ""), body.get("usage", {})


class ApiVLMClient(_ApiBase):
    """API VLM。素材必须以**原始文件**提交，不做任何客户端二次压缩。"""


class ApiLLMClient(_ApiBase):
    """API LLM。纯文本调用。"""


# --------------------------------------------------------------------------- #
#: Gemini 默认视频采样率。**不要改回默认值**——
#: 实测默认为 1fps（5 秒视频仅 285 VIDEO token），膝屈曲峰值出现在摆动相的
#: 极短窗口里会被整帧跳过，导致模型把"谁屈膝更深"答反。12fps 下恢复正常。
GEMINI_VIDEO_FPS = 12.0

_MIME = {".mp4": "video/mp4", ".gif": "image/gif", ".png": "image/png",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webm": "video/webm"}


class _GeminiBase(_ApiBase):
    """Gemini 适配层（google-genai SDK）。

    走 ``models.generate_content`` 而非 ``interactions.create``：后者在 SDK 2.30.1 上
    **设不了 temperature / seed**（明确报 TypeError），温度不可控即实验不可复现。

    视频采样率必须显式指定，且不能用 ``Part.from_uri(video_metadata=...)``——
    该 classmethod 不收这个参数；但 ``Part`` 本身有该字段，故直接构造 Part。

    ``delete_uploads`` 默认开启：云端文件会过期，证据的权威副本是本地文件，
    上传件只是临时搬运，用完即删以免堆积。本地路径与 sha256 都会记入 request。
    """

    def __init__(self, model: str, *, video_fps: float = GEMINI_VIDEO_FPS,
                 json_mode: bool = True, delete_uploads: bool = True,
                 upload_timeout_s: float = 300.0, retry_backoff_s: float = 4.0,
                 **kw: Any) -> None:
        super().__init__(model, **kw)
        self.video_fps = float(video_fps)
        self.json_mode = bool(json_mode)
        self.delete_uploads = bool(delete_uploads)
        self.upload_timeout_s = float(upload_timeout_s)
        self.retry_backoff_s = float(retry_backoff_s)
        self._client: Any = None
        self._uploads: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _sdk(self):
        if self._client is None:
            from google import genai            # 惰性导入：无网络的机器也能导入本模块
            self._client = genai.Client()
        return self._client

    def _upload(self, path: Path) -> Any:
        import hashlib
        client = self._sdk()
        raw = path.read_bytes()
        f = client.files.upload(file=str(path))
        t0 = time.time()
        while True:
            f = client.files.get(name=f.name)
            state = f.state.name if hasattr(f.state, "name") else str(f.state)
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError(f"Gemini 文件处理失败：{path}")
            if time.time() - t0 > self.upload_timeout_s:
                raise TimeoutError(f"等待文件 ACTIVE 超时（{self.upload_timeout_s}s）：{path}")
            time.sleep(2.0)
        self._uploads.append({
            "local_path": str(path),
            "remote_name": f.name,
            "uri": f.uri,
            "mime_type": f.mime_type,
            "size_bytes": getattr(f, "size_bytes", None),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        return f

    def _parts(self, media: Sequence[Path]) -> List[Any]:
        from google.genai import types
        out = []
        for p in media:
            p = Path(p)
            f = self._upload(p)
            fd = types.FileData(file_uri=f.uri,
                                mime_type=f.mime_type or _MIME.get(p.suffix.lower(), ""))
            if str(f.mime_type or "").startswith("video"):
                out.append(types.Part(
                    file_data=fd,
                    video_metadata=types.VideoMetadata(fps=self.video_fps)))
            else:
                out.append(types.Part(file_data=fd))
        return out

    def _cleanup(self) -> None:
        if not (self.delete_uploads and self._uploads):
            return
        client = self._sdk()
        for rec in self._uploads:
            try:
                client.files.delete(name=rec["remote_name"])
                rec["deleted"] = True
            except Exception as e:                      # noqa: BLE001
                rec["deleted"] = False
                rec["delete_error"] = str(e)

    # ------------------------------------------------------------------ #
    def _invoke(self, system: str, user: str,
                media: Sequence[Path]) -> Dict[str, Any]:
        from google.genai import types
        self._uploads = []
        cfg_kw: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "system_instruction": system,
        }
        if self.seed is not None:
            cfg_kw["seed"] = self.seed
        if self.json_mode:
            cfg_kw["response_mime_type"] = "application/json"

        last: Optional[Exception] = None
        try:
            contents = self._parts(media) + [user]
            for attempt in range(self.max_retries + 1):
                try:
                    resp = self._sdk().models.generate_content(
                        model=self.model, contents=contents,
                        config=types.GenerateContentConfig(**cfg_kw))
                    return self._to_body(resp)
                except Exception as e:                  # noqa: BLE001
                    last = e
                    if attempt >= self.max_retries:
                        break
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
        finally:
            self._cleanup()
        raise RuntimeError(f"Gemini 调用失败（已重试 {self.max_retries} 次）：{last}") from last

    def _to_body(self, resp: Any) -> Dict[str, Any]:
        """把 SDK 响应对象转成可落盘的 dict，并提取复现所需的关键字段。"""
        dumped: Any
        try:
            dumped = resp.model_dump(mode="json")
        except Exception:                               # noqa: BLE001
            dumped = str(resp)
        cand = (getattr(resp, "candidates", None) or [None])[0]
        return {
            "text": getattr(resp, "text", "") or "",
            "model": getattr(resp, "model_version", "") or self.model,
            "response_id": getattr(resp, "response_id", None),
            "finish_reason": str(getattr(cand, "finish_reason", "")) if cand else "",
            "service_tier": self._service_tier(resp),
            "usage": self._usage(resp),
            "uploads": list(self._uploads),
            "raw": dumped,
        }

    @staticmethod
    def _service_tier(resp: Any) -> str:
        http = getattr(resp, "sdk_http_response", None)
        headers = getattr(http, "headers", None) or {}
        try:
            return str(headers.get("x-gemini-service-tier", ""))
        except Exception:                               # noqa: BLE001
            return ""

    @staticmethod
    def _usage(resp: Any) -> Dict[str, Any]:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            return {}
        out = {
            "prompt_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            # 思考 token 实测远大于输出 token（13 输出 vs 488 思考），计费必须算上
            "thought_tokens": getattr(u, "thoughts_token_count", None),
            "total_tokens": getattr(u, "total_token_count", None),
        }
        details = getattr(u, "prompt_tokens_details", None) or []
        out["by_modality"] = {
            str(getattr(d, "modality", "")).split(".")[-1]: getattr(d, "token_count", None)
            for d in details
        }
        return out

    @staticmethod
    def _unpack(body: Dict[str, Any]) -> tuple:
        return body.get("text", ""), body.get("model", ""), body.get("usage", {})


class GeminiVLMClient(_GeminiBase):
    """Gemini VLM：视频（或图像）+ 文本 → 结构化 JSON 打分。"""


class GeminiLLMClient(_GeminiBase):
    """Gemini LLM：纯文本调参。不传素材，故采样率设置不生效。"""

    def __init__(self, model: str, **kw: Any) -> None:
        kw.setdefault("delete_uploads", False)
        super().__init__(model, **kw)


# --------------------------------------------------------------------------- #
def parse_or_raise(resp: RawResponse) -> Dict[str, Any]:
    """解析返回体中的 JSON；失败时抛出但保证原文已在 ``resp`` 中可追溯。"""
    return extract_json(resp.text)
