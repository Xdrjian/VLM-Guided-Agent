#!/usr/bin/env python
"""Gemini API 能力探针 —— **只能在 Linux 机上运行**。

目的：一次性取回编写 API 适配层所需的全部事实，避免反复试错。
每个探针独立捕获异常，一个失败不影响其余。

用法::

    # 先渲染出两段时长不同的视频
    python scripts/probe_gemini.py \
        --video-a ../pipeline/probe/slow_r/slow_video.mp4 \
        --video-b ../pipeline/probe/fast_r/fast_video.mp4

    # 只想快速验证响应体结构，可只给一段
    python scripts/probe_gemini.py --video-a xxx.mp4 --only 1,2

输出全部写到 stdout 与 ``--out`` 指定的 json，请把两者都回传。
脚本不打印 API Key，但**响应体可能包含文件 URI**，回传前自行过目。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# 必须在任何输出（含 argparse 的 --help）之前切，否则 cp1252 控制台会直接崩
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

RESULTS: Dict[str, Any] = {}


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def probe(key: str, title: str):
    """装饰器：运行一个探针并把结果/异常收进 RESULTS。

    ``key`` 可被调用方通过关键字参数 ``key=`` 覆盖——同一探针跑多次时
    若不区分键，后一次会静默覆盖前一次的结果（第二批已踩过这个坑）。
    """
    def deco(fn):
        def wrapper(*a, **kw):
            k = kw.pop("key", key) if "key" in kw else key
            section(title)
            try:
                out = fn(*a, **kw)
                RESULTS[k] = {"ok": True, "result": out}
                return out
            except Exception as e:                      # noqa: BLE001
                print(f"[FAIL] {type(e).__name__}: {e}")
                traceback.print_exc()
                RESULTS[k] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                return None
        return wrapper
    return deco


def describe(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """把任意 SDK 对象摊平成可 json 化的结构，用于看清字段布局。"""
    if depth > max_depth:
        return f"<{type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [describe(x, depth + 1, max_depth) for x in obj[:5]]
    if isinstance(obj, dict):
        return {k: describe(v, depth + 1, max_depth) for k, v in list(obj.items())[:30]}
    if hasattr(obj, "model_dump"):                      # pydantic v2
        try:
            return describe(obj.model_dump(), depth + 1, max_depth)
        except Exception:                               # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        return {k: describe(v, depth + 1, max_depth)
                for k, v in vars(obj).items() if not k.startswith("_")}
    return f"<{type(obj).__name__}>"


# --------------------------------------------------------------------------- #
@probe("env", "① 环境：SDK 版本与关键依赖")
def probe_env() -> Dict[str, Any]:
    import google.genai as genai
    info = {"google_genai_version": getattr(genai, "__version__", "unknown"),
            "python": sys.version.split()[0]}
    try:
        from importlib.metadata import version
        for pkg in ("google-genai", "google-generativeai", "httpx"):
            try:
                info[pkg] = version(pkg)
            except Exception:                           # noqa: BLE001
                info[pkg] = "not installed"
    except Exception:                                   # noqa: BLE001
        pass
    # 顶层可用入口，用于确认 interactions / models 两套接口是否都在
    client = genai.Client()
    info["client_attrs"] = [a for a in dir(client) if not a.startswith("_")]
    for sub in ("interactions", "models", "files"):
        obj = getattr(client, sub, None)
        info[f"{sub}_methods"] = (
            [a for a in dir(obj) if not a.startswith("_")] if obj is not None else None)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


@probe("models", "② 可用模型列表（找带日期的固定版本号）")
def probe_models() -> List[Dict[str, Any]]:
    from google import genai
    client = genai.Client()
    out = []
    for m in client.models.list():
        name = getattr(m, "name", str(m))
        if "gemini" not in str(name).lower():
            continue
        out.append({
            "name": name,
            "display_name": getattr(m, "display_name", None),
            "version": getattr(m, "version", None),
            "input_token_limit": getattr(m, "input_token_limit", None),
            "output_token_limit": getattr(m, "output_token_limit", None),
            "supported_actions": getattr(m, "supported_actions", None),
        })
    for row in out:
        print(f"  {row['name']:<55} v={row['version']}")
    print(f"\n共 {len(out)} 个 gemini 模型。"
          f"\n>>> 请在上面找出 '3.8-flash' 对应的**带日期/带版本号**的固定标识 <<<")
    return out


def _upload(client, path: Path, timeout_s: float = 300.0):
    """上传并等待 ACTIVE。返回 file 对象。"""
    print(f"  上传 {path}（{path.stat().st_size / 1e6:.2f} MB）")
    f = client.files.upload(file=str(path))
    t0 = time.time()
    while True:
        f = client.files.get(name=f.name)
        state = f.state.name if hasattr(f.state, "name") else str(f.state)
        if state == "ACTIVE":
            print(f"  ACTIVE，耗时 {time.time() - t0:.1f}s")
            return f
        if state == "FAILED":
            raise RuntimeError(f"文件处理失败：{describe(f)}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"等待 ACTIVE 超时（{timeout_s}s），当前 {state}")
        time.sleep(2)


@probe("upload", "③ 上传：文件对象字段、处理耗时、过期时间")
def probe_upload(video_a: Path) -> Dict[str, Any]:
    from google import genai
    client = genai.Client()
    f = _upload(client, video_a)
    info = describe(f)
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    print("\n>>> 重点看有无 expiration_time / video_metadata / size_bytes <<<")
    return info


@probe("response", "④ 单视频调用：完整响应体结构（定位 model 回显 / usage / finish_reason）")
def probe_response(video_a: Path, model: str) -> Dict[str, Any]:
    from google import genai
    client = genai.Client()
    f = _upload(client, video_a)
    resp = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": f.uri, "mime_type": f.mime_type},
            {"type": "text", "text": "用一句话描述这段视频中人物在做什么。"},
        ],
    )
    print("--- output_text ---")
    print(getattr(resp, "output_text", "<无 output_text 属性>"))
    print("\n--- 响应体结构 ---")
    body = describe(resp, max_depth=4)
    print(json.dumps(body, ensure_ascii=False, indent=2, default=str)[:6000])
    print("\n>>> 重点确认：实际回显的 model 标识、token 用量字段名、finish_reason <<<")
    return {"output_text": getattr(resp, "output_text", None), "body": body}


@probe("two_videos", "⑤ 一次调用塞两段视频（核心设计：历史最佳 vs 本轮候选 同场对比）")
def probe_two_videos(video_a: Path, video_b: Path, model: str) -> Dict[str, Any]:
    from google import genai
    client = genai.Client()
    fa, fb = _upload(client, video_a), _upload(client, video_b)
    prompt = (
        "以下按顺序提供两段渲染视频，第一段记为【甲】，第二段记为【乙】。\n"
        "请只回答三个问题，不要做其他评价：\n"
        "1. 甲和乙，哪一段的行走速度更快？\n"
        "2. 更快的那一段，其步频大约是另一段的多少倍？给出一个数字。\n"
        "3. 你是依据画面中的什么线索做出判断的？若无法判断请直接说无法判断。"
    )
    resp = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": fa.uri, "mime_type": fa.mime_type},
            {"type": "video", "uri": fb.uri, "mime_type": fb.mime_type},
            {"type": "text", "text": prompt},
        ],
    )
    text = getattr(resp, "output_text", "")
    print(text)
    print("\n>>> 关键：模型是否真的把两段视频当成两个独立对象、顺序是否对得上 <<<")
    return {"output_text": text, "usage": describe(getattr(resp, "usage", None))}


@probe("config", "⑥ 生成参数：温度 / seed / 系统提示 / JSON 强制输出")
def probe_config(video_a: Path, model: str) -> Dict[str, Any]:
    """逐个试探参数名。不同 SDK 版本差异很大，这里穷举常见写法。"""
    from google import genai
    client = genai.Client()
    f = _upload(client, video_a)
    schema_prompt = ('只输出一个 JSON 对象，不要任何其他文字：'
                     '{"subject": str, "action": str, "confidence": int}')
    base_input = [
        {"type": "video", "uri": f.uri, "mime_type": f.mime_type},
        {"type": "text", "text": schema_prompt},
    ]

    attempts = [
        ("interactions + config dict",
         lambda: client.interactions.create(
             model=model, input=base_input,
             config={"temperature": 0.0, "seed": 0, "max_output_tokens": 512,
                     "response_mime_type": "application/json"})),
        ("interactions + 顶层 kwargs",
         lambda: client.interactions.create(
             model=model, input=base_input, temperature=0.0, seed=0)),
        ("interactions + system_instruction",
         lambda: client.interactions.create(
             model=model, input=base_input,
             system_instruction="你是一个只输出 JSON 的评估模块。")),
        ("models.generate_content + GenerateContentConfig",
         lambda: _generate_content_variant(client, model, f, schema_prompt)),
    ]

    out = {}
    for label, fn in attempts:
        try:
            r = fn()
            txt = getattr(r, "output_text", None) or getattr(r, "text", None)
            out[label] = {"ok": True, "text": (txt or "")[:400]}
            print(f"  [OK]   {label}")
            print(f"         -> {(txt or '')[:200]}")
        except Exception as e:                          # noqa: BLE001
            out[label] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
    print("\n>>> 我需要知道哪一种写法可用，适配层按可用的那种实现 <<<")
    return out


def _generate_content_variant(client, model: str, f, prompt: str):
    from google.genai import types
    return client.models.generate_content(
        model=model,
        contents=[types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type), prompt],
        config=types.GenerateContentConfig(
            temperature=0.0, seed=0, max_output_tokens=512,
            response_mime_type="application/json",
            system_instruction="你是一个只输出 JSON 的评估模块。"),
    )


@probe("aspect", "⑦ 画幅风险：3 机位横向拼接（约 5.3:1）是否被压扁到看不清")
def probe_aspect(video_a: Path, model: str) -> Dict[str, Any]:
    from google import genai
    client = genai.Client()
    f = _upload(client, video_a)
    resp = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": f.uri, "mime_type": f.mime_type},
            {"type": "text", "text":
                "这段视频把同一动作的三个机位横向拼在了一张画面里。请回答：\n"
                "1. 你能分辨出几个人物？分别是什么视角？\n"
                "2. 在最左侧那个视角中，人物膝关节的最大弯曲角度大约是多少度？\n"
                "3. 人物双手拿着什么？它是平放的还是倾斜的？\n"
                "4. 画面清晰度是否足以回答上述问题？若不足请明确说明。"},
        ],
    )
    text = getattr(resp, "output_text", "")
    print(text)
    print("\n>>> 若模型分不清三个机位或看不清膝角，就得改成分机位分别上传 <<<")
    return {"output_text": text}


def _video_part(types, f, fps: Optional[float] = None):
    """构造视频 Part。``fps`` 非空时附带采样率。

    不能用 ``Part.from_uri(video_metadata=...)``——该 classmethod 不收这个参数
    （SDK 2.30.1 实测 TypeError）；但 ``Part`` 本身有 ``video_metadata`` 字段，
    故直接构造。默认采样率为 1fps，5 秒视频只有约 5 帧，膝屈曲峰值会被整帧跳过。
    """
    fd = types.FileData(file_uri=f.uri, mime_type=f.mime_type)
    if fps is None:
        return types.Part(file_data=fd)
    return types.Part(file_data=fd, video_metadata=types.VideoMetadata(fps=float(fps)))


@probe("fps_control", "⑬ 采样率 × 温度：哪种写法能同时拿到这两样")
def probe_fps_control(video_a: Path, model: str, fps: float = 12.0) -> Dict[str, Any]:
    """第一批证实 ``interactions`` 设不了温度，第二批证实默认采样率是 1fps。
    二者必须同时满足，否则要么看不见膝角、要么实验不可复现。本探针找出交集。

    客观验证手段：对比 ``prompt_tokens_details`` 里 VIDEO 的 token 数——
    若采样率真的生效，5 秒视频的 VIDEO token 应从约 285 升到数千量级。
    """
    from google import genai
    from google.genai import types
    client = genai.Client()
    f = _upload(client, video_a)
    q = "这段视频里人物左脚触地几次？请数出次数并给出每次的大致时间。"

    def show(label: str, resp, kind: str) -> Dict[str, Any]:
        if kind == "gc":
            text = getattr(resp, "text", "") or ""
            u = describe(getattr(resp, "usage_metadata", None), max_depth=4)
            vid = None
            for d in (u.get("prompt_tokens_details") or []):
                if str(d.get("modality", "")).upper().endswith("VIDEO"):
                    vid = d.get("token_count")
        else:
            text = getattr(resp, "output_text", "") or ""
            u = describe(getattr(resp, "usage", None), max_depth=4)
            vid = u.get("total_input_tokens")
        print(f"  [OK]   {label}")
        print(f"         VIDEO tokens = {vid}（1fps 基线约 285）")
        print(f"         {text[:160]}")
        return {"ok": True, "video_tokens": vid, "text": text, "usage": u}

    out: Dict[str, Any] = {}
    trials = [
        ("A. generate_content + Part(video_metadata=fps) + temperature", "gc",
         lambda: client.models.generate_content(
             model=model, contents=[_video_part(types, f, fps), q],
             config=types.GenerateContentConfig(temperature=0.0, seed=0))),
        ("B. generate_content 不设 fps（对照基线）", "gc",
         lambda: client.models.generate_content(
             model=model, contents=[_video_part(types, f, None), q],
             config=types.GenerateContentConfig(temperature=0.0, seed=0))),
        ("C. interactions + processing.fps（无温度）", "it",
         lambda: client.interactions.create(
             model=model, input=[
                 {"type": "video", "uri": f.uri, "mime_type": f.mime_type,
                  "processing": {"type": "static", "fps": fps}},
                 {"type": "text", "text": q}])),
        ("D. interactions + processing.fps + extra_body 温度", "it",
         lambda: client.interactions.create(
             model=model, input=[
                 {"type": "video", "uri": f.uri, "mime_type": f.mime_type,
                  "processing": {"type": "static", "fps": fps}},
                 {"type": "text", "text": q}],
             extra_body={"generation_config": {"temperature": 0.0, "seed": 0}})),
    ]
    for label, kind, fn in trials:
        try:
            out[label] = show(label, fn(), kind)
        except Exception as e:                          # noqa: BLE001
            out[label] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
    print("\n>>> 优先采用 A；A 不行再看 D；两者都不行则必须在"
          "可复现性与可见性之间做取舍 <<<")
    return out


@probe("cleanup", "⑧ 文件清理：能否主动删除（证据包不应依赖云端留存）")
def probe_cleanup() -> Dict[str, Any]:
    from google import genai
    client = genai.Client()
    listed = list(client.files.list())
    info = {"count_before": len(listed)}
    print(f"  当前云端有 {len(listed)} 个文件")
    if listed:
        name = listed[0].name
        try:
            client.files.delete(name=name)
            info["delete_ok"] = True
            print(f"  删除 {name} 成功")
        except Exception as e:                          # noqa: BLE001
            info["delete_ok"] = False
            info["delete_error"] = str(e)
            print(f"  删除失败：{e}")
    return info


# ------------------------- 第二批：第一批结果暴露出来的新未知 ------------- #
@probe("gc_body", "⑨ generate_content 的完整响应体（适配层实际要走这条路）")
def probe_gc_body(video_a: Path, model: str) -> Dict[str, Any]:
    """第一批已证实：``interactions.create`` **无法设温度/seed**，只能用
    ``models.generate_content``。但当时只看了它的文本输出，没看响应体结构，
    而 ``RawResponse`` 要记录 model 回显 / usage / finish_reason。"""
    from google import genai
    from google.genai import types
    client = genai.Client()
    f = _upload(client, video_a)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type),
                  '只输出一个 JSON：{"action": str, "confidence": int}'],
        config=types.GenerateContentConfig(
            temperature=0.0, seed=0, max_output_tokens=512,
            response_mime_type="application/json",
            system_instruction="你是一个只输出 JSON 的评估模块。"),
    )
    print("--- text ---")
    print(getattr(resp, "text", "<无 text 属性>"))
    body = describe(resp, max_depth=6)
    print("\n--- 响应体结构 ---")
    print(json.dumps(body, ensure_ascii=False, indent=2, default=str)[:6000])
    print("\n>>> 重点：model_version / usage_metadata 字段名 / candidates[0].finish_reason <<<")
    return {"text": getattr(resp, "text", None), "body": body}


@probe("sampling", "⑩ 视频采样率：5 秒视频仅 ~270 input tokens，模型到底看到几帧？")
def probe_sampling(video_a: Path, model: str) -> Dict[str, Any]:
    """第一批实测：5s 视频仅 297 input tokens，暗示采样率可能低到 ~1fps。
    若属实，118 帧的动作只有 5 帧被看到，还不如宫格图的 8 帧——
    这会动摇"视频信息量更大"这一前提。逐个试探可否调高采样率。"""
    from google import genai
    from google.genai import types
    client = genai.Client()
    f = _upload(client, video_a)
    q = "这段视频里人物指到了几次地？请数出左脚触地的次数，并给出每次的大致时间。"

    def run(label: str, part, cfg):
        r = client.models.generate_content(model=model, contents=[part, q], config=cfg)
        u = describe(getattr(r, "usage_metadata", None), max_depth=3)
        print(f"  [{label}] tokens={json.dumps(u, ensure_ascii=False, default=str)}")
        print(f"           {(getattr(r, 'text', '') or '')[:200]}")
        return {"usage": u, "text": getattr(r, "text", None)}

    out = {}
    base = types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)
    for label, maker in [
        ("默认", lambda: (base, types.GenerateContentConfig(temperature=0.0))),
        ("media_resolution=HIGH", lambda: (
            base, types.GenerateContentConfig(
                temperature=0.0,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH))),
        ("video_metadata fps=8", lambda: (
            types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type,
                                video_metadata=types.VideoMetadata(fps=8.0)),
            types.GenerateContentConfig(temperature=0.0))),
        ("video_metadata fps=24", lambda: (
            types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type,
                                video_metadata=types.VideoMetadata(fps=24.0)),
            types.GenerateContentConfig(temperature=0.0))),
    ]:
        try:
            part, cfg = maker()
            out[label] = run(label, part, cfg)
        except Exception as e:                          # noqa: BLE001
            out[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{label}] FAIL {type(e).__name__}: {e}")
    print("\n>>> 重点：input tokens 是否随 fps 上升；数脚步是否变准 <<<")
    return out


@probe("knee", "⑪ 膝角相对分辨（核心参数 knee_flex_scale 能否被看见）")
def probe_knee(base_v: Path, deep_v: Path, model: str, label: str,
               key: str = "knee", fps: Optional[float] = None) -> Dict[str, Any]:
    """第一批已显示：模型**绝对**膝角估计大错（说 30~40°，实测峰值 98.8°）。
    但我们并不需要绝对值——只需要它能**分辨两个候选谁屈膝更深**。
    这才是闭环能否被驱动的真正前提。

    第二批实测：1 fps 下模型把方向答反了（答【甲】、真值【乙】）。
    传入 ``fps`` 可复测提高采样率后是否修复。"""
    from google import genai
    from google.genai import types
    client = genai.Client()
    fa, fb = _upload(client, base_v), _upload(client, deep_v)
    prompt = (
        "以下按顺序提供两段渲染视频，第一段记为【甲】，第二段记为【乙】。\n"
        "两段动作的唯一区别在于膝关节的弯曲程度。请只回答三个问题：\n"
        "1. 哪一段的屈膝更深？\n"
        "2. 两者的膝关节屈曲峰值大约相差多少度？给出一个数字。\n"
        "3. 你是从哪个机位、哪个时刻看出来的？若无法分辨请直接说无法分辨。"
    )
    resp = client.models.generate_content(
        model=model,
        contents=[_video_part(types, fa, fps), _video_part(types, fb, fps), prompt],
        config=types.GenerateContentConfig(temperature=0.0, seed=0),
    )
    text = getattr(resp, "text", "") or ""
    u = describe(getattr(resp, "usage_metadata", None), max_depth=4)
    print(f"  [{label}]  fps={fps or '默认(1)'}")
    print(text)
    print(f"\n  tokens: {json.dumps(u.get('prompt_tokens_details'), ensure_ascii=False, default=str)}")
    print("  >>> 甲=基准，乙=屈膝更深。模型必须答【乙】 <<<")
    return {"label": label, "fps": fps, "output_text": text, "usage": u}


@probe("noise", "⑫ 正式 VLM prompt：JSON 合规性 + 三次重复的打分噪声")
def probe_noise(video_a: Path, video_b: Path, model: str,
                repeats: int = 3) -> Dict[str, Any]:
    """用**流程里真正会用的那份 prompt**跑，而非另写一份——否则摸到的噪声
    不是实验里的噪声。结果直接决定 `vlm_repeats` / `improvement_margin` / `patience`。"""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from motionedit.prompts import VideoSpec, VLM_SYSTEM, render_vlm_prompt, extract_json
    from motionedit.records import SCORE_RANGES

    from google import genai
    from google.genai import types
    client = genai.Client()
    fa, fb = _upload(client, video_a), _upload(client, video_b)
    prompt = render_vlm_prompt(1, 30.0, None, VideoSpec())
    print(f"  prompt 长度 {len(prompt)} 字符")

    rows: List[Dict[str, Any]] = []
    for k in range(repeats):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_uri(file_uri=fa.uri, mime_type=fa.mime_type),
                          types.Part.from_uri(file_uri=fb.uri, mime_type=fb.mime_type),
                          prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0, seed=k, max_output_tokens=2048,
                    response_mime_type="application/json",
                    system_instruction=VLM_SYSTEM),
            )
            text = getattr(resp, "text", "") or ""
            data = extract_json(text)
            bad = [f"{k2}={data.get(k2)}" for k2, (lo, hi) in SCORE_RANGES.items()
                   if not isinstance(data.get(k2), int) or not lo <= data[k2] <= hi]
            rows.append({"seed": k, "parsed": True, "scores":
                         {k2: data.get(k2) for k2 in SCORE_RANGES},
                         "out_of_range": bad,
                         "has_comments": isinstance(data.get("comments"), dict),
                         "usage": describe(getattr(resp, "usage_metadata", None), max_depth=3),
                         "comments": data.get("comments")})
            print(f"  第{k + 1}次  {rows[-1]['scores']}  越界={bad or '无'}")
        except Exception as e:                          # noqa: BLE001
            rows.append({"seed": k, "parsed": False, "error": f"{type(e).__name__}: {e}"})
            print(f"  第{k + 1}次  FAIL {type(e).__name__}: {e}")

    ok = [r for r in rows if r.get("parsed")]
    spread = {}
    for dim in SCORE_RANGES:
        vals = [r["scores"][dim] for r in ok
                if isinstance(r["scores"].get(dim), int)]
        spread[dim] = (max(vals) - min(vals)) if len(vals) > 1 else None
    print(f"\n  各维度极差：{spread}")
    print("  >>> 全部 ≤1 → vlm_repeats=1；任一 ≥2 → 必须 vlm_repeats=3 取中位数 <<<")
    return {"rows": rows, "spread": spread}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini API 能力探针（仅限 Linux 机）")
    ap.add_argument("--video-a", type=Path, required=True, help="较慢的那段")
    ap.add_argument("--video-b", type=Path, default=None, help="较快的那段")
    ap.add_argument("--knee-base", type=Path, default=None,
                    help="⑪ 用：knee_flex_scale=1.0 的视频")
    ap.add_argument("--knee-mid", type=Path, default=None,
                    help="⑪ 用：knee_flex_scale=1.25（现实难度）")
    ap.add_argument("--knee-deep", type=Path, default=None,
                    help="⑪ 用：knee_flex_scale=1.6（简单难度，用于定位失败原因）")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="⑬⑭ 用的视频采样率（默认值来自网页端实测表现良好的 12）")
    ap.add_argument("--out", type=Path, default=Path("probe_gemini_result.json"))
    ap.add_argument("--only", default=None,
                    help="只跑指定编号，逗号分隔，如 1,2,4")
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(",")} if args.only else None

    def want(n: int) -> bool:
        return only is None or n in only

    if want(1):
        probe_env()
    if want(2):
        probe_models()
    if want(3):
        probe_upload(args.video_a)
    if want(4):
        probe_response(args.video_a, args.model)
    if want(5):
        if args.video_b:
            probe_two_videos(args.video_a, args.video_b, args.model)
        else:
            print("\n[跳过 ⑤] 未提供 --video-b")
    if want(6):
        probe_config(args.video_a, args.model)
    if want(7):
        probe_aspect(args.video_a, args.model)
    if want(8):
        probe_cleanup()
    if want(9):
        probe_gc_body(args.video_a, args.model)
    if want(10):
        probe_sampling(args.video_a, args.model)
    if want(11):
        if args.knee_base and args.knee_mid:
            probe_knee(args.knee_base, args.knee_mid, args.model,
                       "现实难度 1.00 vs 1.25", key="knee_mid_1fps")
        else:
            print("\n[跳过 ⑪-a] 未提供 --knee-base / --knee-mid")
        if args.knee_base and args.knee_deep:
            probe_knee(args.knee_base, args.knee_deep, args.model,
                       "简单难度 1.00 vs 1.60", key="knee_deep_1fps")
        else:
            print("[跳过 ⑪-b] 未提供 --knee-deep")
    if want(12):
        if args.video_b:
            probe_noise(args.video_a, args.video_b, args.model)
        else:
            print("\n[跳过 ⑫] 未提供 --video-b")
    if want(13):
        probe_fps_control(args.video_a, args.model, args.fps)
    if want(14):
        # 提高采样率后复测膝角分辨——这是 1fps 失败的直接对照组
        if args.knee_base and args.knee_deep:
            probe_knee(args.knee_base, args.knee_deep, args.model,
                       f"简单难度 1.00 vs 1.60 @ {args.fps}fps",
                       key=f"knee_deep_{args.fps:g}fps", fps=args.fps)
        if args.knee_base and args.knee_mid:
            probe_knee(args.knee_base, args.knee_mid, args.model,
                       f"现实难度 1.00 vs 1.25 @ {args.fps}fps",
                       key=f"knee_mid_{args.fps:g}fps", fps=args.fps)
        if not args.knee_base:
            print("\n[跳过 ⑳] 未提供膝角素材")

    args.out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    section("汇总")
    for k, v in RESULTS.items():
        print(f"  {k:<12} {'OK' if v.get('ok') else 'FAIL'}")
    print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
