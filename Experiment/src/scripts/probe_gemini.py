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
    """装饰器：运行一个探针并把结果/异常收进 RESULTS。"""
    def deco(fn):
        def wrapper(*a, **kw):
            section(title)
            try:
                out = fn(*a, **kw)
                RESULTS[key] = {"ok": True, "result": out}
                return out
            except Exception as e:                      # noqa: BLE001
                print(f"[FAIL] {type(e).__name__}: {e}")
                traceback.print_exc()
                RESULTS[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
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


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini API 能力探针（仅限 Linux 机）")
    ap.add_argument("--video-a", type=Path, required=True, help="较慢的那段")
    ap.add_argument("--video-b", type=Path, default=None, help="较快的那段")
    ap.add_argument("--model", default="gemini-3.8-flash")
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

    args.out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    section("汇总")
    for k, v in RESULTS.items():
        print(f"  {k:<12} {'OK' if v.get('ok') else 'FAIL'}")
    print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
