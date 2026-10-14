#!/usr/bin/env python
"""检视一次外层循环运行的证据包。

把分数、评语、参数、指令跟踪、LLM 理由摊平成可直接阅读/粘贴的形式，
并做**信号诊断**：哪些维度在整个运行中从未变化、总分实际被谁驱动。

一个维度若全程恒定，它对收敛判断的贡献就是零——分数看起来有五维，
实际可能只有一维在起作用。这件事从总分上看不出来，必须逐维摊开。

用法::

    python scripts/inspect_run.py --run-dir ../pipeline/runs/exp02
    python scripts/inspect_run.py --run-dir ../pipeline/runs/exp02 --prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionedit.records import (DEFAULT_SCORE_WEIGHTS, SCORE_FIELDS,  # noqa: E402
                                SCORE_RANGES, VLMScore)


def banner(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _wrap(s: str, width: int = 68, indent: str = " " * 12) -> str:
    s = " ".join(str(s).split())
    out, line = [], ""
    for ch in s:
        line += ch
        if len(line) >= width:
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return f"\n{indent}".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="检视外层循环证据包")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prompts", action="store_true", help="附带 prompt 原文长度与开头")
    ap.add_argument("--max-comment", type=int, default=200)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rounds_dir = run / "rounds"
    if not rounds_dir.exists():
        print(f"找不到 {rounds_dir}")
        return 1
    dirs = sorted(d for d in rounds_dir.iterdir() if d.is_dir())

    meta = _load(run / "run_meta.json") or {}
    banner("运行元信息")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    records: List[Dict[str, Any]] = []
    for d in dirs:
        rec = _load(d / "record.json")
        if rec is None:
            continue
        rec["_dir"] = d
        rec["_solve"] = _load(d / "solve_report.json")
        rec["_llm"] = _load(d / "llm_response.json")
        rec["_vlm"] = _load(d / "vlm_response.json")
        records.append(rec)

    # ------------------------------------------------------------------ #
    banner("逐轮明细")
    for rec in records:
        tag = ("REF" if rec["is_reference"] else
               ("BEST" if rec["accepted"] else
                ("ROLLED-BACK" if rec["rolled_back"] else "cand")))
        print(f"\n--- round {rec['round_index']:03d}  [{tag}]  "
              f"total={rec.get('total_score')}  status={rec.get('solver_status')}")

        p = rec["params"]
        nonid = {k: v for k, v in p.items()
                 if abs(v - (1.0 if "scale" in k else 0.0)) > 1e-9}
        print(f"  参数（仅列非恒等项）：{json.dumps(nonid, ensure_ascii=False) or '全部恒等'}")

        sc = rec.get("score")
        if sc:
            print(f"  打分：{' '.join(f'{k}={sc[k]}' for k in SCORE_FIELDS)}")
            comp = sc.get("comparison") or {}
            if comp:
                cells = " ".join(f"{k}={comp.get(k, '?')}" for k in SCORE_FIELDS)
                print(f"  对比判定：{cells}")
            for k in SCORE_FIELDS:
                c = (sc.get("comments") or {}).get(k, "")
                if c:
                    print(f"      {k:<12} {_wrap(c[:args.max_comment])}")

        sr = rec.get("_solve")
        if sr:
            tr = sr.get("tracking") or {}
            if tr:
                worst = max(tr.items(), key=lambda kv: kv[1]["rel_error"])
                print(f"  指令跟踪：最大误差 {worst[0]}={worst[1]['rel_error'] * 100:.1f}% "
                      f"（期望 {worst[1]['expected']:.3f} / 实际 {worst[1]['actual']:.3f}）")
            if sr.get("violations"):
                for v in sr["violations"]:
                    print(f"      ⚠ {v}")

        llm = rec.get("_llm")
        if llm:
            try:
                txt = json.loads(llm["text"]) if isinstance(llm.get("text"), str) else {}
            except Exception:                           # noqa: BLE001
                txt = {}
            reason = txt.get("reasoning", "")
            if reason:
                print(f"  LLM 理由：{_wrap(reason[:600], indent=' ' * 12)}")

        vlm = rec.get("_vlm")
        if vlm and vlm.get("usage"):
            u = vlm["usage"]
            print(f"  VLM 用量：{json.dumps(u, ensure_ascii=False)}")
        if vlm:
            print(f"  VLM 回显模型：{vlm.get('model')}  来源：{vlm.get('source')}")

    # ------------------------------------------------------------------ #
    banner("信号诊断：总分到底被谁驱动")
    scored = [r for r in records if r.get("score")]
    if len(scored) < 2:
        print("有效评分少于 2 轮，无法诊断。")
        return 0

    print(f"\n{'维度':<14}{'权重':>6}{'取值范围':>10}  各轮取值           极差  "
          f"对总分的最大影响")
    dead: List[str] = []
    for k in SCORE_FIELDS:
        vals = [r["score"][k] for r in scored]
        lo, hi = SCORE_RANGES[k]
        w = DEFAULT_SCORE_WEIGHTS[k]
        span = max(vals) - min(vals)
        # 该维度若跑满整个分档，能让总分移动多少
        reach = w / sum(DEFAULT_SCORE_WEIGHTS.values())
        used = reach * span / (hi - lo)
        if span == 0:
            dead.append(k)
        print(f"{k:<14}{w:>6.0f}{f'{lo}-{hi}':>10}  {str(vals):<18}{span:>4}  "
              f"实际移动 {used:.4f} / 满量程 {reach:.4f}")

    if dead:
        print(f"\n  ⚠ 以下维度全程恒定、对收敛判断贡献为零：{dead}")
        alive = [k for k in SCORE_FIELDS if k not in dead]
        wa = sum(DEFAULT_SCORE_WEIGHTS[k] for k in alive)
        wt = sum(DEFAULT_SCORE_WEIGHTS.values())
        print(f"  ⚠ 实际在起作用的只有 {alive}，占总权重 {wa}/{wt} = {wa / wt:.1%}")
        if alive == ["naturalness"]:
            print("  ⚠ 只剩自然度在动。它系统性地给'被编辑过的动作'扣分，"
                  "因此每个候选都必然低于 A_ref——循环结构上无法接受任何解。")

    # 评语是否在描述差异却不反映到分数上
    banner("评语 vs 分数：是否看得见却不敢给分")
    for k in SCORE_FIELDS:
        vals = [r["score"][k] for r in scored]
        if max(vals) != min(vals):
            continue
        texts = [(r["round_index"], (r["score"].get("comments") or {}).get(k, ""))
                 for r in scored]
        uniq = {t for _, t in texts if t}
        print(f"\n  {k}（分数恒为 {vals[0]}，但给出了 {len(uniq)} 种不同评语）：")
        for idx, t in texts:
            if t:
                print(f"      r{idx:03d}: {_wrap(t[:args.max_comment], indent=' ' * 12)}")

    if args.prompts:
        banner("prompt 原文摘要")
        for rec in records:
            for kind in ("vlm", "llm"):
                p = rec["_dir"] / f"{kind}_prompt.txt"
                if p.exists():
                    txt = p.read_text(encoding="utf-8")
                    print(f"\n--- round {rec['round_index']:03d} {kind} "
                          f"（{len(txt)} 字符）---")
                    print(txt[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
