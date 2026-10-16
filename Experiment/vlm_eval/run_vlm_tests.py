#!/usr/bin/env python
"""一键运行 VLM / prompt 的判官性质测试。

四项测试，全部用**流程中真正使用的那份 prompt**（``render_vlm_prompt``），
而不是另写一套——否则测到的不是实验里的那个判官。

================================================================  ========================
测试                                                              失败意味着
================================================================  ========================
T1 同一性   同一段视频提交两次                                     判官会凭空捏造差异
T2 位置不变性  同一对素材正反两种顺序各问一次                        判官受提交顺序影响
T3 单调排序  单一参数的低/中/高三档，两两比较                        判官的偏好不可传递
T4 自然度对齐  真值 / 大幅编辑 / 人为破坏，两两比较                   自然度维度不可用
================================================================  ========================

默认**不给锚点分数**（``--with-anchor`` 可开）：锚点分数本身是混淆因素，
会把"判官的固有性质"和"锚定效应"混在一起。

用法（Linux，需环境变量里的 Gemini Key）::

    cd Experiment/vlm_eval
    python make_materials.py            # 先生成素材
    python run_vlm_tests.py             # 跑全部四项
    python run_vlm_tests.py --only 1 2 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))

from motionedit.clients import GEMINI_VIDEO_FPS, GeminiVLMClient   # noqa: E402
from motionedit.prompts import (VLM_SYSTEM, VideoSpec,             # noqa: E402
                                extract_json, render_vlm_prompt)
from motionedit.records import SCORE_FIELDS, VLMScore              # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def banner(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


@dataclass
class Call:
    """一次成对评判的结果。"""
    first: str
    second: str
    repeat: int
    score: Optional[VLMScore] = None
    error: str = ""
    raw: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    def verdict(self, dim: str) -> str:
        return (self.score.comparison.get(dim, "missing")
                if self.score else "error")

    def magnitude(self, dim: str) -> str:
        return (self.score.magnitude.get(dim, "missing")
                if self.score else "error")

    def winner(self, dim: str) -> Optional[str]:
        """该维度上胜出的素材名；打平/看不清/出错返回 None。"""
        v = self.verdict(dim)
        return self.first if v == "first" else (self.second if v == "second" else None)


class Judge:
    """对真实 prompt 的薄封装，顺带落盘全部原始返回。"""

    def __init__(self, args, out_dir: Path):
        self.args = args
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec = VideoSpec()
        self.calls: List[Call] = []
        self.anchor = None
        if args.with_anchor:
            from motionedit.params import GaitParams
            from motionedit.records import RoundRecord
            self.anchor = RoundRecord(
                round_index=0, params=GaitParams(), is_reference=True,
                score=VLMScore(weight=2, time=3, space=3, flow=3, naturalness=7))

    def ask(self, materials: Dict[str, Path], a: str, b: str, k: int) -> Call:
        c = Call(first=a, second=b, repeat=k)
        prompt = render_vlm_prompt(1, self.args.weight, self.anchor,
                                   self.spec, paired=True)
        try:
            cli = GeminiVLMClient(self.args.model, temperature=self.args.temperature,
                                  seed=k, max_tokens=self.args.max_tokens,
                                  max_retries=self.args.max_retries,
                                  video_fps=self.args.fps)
            resp = cli.call(VLM_SYSTEM, prompt, [materials[a], materials[b]])
            c.raw, c.usage = resp.text, resp.usage
            sc = VLMScore.from_dict(extract_json(resp.text))
            problems = sc.validate()
            if problems:
                c.error = f"取值非法：{problems}"
            else:
                c.score = sc
        except Exception as e:                           # noqa: BLE001
            c.error = f"{type(e).__name__}: {e}"
        self.calls.append(c)
        tag = "ERR " if c.error else "    "
        line = " ".join(f"{d[:4]}={c.verdict(d)[:4]}/{c.magnitude(d)[:4]}"
                        for d in SCORE_FIELDS)
        print(f"  {tag}[{a} | {b}] #{k}  {line}")
        if c.error:
            print(f"       {c.error}")
        self._dump()
        return c

    def _dump(self) -> None:
        rows = [{"first": c.first, "second": c.second, "repeat": c.repeat,
                 "error": c.error, "raw": c.raw, "usage": c.usage,
                 "score": c.score.to_dict() if c.score else None}
                for c in self.calls]
        (self.out / "raw_calls.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
def t1_identity(j: Judge, mats: Dict[str, Path], name: str, n: int) -> dict:
    banner("T1 同一性：同一段视频提交两次，不得捏造差异")
    invented: List[str] = []
    for k in range(n):
        c = j.ask(mats, name, name, k)
        if c.score is None:
            continue
        for d in SCORE_FIELDS:
            if c.verdict(d) in ("first", "second") or c.magnitude(d) != "none":
                invented.append(f"#{k} {d}: {c.verdict(d)}/{c.magnitude(d)}")
    ok = not invented
    print(f"\n  结论：{PASS if ok else FAIL}"
          f"  —— {'全部判为打平' if ok else f'捏造了 {len(invented)} 处差异'}")
    for s in invented:
        print(f"      {s}")
    return {"status": PASS if ok else FAIL, "invented": invented}


def t2_position(j: Judge, mats: Dict[str, Path], a: str, b: str, n: int) -> dict:
    banner(f"T2 位置不变性：{a} 与 {b} 正反两序各问 {n} 次")
    _MIRROR = {"first": "second", "second": "first",
               "tie": "tie", "insufficient": "insufficient"}
    rows, consistent, flipped, degraded = [], 0, 0, 0
    for k in range(n):
        fwd = j.ask(mats, a, b, k)
        rev = j.ask(mats, b, a, k)
        if fwd.score is None or rev.score is None:
            continue
        for d in SCORE_FIELDS:
            vf, vr = fwd.verdict(d), rev.verdict(d)
            if _MIRROR.get(vf) == vr:
                consistent += 1
                tag = "一致"
            elif vf in ("first", "second") and vr in ("first", "second"):
                flipped += 1
                tag = "**位置偏差**"
            else:
                degraded += 1
                tag = "强弱不一致"
            rows.append((k, d, vf, vr, tag))
    total = consistent + flipped + degraded
    rate = consistent / total if total else 0.0
    print(f"\n  {'轮':>3} {'维度':<12}{'正序':<14}{'反序':<14}判定")
    for k, d, vf, vr, tag in rows:
        print(f"  {k:>3} {d:<12}{vf:<14}{vr:<14}{tag}")
    ok = flipped == 0 and rate >= 0.8
    print(f"\n  结论：{PASS if ok else FAIL}  一致 {consistent}/{total}"
          f"（{rate:.0%}）| 位置偏差 {flipped} | 强弱不一致 {degraded}")
    if flipped:
        print("      位置偏差 = 正反两序都说同一个位置更好，"
              "说明判断受顺序而非内容驱动")
    return {"status": PASS if ok else FAIL, "consistent": consistent,
            "flipped": flipped, "degraded": degraded, "rate": rate}


def _cycle(wins: Dict[Tuple[str, str], Optional[str]], items: List[str]) -> Optional[str]:
    """三元组中是否存在循环偏好。返回循环描述，无循环返回 None。"""
    edge = {}
    for (x, y), w in wins.items():
        if w is None:
            continue
        loser = y if w == x else x
        edge[(w, loser)] = True
    for x, y, z in ((items[0], items[1], items[2]),
                    (items[0], items[2], items[1])):
        if edge.get((x, y)) and edge.get((y, z)) and edge.get((z, x)):
            return f"{x} > {y} > {z} > {x}"
    return None


def _pairwise(j: Judge, mats, items: List[str], n: int
              ) -> Dict[Tuple[str, str], List[Call]]:
    """两两比较。重复时**交替提交顺序**，使 T3/T4 对位置偏差免疫——

    否则一个"永远说第一段更好"的坏判官只要赶上期望顺序就能蒙混过关。
    ``Call.winner`` 返回的是素材名而非位置，故计票不受顺序影响。
    """
    out: Dict[Tuple[str, str], List[Call]] = {}
    for x, y in combinations(items, 2):
        calls = []
        for k in range(n):
            a, b = (y, x) if k % 2 else (x, y)
            calls.append(j.ask(mats, a, b, k))
        out[(x, y)] = calls
    return out


def _majority(calls: List[Call], dim: str) -> Optional[str]:
    """多次重复中的多数胜者；平局或全为 tie 时返回 None。"""
    tally: Dict[str, int] = {}
    for c in calls:
        w = c.winner(dim)
        if w:
            tally[w] = tally.get(w, 0) + 1
    if not tally:
        return None
    top = max(tally.values())
    winners = [k for k, v in tally.items() if v == top]
    return winners[0] if len(winners) == 1 else None


def _short(items: List[str]) -> Dict[str, str]:
    """去掉共同前缀的短标签，否则 knee_low / knee_mid 截断后长得一模一样。"""
    if len(items) < 2:
        return {i: i for i in items}
    pre = 0
    while all(len(i) > pre + 1 and i[pre] == items[0][pre] for i in items):
        pre += 1
    return {i: (i[pre:] or i) for i in items}


def t3_monotonic(j: Judge, mats: Dict[str, Path], items: List[str],
                 n: int, truth: Dict[str, float]) -> dict:
    banner(f"T3 单调排序：{' < '.join(items)}（按膝屈曲峰值递增），两两比较")
    pairs = _pairwise(j, mats, items, n)
    result: Dict[str, Any] = {}
    sh = _short(items)
    heads = [f"{sh[a]}v{sh[b]}" for a, b in pairs]
    print(f"\n  {'维度':<13}" + "".join(f"{h:<12}" for h in heads) + "循环  与真值方向")
    any_cycle = False
    for d in SCORE_FIELDS:
        wins = {k: _majority(v, d) for k, v in pairs.items()}
        cyc = _cycle(wins, items)
        any_cycle = any_cycle or bool(cyc)
        cells = [str(wins[k] or "tie")[:11] for k in pairs]
        # 与真值方向一致 = 膝屈曲更大的那一档胜出
        agree = sum(1 for (x, y), w in wins.items()
                    if w is not None and (truth[w] > truth[x if w == y else y]))
        decided = sum(1 for w in wins.values() if w is not None)
        print(f"  {d:<13}" + "".join(f"{c:<12}" for c in cells)
              + f"{'有!' if cyc else '无':<6}{agree}/{decided}")
        result[d] = {"wins": {f"{a}|{b}": w for (a, b), w in wins.items()},
                     "cycle": cyc, "decided": decided,
                     "agree_with_truth": f"{agree}/{decided}"}
    # "全是打平"也不会出现循环，但那说明判官根本没给出判定，不能算通过
    decided_total = sum(v["decided"] for v in result.values())
    if any_cycle:
        status = FAIL
        note = "存在循环排序，偏好不可传递"
    elif decided_total == 0:
        status = WARN
        note = "未出现循环，但全部维度都判为打平/看不清——该测试没有取到信号"
    else:
        status = PASS
        note = f"未出现循环排序（共 {decided_total} 次有效判定）"
    print(f"\n  结论：{status}  —— {note}")
    print("      与真值方向一致率仅供参考：VLM 不必认同"
          "「屈膝更深 = 更像负重」，但绝不可自相矛盾。")
    return {"status": status, "decided": decided_total, "per_dim": result}


def t4_naturalness(j: Judge, mats: Dict[str, Path], items: List[str], n: int) -> dict:
    banner(f"T4 自然度对齐：期望 {items[0]} > {items[1]} > {items[2]}")
    pairs = _pairwise(j, mats, items, n)
    d = "naturalness"
    wins = {k: _majority(v, d) for k, v in pairs.items()}
    cyc = _cycle(wins, items)
    expected = {(items[0], items[1]): items[0],
                (items[1], items[2]): items[1],
                (items[0], items[2]): items[0]}
    hits = [k for k, w in wins.items() if w == expected[k]]
    print(f"\n  {'对比':<28}{'判定胜者':<14}{'期望':<14}是否一致")
    for k in pairs:
        print(f"  {k[0] + ' vs ' + k[1]:<28}{str(wins[k] or 'tie'):<14}"
              f"{expected[k]:<14}{'是' if wins[k] == expected[k] else '否'}")
    abs_scores: Dict[str, List[int]] = {}
    for (x, y), calls in pairs.items():
        for c in calls:
            if c.score:
                abs_scores.setdefault(y, []).append(c.score.naturalness)
    if abs_scores:
        print("\n  绝对自然度分（注：配对格式下绝对分是给「动作 B」即后一段的）")
        for k, v in abs_scores.items():
            print(f"      {k:<14}{v}")
    ok = (not cyc) and len(hits) == 3
    status = PASS if ok else (WARN if not cyc and len(hits) >= 2 else FAIL)
    print(f"\n  结论：{status}  —— 符合期望 {len(hits)}/3"
          f"，{'无循环' if not cyc else f'存在循环 {cyc}'}")
    return {"status": status, "hits": len(hits), "cycle": cyc,
            "wins": {f"{a}|{b}": w for (a, b), w in wins.items()},
            "abs_naturalness": abs_scores}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="VLM 判官性质测试")
    ap.add_argument("--materials", type=Path, default=HERE / "materials")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="只跑指定测试编号，如 --only 1 2")
    ap.add_argument("--repeats", type=int, default=2,
                    help="每个比较重复几次。调大可把噪声与偏差分开")
    ap.add_argument("--weight", type=float, default=30.0)
    ap.add_argument("--with-anchor", action="store_true",
                    help="附带锚点分数（默认不给，避免锚定效应污染判官性质）")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--fps", type=float, default=GEMINI_VIDEO_FPS)
    args = ap.parse_args()

    root = Path(args.materials)
    idx_path = root / "index.json"
    if not idx_path.exists():
        print(f"找不到 {idx_path}，请先运行 make_materials.py")
        return 1
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    mats = {k: root / v["video"] for k, v in index.items()}
    missing = [k for k, p in mats.items() if not p.exists()]
    if missing:
        print(f"素材缺失：{missing}")
        return 1

    want = set(args.only) if args.only else {1, 2, 3, 4}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / stamp
    j = Judge(args, out)

    banner("素材")
    print(f"{'名称':<14}{'帧数':>6}{'时长s':>8}{'左膝峰值':>10}{'大小MB':>9}")
    for k, v in index.items():
        print(f"{k:<14}{v['num_frames']:6d}{v['duration_s']:8.2f}"
              f"{v['knee_flex_left_max']:10.1f}{v['size_mb']:9.2f}")
    print(f"\n模型 {args.model} @ {args.fps:g}fps | 温度 {args.temperature} | "
          f"重复 {args.repeats} | 锚点 {'给' if args.with_anchor else '不给'}")

    report: Dict[str, Any] = {"stamp": stamp, "model": args.model,
                              "repeats": args.repeats,
                              "with_anchor": args.with_anchor}
    knees = ["knee_low", "knee_mid", "knee_high"]
    truth = {k: index[k]["knee_flex_left_max"] for k in knees if k in index}

    if 1 in want:
        report["T1"] = t1_identity(j, mats, "a_ref", args.repeats)
    if 2 in want:
        report["T2"] = t2_position(j, mats, "knee_low", "knee_high", args.repeats)
    if 3 in want:
        report["T3"] = t3_monotonic(j, mats, knees, args.repeats, truth)
    if 4 in want:
        report["T4"] = t4_naturalness(
            j, mats, ["a_ref", "super_heavy", "chaotic"], args.repeats)

    banner("汇总")
    names = {"T1": "同一性", "T2": "位置不变性", "T3": "单调排序", "T4": "自然度对齐"}
    for k, nm in names.items():
        if k in report:
            print(f"  {k} {nm:<10}{report[k]['status']}")
    tok = sum(c.usage.get("total_tokens") or 0 for c in j.calls)
    errs = sum(1 for c in j.calls if c.error)
    print(f"\n  共 {len(j.calls)} 次调用，{errs} 次失败，约 {tok} tokens")
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
