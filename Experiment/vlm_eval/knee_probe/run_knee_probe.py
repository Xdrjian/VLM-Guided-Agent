#!/usr/bin/env python
"""膝屈曲聚焦探针：把「感知能力」与「评分协议」分开测。

为什么要单开一个
----------------
主测试 T3 的三档膝屈曲（98.8° / 128.4° / 146.6°）在五个维度上**全部判为 tie**。
但探针 ⑭ 实测过：同一模型在 12fps 下被**直接问**"谁屈膝更深"时答对且误差 18%。

两件事因此必须分开：

    感知层：模型能否看出膝屈曲差异            <- 本脚本测这个
    协议层：该差异能否传导到 Weight/Space 评分 <- 主测试 T3 测这个

若本脚本通过而 T3 不通过，瓶颈在**维度定义与映射**（屈膝更深≠判官眼中的更重），
而不是模型看不见；那时该改的是 BARS 与观察要点，不是渲染或模型。
若本脚本也不通过，才是感知能力的天花板。

提示词设计
----------
刻意做到最简，但保留我们在主流程中确立的几条纪律：
1. 先写证据（机位 + 时间段），再给数值；
2. 角度**给出操作性定义**并锚定 0°，否则"膝关节角度"有两种互补读法，
   实测模型会用「大腿与小腿的夹角」而我们的特征用的是「屈曲角 = 180° − 夹角」；
   两个数都要，互为交叉校验；
3. 提供 ``tie`` / ``insufficient`` 出口，不预设差异一定存在；
4. 用中性的"动作 A / 动作 B"，重复时交替顺序。

真值不从 index.json 读，而是直接从每份素材的 motion.pkl 重算，
取左右膝峰值的较大者——模型未必分得清左右腿。

用法::

    cd Experiment/vlm_eval/knee_probe
    python run_knee_probe.py                  # 配对 + 单段两种模式
    python run_knee_probe.py --mode pair --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = EVAL.parents[1]
sys.path.insert(0, str(EVAL.parents[0] / "src"))

from motionedit.clients import GEMINI_VIDEO_FPS, GeminiVLMClient   # noqa: E402
from motionedit.features import extract_features                   # noqa: E402
from motionedit.motion_io import load_motion                       # noqa: E402
from motionedit.prompts import extract_json                        # noqa: E402
from motionedit.skeleton import Skeleton                           # noqa: E402

SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
ITEMS = ["knee_low", "knee_mid", "knee_high"]

SYSTEM = (
    "You are a human motion analysis observer. Your task is to estimate the joint angle from the rendered video.\n"
    "You must output exactly one JSON object and nothing else."
)

#: Operational definition of the angle. Without this definition it is not comparable: the model usually reads the
#: included angle between thigh and shank, while our feature extraction uses flexion angle = 180° - included angle.
_ANGLE_DEF = """[Knee angle definition]
For the same time instant and the same leg, there are two complementary readings; provide both:
- ``included_angle_deg``: angle between the thigh and shank. When the leg is fully extended, this is 180°; when the knee is heavily bent and the shank folds toward the thigh, it approaches 0°.
- ``flexion_deg``: flexion angle. When the leg is fully extended, this is 0°; the more the knee folds, the larger this value becomes.
The relationship is ``flexion_deg = 180 - included_angle_deg``. If the two numbers you provide do not satisfy this relationship, your reading is wrong; recheck before outputting.

[What to measure]
Use the moment in the full video where the knee is most flexed (peak flexion), using the deeper of the left/right knee if needed, without distinguishing left and right.
This usually occurs during the swing phase when the shank folds up, or during the support phase when the knee buffers the landing.
"""

_RULES = """[Observation rules]
1. First state in ``observations`` which camera view and which time window you used to identify the peak value, then give the numbers. Keep it to a single sentence, under 40 characters if possible.
2. If the view, occlusion, or resolution makes the measurement impossible, set ``observations`` to ``insufficient_visibility`` and fill the corresponding numeric fields with ``null``. This is a valid outcome, not a failure.
3. Do not guess just to provide an answer, and do not assume two clips are the same just because they look similar.
4. The render is a plain humanoid without textures; this is part of the experimental setup and should not affect the reading.
Each clip is a stitched 3-view frame with side view, front view, and 45° front-up view from left to right. The sagittal view (leftmost camera) is usually the clearest for knee-angle measurement.
"""

PAIR_FORMAT = """Strictly output JSON in the following format, with fields in this order and no extra fields:
{
  "peak_observation_A": {
    "view": "side"|"front"|"diag"|str,
    "time_range": "...",
    "evidence": "..."
  },
  "peak_observation_B": {
    "view": "side"|"front"|"diag"|str,
    "time_range": "...",
    "evidence": "..."
  },
  "deeper": "A"|"B"|"tie"|"insufficient",
  "difference": "none"|"slight"|"clear"|"large",
  "confidence": "low"|"medium"|"high"
}

- ``peak_observation_A`` and ``peak_observation_B`` record the peak observation for each action.
- ``view`` should indicate which camera view was used to observe the peak (for example side / front / diag).
- ``time_range`` should indicate the time interval where the peak is observed, such as "00:02.250-00:02.333".
- ``evidence`` should be a short sentence explaining why this is the peak.
- ``deeper`` indicates which action is more flexed (larger flexion angle); if they are equal, use ``tie``; if the comparison is not possible, use ``insufficient``. **Note: It is virtually impossible for the result to be a ``tie``.**
- ``difference`` describes the magnitude of the comparison: ``none`` = nearly identical, ``slight`` = small difference, ``clear`` = obvious difference, ``large`` = very large difference.
"""

SOLO_FORMAT = """Strictly output JSON in the following format, with fields in this order and no extra fields:
{"observations": str,
 "included_angle_deg": number|null,
 "flexion_deg": number|null,
 "confidence": "low"|"medium"|"high"}
"""


def pair_prompt() -> str:
    return f"""The following are two rendered videos. The first is labeled **Action A** and the second is **Action B**.
Both are target actions to compare. The submission order contains no hidden cue.

{_ANGLE_DEF}

{_RULES}

[Output format]
{PAIR_FORMAT}
"""


def solo_prompt() -> str:
    return f"""The following is one rendered video.

{_ANGLE_DEF}

{_RULES}

[Output format]
{SOLO_FORMAT}
"""


# --------------------------------------------------------------------------- #
def banner(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def ground_truth(materials: Path) -> Dict[str, float]:
    """从每份素材的 motion.pkl 重算膝屈曲峰值，取左右腿较大者。"""
    skel = Skeleton.from_json(SKEL)
    out: Dict[str, float] = {}
    for name in ITEMS:
        pkl = materials / name / "motion.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"缺少 {pkl}，请先运行 ../make_materials.py")
        seq = load_motion(pkl, bone_order=skel.bone_names)
        f, _ = extract_features(seq, skel, tray_name="draw_organizer_tray")
        out[name] = max(f.knee_flex_left_max, f.knee_flex_right_max)
    return out


def _num(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class Probe:
    def __init__(self, args, out: Path):
        self.args, self.out = args, out
        out.mkdir(parents=True, exist_ok=True)
        self.log: List[dict] = []

    def _call(self, prompt: str, videos: List[Path], seed: int) -> Optional[dict]:
        cli = GeminiVLMClient(self.args.model, temperature=self.args.temperature,
                              seed=seed, max_tokens=self.args.max_tokens,
                              max_retries=self.args.max_retries,
                              video_fps=self.args.fps)
        rec: dict = {"videos": [str(v) for v in videos], "seed": seed}
        try:
            resp = cli.call(SYSTEM, prompt, videos)
            rec["raw"], rec["usage"] = resp.text, resp.usage
            rec["finish_reason"] = (resp.body or {}).get("finish_reason")
            rec["max_output_tokens"] = (resp.body or {}).get("max_output_tokens")
            rec["data"] = extract_json(resp.text)
        except Exception as e:                           # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            tag = " | ".join(Path(v).parent.name for v in videos)
            print(f"    [调用失败] {tag} seed={seed}  {rec['error'][:150]}")
        self.log.append(rec)
        (self.out / "raw_calls.json").write_text(
            json.dumps(self.log, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec.get("data")


# --------------------------------------------------------------------------- #
def run_solo(p: Probe, mats: Dict[str, Path], gt: Dict[str, float],
             n: int) -> dict:
    banner("A · 单段绝对读数：每段单独问峰值屈曲角")
    rows: List[dict] = []
    print(f"\n  {'素材':<12}{'真值°':>8}{'模型屈曲°':>11}{'夹角°':>8}"
          f"{'互补校验':>10}{'误差°':>8}  置信  依据")
    for name in ITEMS:
        for k in range(n):
            d = p._call(solo_prompt(), [mats[name]], k)
            if not d:
                print(f"  {name:<12}{gt[name]:8.1f}   调用失败")
                continue
            fx, inc = _num(d.get("flexion_deg")), _num(d.get("included_angle_deg"))
            comp = (abs((fx + inc) - 180.0) if fx is not None and inc is not None
                    else None)
            err = abs(fx - gt[name]) if fx is not None else None
            rows.append({"material": name, "repeat": k, "truth": gt[name],
                         "flexion": fx, "included": inc,
                         "complement_err": comp, "abs_err": err,
                         "confidence": d.get("confidence")})
            obs = str(d.get("observations", ""))[:34]
            print(f"  {name:<12}{gt[name]:8.1f}"
                  f"{(f'{fx:.0f}' if fx is not None else 'null'):>11}"
                  f"{(f'{inc:.0f}' if inc is not None else 'null'):>8}"
                  f"{(f'{comp:.0f}' if comp is not None else '-'):>10}"
                  f"{(f'{err:.0f}' if err is not None else '-'):>8}"
                  f"  {str(d.get('confidence', '?'))[:6]:<8}{obs}")

    ok = [r for r in rows if r["abs_err"] is not None]
    if ok:
        mae = sum(r["abs_err"] for r in ok) / len(ok)
        bias = sum(r["flexion"] - r["truth"] for r in ok) / len(ok)
        print(f"\n  绝对误差均值 {mae:.1f}°，系统偏置 {bias:+.1f}°"
              f"（正=高估），有效读数 {len(ok)}/{len(rows)}")
        # 相关性：读数能否随真值单调上升，比绝对准确度更关键
        by = {m: [r["flexion"] for r in ok if r["material"] == m] for m in ITEMS}
        means = {m: (sum(v) / len(v) if v else None) for m, v in by.items()}
        seq_ok = all(means[a] is not None for a in ITEMS) and \
            means[ITEMS[0]] < means[ITEMS[1]] < means[ITEMS[2]]
        print(f"  各档读数均值：" +
              "  ".join(f"{m}={means[m]:.0f}°" if means[m] is not None else f"{m}=null"
                        for m in ITEMS))
        print(f"  单调递增：{'是' if seq_ok else '否'}"
              f"（真值 {gt[ITEMS[0]]:.0f} < {gt[ITEMS[1]]:.0f} < {gt[ITEMS[2]]:.0f}）")
    else:
        mae = bias = None
        seq_ok = False
        print("\n  无有效读数")
    return {"rows": rows, "mae": mae, "bias": bias, "monotonic": seq_ok}


def run_pair(p: Probe, mats: Dict[str, Path], gt: Dict[str, float],
             n: int) -> dict:
    banner("B · 配对比较：两两判断谁屈膝更深，并给出差值")
    results: List[dict] = []
    print(f"\n  {'比较':<26}{'模型判定':<12}{'真值':<12}{'方向':<6}"
          f"{'模型差°':>9}{'真值差°':>9}  置信")
    for x, y in combinations(ITEMS, 2):
        for k in range(n):
            a, b = (y, x) if k % 2 else (x, y)      # 交替顺序，抵消位置偏差
            d = p._call(pair_prompt(), [mats[a], mats[b]], k)
            if not d:
                continue
            verdict = str(d.get("deeper", "?"))
            picked = {"A": a, "B": b}.get(verdict)
            truth_win = x if gt[x] > gt[y] else y
            gt_diff = abs(gt[x] - gt[y])
            diff = _num(d.get("difference_deg"))
            if diff is None:
                diff = _num((d.get("difference") if isinstance(d.get("difference"), (int, float)) else None))
            diff_level = str(d.get("difference", "?"))
            obs_a = d.get("peak_observation_A") or {}
            obs_b = d.get("peak_observation_B") or {}
            agree = (picked == truth_win) if picked else None
            results.append({"pair": [x, y], "order": [a, b], "repeat": k,
                            "picked": picked, "verdict": verdict,
                            "truth": truth_win, "agree": agree,
                            "model_diff": diff, "truth_diff": gt_diff,
                            "difference": diff_level,
                            "peak_observation_A": obs_a,
                            "peak_observation_B": obs_b,
                            "confidence": d.get("confidence")})
            print(f"  {a + ' | ' + b:<26}{str(picked or verdict):<12}{truth_win:<12}"
                  f"{('对' if agree else ('错' if agree is False else '—')):<6}"
                  f"{(f'{diff:.0f}' if diff is not None else '-'):>9}{gt_diff:9.1f}"
                  f"  {str(d.get('confidence', '?'))[:6]}  {diff_level}")

    decided = [r for r in results if r["picked"]]
    right = [r for r in decided if r["agree"]]
    ties = [r for r in results if r["verdict"] in ("tie", "insufficient")]
    print(f"\n  方向判定：{len(right)}/{len(decided)} 正确"
          f"（另有 {len(ties)} 次判为打平/看不清，共 {len(results)} 次）")
    withdiff = [r for r in decided if r["model_diff"] is not None]
    if withdiff:
        mae = sum(abs(r["model_diff"] - r["truth_diff"]) for r in withdiff) / len(withdiff)
        print(f"  差值估计误差均值 {mae:.1f}°")
    else:
        mae = None

    # 循环偏好检查
    wins: Dict[Tuple[str, str], Optional[str]] = {}
    for x, y in combinations(ITEMS, 2):
        tally: Dict[str, int] = {}
        for r in results:
            if r["pair"] == [x, y] and r["picked"]:
                tally[r["picked"]] = tally.get(r["picked"], 0) + 1
        wins[(x, y)] = (max(tally, key=tally.get)                 # type: ignore[arg-type]
                        if tally and len(set(tally.values())) >= 1 and
                        len([k for k, v in tally.items()
                             if v == max(tally.values())]) == 1 else None)
    cyc = None
    edge = {}
    for (x, y), w in wins.items():
        if w:
            edge[(w, y if w == x else x)] = True
    for a, b, c in ((ITEMS[0], ITEMS[1], ITEMS[2]), (ITEMS[0], ITEMS[2], ITEMS[1])):
        if edge.get((a, b)) and edge.get((b, c)) and edge.get((c, a)):
            cyc = f"{a} > {b} > {c} > {a}"
    print(f"  循环偏好：{cyc or '无'}")
    return {"results": results, "accuracy": (len(right) / len(decided)
                                             if decided else None),
            "ties": len(ties), "diff_mae": mae, "cycle": cyc}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="膝屈曲聚焦探针")
    ap.add_argument("--materials", type=Path, default=EVAL / "materials")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--mode", choices=("pair", "solo", "both"), default="both")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--temperature", type=float, default=0.0)
    # 思考 token 与正文共享这个预算，给小了 JSON 会从中间断掉。
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--fps", type=float, default=GEMINI_VIDEO_FPS)
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    root = Path(args.materials)
    mats = {n: root / n / "video.mp4" for n in ITEMS}
    missing = [n for n, v in mats.items() if not v.exists()]
    if missing:
        print(f"素材缺失 {missing}，请先运行 ../make_materials.py")
        return 1

    gt = ground_truth(root)
    banner("真值（由各素材的 motion.pkl 重算，取左右膝峰值较大者）")
    for n in ITEMS:
        print(f"  {n:<12}屈曲峰值 {gt[n]:6.1f}°   夹角 {180 - gt[n]:6.1f}°")
    print(f"\n  相邻档差距：{gt[ITEMS[1]] - gt[ITEMS[0]]:.1f}° / "
          f"{gt[ITEMS[2]] - gt[ITEMS[1]]:.1f}°，"
          f"两端差距 {gt[ITEMS[2]] - gt[ITEMS[0]]:.1f}°")
    print(f"  模型 {args.model} @ {args.fps:g}fps | 温度 {args.temperature} | "
          f"重复 {args.repeats}")
    if args.show_prompt:
        banner("配对提示词原文")
        print(pair_prompt())

    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    p = Probe(args, out)
    report: Dict[str, Any] = {"ground_truth": gt, "model": args.model,
                              "repeats": args.repeats}
    if args.mode in ("solo", "both"):
        report["solo"] = run_solo(p, mats, gt, args.repeats)
    if args.mode in ("pair", "both"):
        report["pair"] = run_pair(p, mats, gt, args.repeats)

    banner("结论")
    s, q = report.get("solo"), report.get("pair")
    if q and q.get("accuracy") is not None:
        acc = q["accuracy"]
        print(f"  配对方向准确率 {acc:.0%}"
              f"（{q['ties']} 次判为打平/看不清）")
        if acc >= 0.8 and not q["cycle"]:
            print("  → 感知层没问题。主测试 T3 全判 tie，说明卡在**协议层**：")
            print("     膝屈曲的变化没有被映射到 Weight / Space 这两个维度上。")
            print("     该改的是 BARS 分档描述与观察要点，不是渲染或模型。")
        else:
            print("  → 感知层本身就不可靠，先解决看得见的问题，再谈评分协议。")
    if s and s.get("mae") is not None:
        print(f"  单段绝对读数误差均值 {s['mae']:.1f}°，"
              f"系统偏置 {s['bias']:+.1f}°，单调性 {'是' if s['monotonic'] else '否'}")
        print("  注：绝对准确度不是必需的——闭环只需要**相对排序**可靠。")
    tok = sum(r.get("usage", {}).get("total_tokens") or 0 for r in p.log)
    bad = [r for r in p.log if "error" in r]
    print(f"\n  共 {len(p.log)} 次调用，{len(bad)} 次失败，约 {tok} tokens")
    for r in bad:
        tag = " | ".join(Path(v).parent.name for v in r["videos"])
        print(f"    失败 {tag} seed={r['seed']}：{r['error'][:150]}")
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
