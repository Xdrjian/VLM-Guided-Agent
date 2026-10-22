#!/usr/bin/env python
"""外层循环驱动脚本（workflow.md §4）。

把四个注入点接到实机实现上并推进循环::

    LLM 给参数 → 求解器生成动作 → 特征提取 → 渲染 → VLM 同场对比打分
        → History 判定接受/回滚 → 终止检查

两种模式
--------
``--mode manual``
    模型调用挂起、材料落盘，人工把答案粘进指定文件后**重跑同一条命令**续跑。
    不需要 API Key，用来验证闭环本身是否通。

``--mode api``
    走 Gemini。API Key 从环境变量读取，本脚本不接收、不打印 Key。

断点续跑
--------
所有中间产物按轮次落盘，``history.json`` 记录进度。重跑同一个 ``--run-dir``
会从上次中断处继续，已完成的轮次不会重算。

示例::

    python scripts/run_loop.py --run-dir ../pipeline/runs/exp01 --mode manual --max-steps 1
    python scripts/run_loop.py --run-dir ../pipeline/runs/exp01 --mode api
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from motionedit.clients import (GEMINI_VIDEO_FPS, GeminiLLMClient,    # noqa: E402
                                GeminiVLMClient, ManualLLMClient,
                                ManualVLMClient)
from motionedit.history import TerminationConfig                      # noqa: E402
from motionedit.outer_loop import LoopConfig, OuterLoop               # noqa: E402
from motionedit.pipeline import ExecCheckConfig, MotionSolveFn        # noqa: E402
from motionedit.prompts import VideoSpec                              # noqa: E402
from motionedit.records import EvidenceStore                          # noqa: E402
from motionedit.renderer import MotionRenderer, RenderConfig          # noqa: E402
from motionedit.solver import SolverConfig                            # noqa: E402

DEFAULT_SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
DEFAULT_MOTION = REPO / "Experiment/pipeline/up_bone/A_ref_straight"
TRAY = "draw_organizer_tray"


def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def build_factories(args, store: EvidenceStore):
    """返回 ``(vlm_factory, llm_factory)``。

    VLM 的 seed 取重复序号 ``k``：``vlm_repeats=1`` 时恒为 0（可复现），
    需要噪声摸底时各次才互不相同，否则温度为 0 会给出完全一样的回答。
    """
    if args.mode == "manual":
        def vlm_factory(idx: int, st: EvidenceStore, k: int):
            d = st.ensure(idx)
            suffix = "" if args.vlm_repeats == 1 else f"_rep{k}"
            return ManualVLMClient(d / f"vlm_prompt{suffix}.txt",
                                   d / f"vlm_answer{suffix}.json",
                                   name=f"vlm{suffix}")

        def llm_factory(idx: int, st: EvidenceStore):
            d = st.ensure(idx)
            return ManualLLMClient(d / "llm_prompt.txt", d / "llm_answer.json",
                                   name="llm")
        return vlm_factory, llm_factory

    def vlm_factory(idx: int, st: EvidenceStore, k: int):
        return GeminiVLMClient(args.vlm_model, temperature=args.temperature,
                               seed=k, max_tokens=args.max_tokens,
                               max_retries=args.max_retries,
                               video_fps=args.fps)

    def llm_factory(idx: int, st: EvidenceStore):
        return GeminiLLMClient(args.llm_model, temperature=args.temperature,
                               seed=0, max_tokens=args.max_tokens,
                               max_retries=args.max_retries)
    return vlm_factory, llm_factory


def main() -> int:
    ap = argparse.ArgumentParser(description="外层循环驱动")
    ap.add_argument("--run-dir", type=Path, required=True, help="本次实验的证据包目录")
    ap.add_argument("--mode", choices=("manual", "api"), default="manual")
    ap.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="A_ref")
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKEL)
    ap.add_argument("--weight", type=float, default=30.0, help="目标重量 kg，全程不变")
    ap.add_argument("--max-steps", type=int, default=None, help="本次最多推进几轮")

    g = ap.add_argument_group("模型")
    g.add_argument("--vlm-model", default="gemini-3.8-flash")
    g.add_argument("--llm-model", default="gemini-3.8-flash")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--max-retries", type=int, default=2)
    g.add_argument("--fps", type=float, default=GEMINI_VIDEO_FPS,
                   help="视频采样率。默认 1fps 下模型会把屈膝深浅答反，不要调低")
    g.add_argument("--vlm-repeats", type=int, default=1,
                   help="同一组素材重复问几次取中位数；A0 实测极差 ≤1，故默认 1")

    g = ap.add_argument_group("终止条件（🔴 占位值，待实测标定）")
    g.add_argument("--patience", type=int, default=4)
    g.add_argument("--score-target", type=float, default=0.90)
    g.add_argument("--max-rounds", type=int, default=20)
    g.add_argument("--improvement-margin", type=float, default=0.02)
    g.add_argument("--ref-naturalness-floor", type=int, default=7,
                   help="第0轮自然度健康检查下限；实测动捕真值稳定得 7 分")

    g = ap.add_argument_group("求解器")
    g.add_argument("--adam-iters", type=int, default=SolverConfig.adam_iters)
    g.add_argument("--lbfgs-iters", type=int, default=SolverConfig.lbfgs_iters)
    g.add_argument("--rel-tol", type=float, default=0.15,
                   help="指令跟踪的相对误差上限，超出即告警")
    g.add_argument("--max-consecutive-failures", type=int, default=2,
                   help="连续多少轮执行失效才终止；单次失效只告警")

    g = ap.add_argument_group("渲染")
    g.add_argument("--video-stride", type=int, default=1)
    g.add_argument("--render-batch", type=int, default=8)
    g.add_argument("--video-batch", type=int, default=6)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    store = EvidenceStore(run_dir)

    banner("1. 装配")
    media = VideoSpec()
    solve_fn = MotionSolveFn(
        args.motion, args.skeleton, tray_name=TRAY,
        solver_cfg=SolverConfig(adam_iters=args.adam_iters,
                                lbfgs_iters=args.lbfgs_iters,
                                device=args.device,
                                log_every=max(args.adam_iters // 4, 1)),
        check=ExecCheckConfig(rel_tol=args.rel_tol,
                              max_consecutive=args.max_consecutive_failures))
    renderer = MotionRenderer(
        RenderConfig(device=args.device, video_stride=args.video_stride,
                     batch_size=args.render_batch,
                     video_batch_size=args.video_batch, write_video=True),
        repo_root=REPO, skeleton=solve_fn.skel)

    ref = solve_fn.ref_features
    print(f"  设备 {args.device} | 模式 {args.mode} | 目标 {args.weight:g}kg")
    print(f"  A_ref {ref.num_frames} 帧 / {ref.duration_s:.2f}s | "
          f"步频 {ref.cadence_steps_per_s:.2f} 步/秒 | 左膝峰值 {ref.knee_flex_left_max:.1f}°")
    if solve_fn.grip is not None:
        print(f"  抓握标定 {json.dumps(solve_fn.grip.summary(), ensure_ascii=False)}")
    if args.mode == "api":
        print(f"  VLM {args.vlm_model} @ {args.fps:g}fps | LLM {args.llm_model} | "
              f"温度 {args.temperature} | 重复 {args.vlm_repeats}")

    vlm_factory, llm_factory = build_factories(args, store)
    cfg = LoopConfig(
        run_dir=run_dir, target_weight_kg=args.weight, media=media,
        termination=TerminationConfig(patience=args.patience,
                                      score_target=args.score_target,
                                      max_rounds=args.max_rounds,
                                      ref_naturalness_floor=args.ref_naturalness_floor,
                                      improvement_margin=args.improvement_margin),
        vlm_repeats=args.vlm_repeats,
        reference_motion=Path(args.motion))
    loop = OuterLoop(cfg, vlm_factory, llm_factory, renderer=renderer, solve_fn=solve_fn)
    # 第 0 轮直接用 A_ref 的特征，不经过求解器（workflow.md §4.4）
    loop.reference_features = {k: v for k, v in ref.to_dict().items()
                               if isinstance(v, (int, float))}

    banner("2. 推进")
    if len(loop.history):
        print(f"  从第 {loop.history.next_round_index} 轮续跑"
              f"（已有 {len(loop.history)} 轮记录）")
    history = loop.run(max_steps=args.max_steps)

    banner("3. 结果")
    for rec in history.records:
        print(f"  {rec.summary()}")
        if rec.warnings:
            for w in rec.warnings:
                print(f"      ⚠ {w}")
    best = history.best
    if best is not None:
        print(f"\n  历史最佳：第 {best.round_index} 轮"
              f"（总分 {best.total_score:.4f}）")
        print(f"  参数 {json.dumps(best.params.to_dict(), ensure_ascii=False)}")
    stop = history.should_stop()
    print(f"\n  终止状态：{stop or '未终止（可继续推进）'}")
    if history.stop_detail:
        print(f"  说明：{history.stop_detail}")
    print(f"  证据包：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
