"""导出 A0 探路阶段所需的 prompt 原文与核查清单。

A0 的目的**不是打分**，而是回答三个"要不要花钱买这家 API"的定性问题：

  ① 分辨力：模型能否看清 2×4 宫格中**单帧**的细节？
     —— 若这一项不过，宫格规格（行列数 / 单格像素）必须在买 API 之前重新设计。
  ② 同场对比：能否同时处理 6 张图并稳定输出合法 JSON？
  ③ 噪声：同一组图问 3 次，分数极差多大？决定 ``vlm_repeats`` 与 ``patience``。

注意：A0 走网页端，存在隐藏系统提示、图像二次压缩、输出后处理、同名模型
静默更新等不可控因素，**其结果不得进入实验记录**，仅用于选型。

用法::

    python scripts/export_prompts.py --out ../pipeline/a0_probe
    python scripts/export_prompts.py --rows 1 --cols 4 --cell-px 768
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionedit.params import GaitParams                       # noqa: E402
from motionedit.prompts import (                               # noqa: E402
    GridSpec, LLM_SYSTEM, VLM_SYSTEM, render_llm_prompt,
    render_vlm_prompt, render_vlm_prompt_round0,
)
from motionedit.records import RoundRecord, VLMScore           # noqa: E402


DISCRIMINATION_PROBE = """以下是一张人体行走动作的宫格图。
{grid_desc}

请不要评分。请仅回答以下三个问题，每题一句话：
1. 第 3 格中，右腿膝关节大致弯曲了多少度？（给出一个估计数值）
2. 从第 1 格到第 8 格，躯干前倾的幅度是变大、变小，还是基本不变？
3. 哪一格中人物的双脚同时着地？

（本次提问用于检验图像分辨力，不是评分任务。若你无法看清细节，请直接说明"看不清"，
不要猜测。）
"""

CHECKLIST = """# A0 探路核查清单

目的：在购买 API 之前，用网页端免费额度确认所选模型**够不够用**。
结果仅用于选型，**不得作为实验数据**（网页端不可复现，理由见 clients.py 顶部说明）。

当前宫格规格：{grid_desc}
宫格图像素：{grid_px[0]} × {grid_px[1]}

---

## ① 分辨力测试（最关键，不过则宫格方案需重新设计）

用 `probe_discrimination.txt`，只提交 **1 张**宫格图。

| 检查项 | 通过标准 |
|---|---|
| 膝关节角度估计 | 给出的数值与实测值偏差 < 20°，且不同帧之间能给出**不同**的数值 |
| 躯干前倾趋势 | 判断方向正确 |
| 双脚同时着地帧 | 指认的格号正确或相邻 |
| 是否诚实 | 看不清时明确说"看不清"，而非编造数值 |

**若失败**：优先依次尝试 (a) 提高 `--cell-px`，(b) 开启人物裁剪让人物填满每格，
(c) 降为 1×4 宫格（每机位 4 帧，分两张图提交）。每改一次重测本项。

## ② 同场对比 + 格式遵从

用 `vlm_round_n.txt`，提交 **6 张**图（左 3 = 历史最佳，右 3 = 候选）。

| 检查项 | 通过标准 |
|---|---|
| 输出为合法 JSON | 能被 `extract_json` 直接解析 |
| 字段完整 | weight/time/space/flow/naturalness + comments 齐全 |
| 分档合法 | Effort 各项 ∈ [1,5]，naturalness ∈ [1,10] |
| 真的在对比 | 评语中出现"相比左列/历史最佳…"之类的具体差异描述，而非泛泛而谈 |
| 未越权评判 | 评语中**不出现**浮空/穿地/打滑/关节超限等物理合理性判断 |

## ③ 噪声摸底（决定两个关键超参）

同一组 6 张图 + 同一 prompt，**新开会话**问 3 次，记录三次分数。

| 维度 | 第1次 | 第2次 | 第3次 | 极差 |
|---|---|---|---|---|
| weight |  |  |  |  |
| time |  |  |  |  |
| space |  |  |  |  |
| flow |  |  |  |  |
| naturalness |  |  |  |  |

结论映射：

- 各维度极差 ≤ 1 → `vlm_repeats=1` 即可；`improvement_margin` 设为 0.02 量级。
- 存在维度极差 ≥ 2 → 必须 `vlm_repeats=3`（取中位数），
  且 `improvement_margin` 与 `patience` 都要上调，否则终止条件测的是噪声而非改进。

## ④ 第0轮 prompt 的健康检查预演

用 `vlm_round_0.txt`，提交 A_ref 的 3 张图。
自然度若 < 8/10，**不要**急着认为动作不自然——这更可能说明 prompt 或渲染质量有问题。

## ⑤ LLM 侧（文本，可与 VLM 用不同模型）

用 `llm_round_n.txt`。

| 检查项 | 通过标准 |
|---|---|
| 输出为合法 JSON | 7 个参数齐全且为数值 |
| 取值在范围内 | 无需裁剪即落在 PARAM_BOUNDS 内 |
| 基底正确 | reasoning 中体现"在历史最佳基础上改"，而非从被回滚的解延拓 |
| 不自我辩护 | 不出现为历史决策辩解的措辞 |

---

## 选型结论记录

| 模型 | ① 分辨力 | ② 对比/格式 | ③ 噪声极差 | ⑤ LLM | 结论 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 A0 探路用 prompt 与核查清单")
    ap.add_argument("--out", default="a0_probe", help="输出目录")
    ap.add_argument("--weight", type=float, default=30.0, help="目标重量 (kg)")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-px", type=int, default=512)
    args = ap.parse_args()

    grid = GridSpec(rows=args.rows, cols=args.cols, cell_px=args.cell_px)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 构造一个示例"历史最佳解"，让 round_n prompt 的锚点段有内容可展示
    best = RoundRecord(
        round_index=1,
        params=GaitParams(knee_flex_scale=1.25, trunk_lean_offset=8.0,
                          cadence_scale=0.90, double_support_offset=0.05,
                          stride_length_scale=0.85, elbow_flex_offset=10.0,
                          tray_chest_distance_scale=0.90),
        features={"knee_flex_peak_deg": 122.7, "trunk_lean_mean_deg": 18.6,
                  "cadence_steps_per_s": 1.26, "stride_length_m": 0.78},
        score=VLMScore(weight=3, time=3, space=3, flow=3, naturalness=8,
                       comments={"weight": "能看出负重但幅度不足",
                                 "time": "节奏偏慢，方向正确",
                                 "space": "步幅收窄合理",
                                 "flow": "略显机械",
                                 "naturalness": "整体自然，手臂略僵"}),
        accepted=True,
    )
    rolled = RoundRecord(
        round_index=2,
        params=GaitParams(knee_flex_scale=1.80, trunk_lean_offset=25.0,
                          cadence_scale=0.70, double_support_offset=0.15,
                          stride_length_scale=0.60, elbow_flex_offset=30.0,
                          tray_chest_distance_scale=0.70),
        features={"knee_flex_peak_deg": 152.1, "trunk_lean_mean_deg": 35.2,
                  "cadence_steps_per_s": 0.98, "stride_length_m": 0.55},
        score=VLMScore(weight=2, time=2, space=2, flow=2, naturalness=4,
                       comments={"weight": "夸张到失真", "time": "过慢",
                                 "space": "步幅过小", "flow": "断续",
                                 "naturalness": "姿态怪异"}),
        rolled_back=True,
    )

    files = {
        "probe_discrimination.txt":
            DISCRIMINATION_PROBE.format(grid_desc=grid.describe()),
        "vlm_round_0.txt":
            f"===== SYSTEM =====\n{VLM_SYSTEM}\n\n===== USER =====\n"
            + render_vlm_prompt_round0(args.weight, grid),
        "vlm_round_n.txt":
            f"===== SYSTEM =====\n{VLM_SYSTEM}\n\n===== USER =====\n"
            + render_vlm_prompt(3, args.weight, best, grid),
        "llm_round_n.txt":
            f"===== SYSTEM =====\n{LLM_SYSTEM}\n\n===== USER =====\n"
            + render_llm_prompt(3, args.weight, best, [rolled],
                                last_features=rolled.features),
        "CHECKLIST.md":
            CHECKLIST.format(grid_desc=grid.describe(), grid_px=grid.grid_px),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")

    print(f"已导出到 {out.resolve()}")
    for name in files:
        print(f"  - {name}")
    print(f"\n宫格规格：{grid.rows}×{grid.cols} = {grid.frames_per_camera} 帧/机位，"
          f"整图 {grid.grid_px[0]}×{grid.grid_px[1]} px，机位数 {len(grid.cameras)}")
    print("图像需在 Linux 机上用渲染器生成；prompt 文本现在即可用于 A0 网页端探路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
