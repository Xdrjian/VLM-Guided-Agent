# motionedit — 负重步态编辑求解器（代码包）

把一段"空手搬运"的动捕动作，编辑成"负重 30kg 搬运"的视觉效果。
本目录是**代码包**；设计依据见仓库根目录的 `workflow.md`（总体架构）与
`Solver_Prework.md`（实现规格）。

---

## 1. 当前进度

| 层级 | 内容 | 状态 |
|---|---|---|
| 第 1 层 · 地基 | 数据读写、骨架正运动学、生物力学测量、关节限位 | ✅ 完成 |
| 第 2 层 · 目标 | 把 7 个宏观参数翻译成逐帧目标曲线 | ✅ 完成（`targets.py`） |
| 第 3 层 · 求解 | 五项损失 + 权重退火 + Adam→L-BFGS | ✅ 完成（`losses.py` / `solver.py`） |
| 第 4 层 · 闭环 | 多视角渲染 → VLM 打分 → LLM 调参 | 🟡 逻辑完成，渲染器与 API 适配层待补 |

> 第 1~3 层已经可以**独立跑完一次完整求解**：给定一组参数，产出新的 `.pkl` 动作文件。
> 第 4 层的循环逻辑（记录 / 历史 / prompt / 客户端 / 终止条件）已完成并通过 113 项自测，
> 渲染与求解以**注入点**形式接入；尚缺两块实机实现：
> ① 多机位渲染器（需 pytorch3d，仅 Linux 机可跑）；② API 适配层（待 A0 探路选型后再写）。

---

## 2. 目录与文件作用

```
Experiment/src/
├── requirements.txt          依赖清单（锁定目标机已验证的版本）
├── README.md                 本文件
├── HYPERPARAMETERS.md        ★ 超参数清单：所有可调量的位置、默认值与调整建议
├── motionedit/               核心包
│   ├── rotation.py           旋转数学：四元数 ↔ 矩阵 ↔ 欧拉角，全部可微
│   ├── skeleton.py           静止骨架读取 + 批量正运动学（FK）+ 可优化骨骼掩码
│   ├── motion_io.py          动作 pkl 的读写与张量化
│   ├── params.py             LLM 可调的 7 个宏观参数及其取值范围
│   ├── phase.py              步态相位识别（触地检测、步频、双支撑相）
│   ├── features.py           生物力学特征测量（膝屈曲、躯干前倾、重心起伏……）
│   ├── rom.py                关节活动度约束（swing-twist 软惩罚）
│   ├── targets.py            参数 → 逐帧目标曲线（时间重映射 + 姿态/根节点目标）
│   ├── losses.py             五项损失函数 + 权重退火 + 自动定标
│   ├── solver.py             求解器主体（增量参数化 + Adam → L-BFGS）
│   ├── tray.py               托盘位姿跟随（双手中点驱动 + 朝向约束 + 平滑）
│   ├── records.py            【第4层】单轮记录与证据包（可复现性的落盘基础）
│   ├── history.py            【第4层】历史最佳锚点 / 回滚语义 / 终止条件
│   ├── prompts.py            【第4层】VLM & LLM prompt 模板（BARS 量表）
│   ├── clients.py            【第4层】模型客户端（手动 / API 双实现，同一接口）
│   ├── montage.py            【第4层】宫格拼图与留底视频（numpy + opencv）
│   └── outer_loop.py         【第4层】外层循环主体（渲染/求解为注入点）
├── scripts/
│   ├── inspect_reference.py  ★ 体检脚本：迁移验证 / 日常诊断
│   ├── run_solve.py          ★ 端到端演示：参数 → 求解 → 落盘
│   └── export_prompts.py     ★ 导出 A0 探路用的 prompt 原文与核查清单
└── tests/
    ├── test_core.py          ★ 第1~3层自检：67 项断言
    └── test_loop.py          ★ 第4层自检：113 项断言（不依赖 torch/opencv/网络）
```

各模块一句话说明：

| 模块 | 输入 → 输出 | 为什么需要它 |
|---|---|---|
| `rotation` | 四元数 ↔ 旋转矩阵 | 优化器要对旋转求梯度，必须全程可微 |
| `skeleton` | 关节旋转 → 全身各部位的空间坐标 | 没有它就无法知道"手在哪""脚有没有着地" |
| `motion_io` | pkl ↔ `[帧数, 骨骼数, 7]` 张量 | 打通数据与 PyTorch |
| `params` | 7 个旋钮的定义与合法区间 | LLM 的输出接口 |
| `phase` | 坐标序列 → 每帧处在步态周期的哪个位置 | 改步频、改双支撑相的前提 |
| `features` | 坐标序列 → 可读的物理量 | 既喂给 LLM 看，也用来定义优化目标 |
| `rom` | 姿态 → 违反人体解剖极限的程度 | 防止优化出"膝盖反折"这类不可能姿势 |
| `targets` | 7 个参数 → 每一帧的具体数值目标 | 求解器需要可比较的目标，而非抽象描述 |
| `losses` | 姿态 → 一个可反向传播的标量 | 定义"什么叫好"，是优化的指挥棒 |
| `solver` | 参考动作 + 目标 → 新动作 | 优化引擎本体 |
| `tray` | 双手位置 → 托盘位姿 | 人动了托盘要跟着动，否则会穿模/脱手 |

### 关联数据文件（不在本目录）

| 路径 | 作用 |
|---|---|
| `humoto/human_model/human_model_up_bone_zup.json` | 静止骨架（65 根骨骼，Z 轴朝上） |
| `Experiment/pipeline/up_bone/A_ref_straight/` | **参考动作**：83 帧直线行走段（已剔除掉头动作） |

---

## 3. 在目标机上安装

目标机：Ubuntu 24.04.4 / Python 3.10.11 / CUDA 13.0 / RTX 5070 Laptop (8GB)。

```bash
source /home/jihan_xu/Desktop/Real_Work/Torch_Cuda_ready/.venv/bin/activate
cd <仓库根>/Experiment/src
python -c "import torch, numpy, yaml; print(torch.__version__, numpy.__version__)"
```

**若上面这行能打印出 `2.9.1+cu128 2.1.3`，则无需安装任何东西，直接跳到第 4 节。**
依赖早已装好，`requirements.txt` 的作用是**核对**而非安装。

三个必须注意的坑：

1. **不要装 `torchaudio`** — 目标机上该包已损坏，且本项目完全用不到。
2. **不要重装 `pytorch3d`** — 目标机没有 `nvcc`，无法从源码编译；现有的
   0.7.8 是预编译版，卸掉就装不回来了。（仅第 4 层渲染阶段用得到）
3. **不要升级 `numpy`** — 2.x 已移除 `np.float_` 等别名，升级可能连带破坏
   torch 的二进制兼容性。

---

## 4. 迁移验证（重点）

### 步骤一：自检测试

```bash
cd <仓库根>/Experiment/src
python tests/test_core.py
```

**预期**：最后一行为

```
结果: 67 passed, 0 failed
```

任何一项 `FAILED` 都说明迁移有问题，**不要继续**，请把完整输出反馈。

### 步骤二：体检脚本

```bash
python scripts/inspect_reference.py
```

**预期输出中，以下数值应与下表完全一致**（这些都是从真实数据算出来的物理量，
与操作系统、CPU/GPU 无关；若有偏差说明数据或依赖出了问题）：

| 检查项 | 预期值 | 说明 |
|---|---|---|
| 骨骼数 | `65` | 骨架完整 |
| 帧数 / 时长 | `83` / `3.458 s` | 参考动作正确 |
| pose 张量形状 | `(83, 65, 7)` | 数据读取正确 |
| 可优化骨骼 | `21 / 65` | 自由度掩码正确 |
| Hips 静止高度 | `0.8827 m` | 骨架尺度正确（单位是米） |
| 角色总高 | `1.5724 m` | 同上 |
| 步频 | `1.412 步/秒` | 步态识别正确 |
| 步态周期 | `34.000 帧` | 同上 |
| 双支撑相占比 | `0.265` | 同上 |
| 步长 | `0.922 m` | 同上 |
| 左 / 右膝屈曲峰值 | `98.825°` / `97.011°` | **正运动学正确**（正常人行走 90~100°） |
| 躯干前倾均值 | `10.513°` | 符合端托姿势 |
| 重心垂直起伏 | `0.101 m` | 正常人行走 8~12 cm |
| 托盘-胸骨距离 | `0.423 m` | 托盘在身前 |
| 相位采样 8 帧 | `[14, 53, 23, 27, 32, 36, 40, 78]` | 抽帧策略确定性 |
| **ROM 总惩罚** | **`0.0000`** | **最关键的一项，见下方说明** |
| 四元数归一化误差 | `< 1e-5`（实测 `5.96e-08`） | 托盘朝向计算稳定 |
| 总结行 | `[通过] 各项指标正常` | — |

浮点数末位出现 ±1 的差异属正常（不同 BLAS 实现），但**整数值与小数点后两位
必须一致**。

### 关于 "ROM 总惩罚 = 0" 为什么最关键

参考动作是真人动捕的正常行走，按定义它**不可能违反人体关节极限**。所以这个
数必须是 0。

历史上这里出过一次严重问题：早期设计采用国际生物力学标准（ISB）的固定欧拉角
分解来判定关节越界，结果它判定这段正常行走有 **18 个关节违规、最严重的说膝盖
弯了 166.9°**（实际只有 98.8°）。根因是本数据使用的骨骼坐标约定与 ISB 假设
不兼容。若未发现，优化器会被这个错误信号严重误导，且**表面上不会报任何错**。

现已改为与坐标约定无关的 swing-twist 分解方案，参考动作惩罚归零。
**因此这一项是整个体检中最有诊断价值的哨兵指标** —— 它一旦非零，就说明限位
表或坐标约定被破坏了，此时绝不能继续做优化。

### 步骤三（可选）：导出 JSON 便于比对

```bash
python scripts/inspect_reference.py --json /tmp/aref.json
```

### 步骤四：端到端求解

```bash
python scripts/run_solve.py --preset heavy --out ../pipeline/solved/heavy
```

**预期**：最后的对比表中，**五项指标的"实际 vs 目标"误差均应 < 1.0**：

```
  指标                          参考        目标        实际        误差
  左膝屈曲峰值 (°)               98.83    123.39    122.74      0.65
  躯干前倾均值 (°)               10.51     18.40     18.64      0.24
  左肘屈曲均值 (°)               67.28     77.19     77.19      0.00
  步频 (步/秒)                  1.41      1.27      1.26      0.01
  托盘-胸骨距离 (m)               0.42      0.36      0.36      0.00
```

同时 `求解` 段的 `rom` 必须是 `0.0000 -> 0.0000`。

> 本机 CPU 上约需 75 秒（300+40 次迭代）；目标机 GPU 上应显著更快。
> 若 GPU 上耗时反而更长，说明 `--device cuda` 未生效，请检查 `torch.cuda.is_available()`。

---

## 5. 日常用法

```bash
# 检查求解器产出的新动作
python scripts/inspect_reference.py --motion <新动作.pkl>

# 换一个参考序列
python scripts/inspect_reference.py --motion ../pipeline/up_bone/<序列名>/
```

脚本返回码：`0` = 通过，`1` = 指标异常，`2` = 文件缺失。可直接用于自动化流水线。

在代码中使用：

```python
from motionedit.skeleton import Skeleton
from motionedit.motion_io import load_motion
from motionedit.features import extract_features
from motionedit.rom import build_rom_penalty

skel = Skeleton.from_json("humoto/human_model/human_model_up_bone_zup.json")
seq  = load_motion("Experiment/pipeline/up_bone/A_ref_straight",
                   bone_order=skel.bone_names, device="cuda")

feats, curves = extract_features(seq, skel, tray_name="draw_organizer_tray")
print(feats.to_dict())          # 可直接 json.dumps 后喂给 LLM

rom = build_rom_penalty(skel)   # 预编译一次，优化循环里反复调用
loss_rom = rom(seq.pose)        # 标量，可反向传播
```

完整求解流程：

```python
from motionedit.params import GaitParams
from motionedit.targets import build_targets
from motionedit.solver import solve, SolverConfig
from motionedit.tray import update_tray_in_sequence

params = GaitParams.from_dict(llm_json).clipped()       # LLM 输出 → 参数
target = build_targets(curves, params, feats.tray_chest_distance_mean)
result = solve(seq, skel, target, cfg=SolverConfig(device="cuda"))

print(result.summary())                                  # 各项损失的收敛情况
new_seq = update_tray_in_sequence(new_seq, skel)         # 托盘跟手
```

**务必检查 `result.converged` 与 `result.notes`** —— 求解器不会抛异常，它会把
ROM 越界、安全阀触发、L-BFGS 线搜索失败等问题记在 `notes` 里并把 `converged`
置 `False`。忽略它等于放过所有物理错误。

---

## 6. 已知约束

- **不要 import `torchaudio`**（目标机上已损坏）。
- 8GB 显存：第 4 层渲染时注意分辨率与批大小，必要时分批渲染。
- `HUMOTO_OBJECT_DIR` 指向 HuMoTo 的对象模型库（`humoto_objects_0805`），
  渲染托盘**网格**时需要它（`<object_name>/<object_name>.obj`）。
  注意：托盘的**逐帧位姿**一直存在于 `pkl['objects']` 中，与该变量无关；
  该变量只影响托盘的可见几何体。未配置时渲染器降级为只渲染人体并告警。
  当前的第 1 层功能不受影响。
- `Spine` 骨骼携带约 `0.0131 m` 的非零平移（其余非根骨骼为 0）。这是数据的
  真实属性，读写时必须原样保留，**不可一律置零**。
