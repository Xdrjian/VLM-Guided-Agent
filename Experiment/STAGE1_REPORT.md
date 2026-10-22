# 阶段一报告：VLM 引导的负重行走动作编辑

> 日期：2026-10-14 · 代码基线：`3ab6dc5` 之后
> 范围：从 A_ref（动捕真值）到"端 30kg 托盘行走"的单任务闭环打通

---

## 0. 一句话结论

**闭环已全程跑通，并在第三次真实运行中首次出现"候选解被接受"（总分 0.5119 → 0.6012）。**
管线本身（求解器、渲染、API、证据包、断点续跑）**可用且有测试覆盖**；
瓶颈已收敛到唯一一处：**VLM 评分信号的分辨力**，五个维度中目前只有两个在动。

---

## 1. 复现实验的命令

全部命令在 `Experiment/src/` 下执行。渲染与求解需要 CUDA + pytorch3d；
API 调用需环境变量中的 Gemini Key（代码不接收、不打印 Key）。

### 1.1 自检

```bash
python tests/test_core.py     # 第 1-3 层：96 项断言
python tests/test_loop.py     # 第 4 层：149 项断言，不依赖 GPU 与网络
```

### 1.2 单次求解（不含 VLM）

```bash
# 内置预设
python scripts/run_solve.py --preset heavy --out ../pipeline/solved/heavy

# 任意参数组合；缺省项自动取恒等值（不是取 preset 值）
python scripts/run_solve.py --params '{"cadence_scale":0.70}' --out ../pipeline/probe/slow
```

### 1.3 渲染

```bash
python scripts/render_motion.py --motion ../pipeline/solved/heavy \
    --out ../pipeline/renders/heavy --tag heavy --video-stride 2
```

产出：3 张宫格图（论文插图用）+ 1 段三机位横拼 mp4（提交给 VLM）+ 元信息 json。

### 1.4 完整闭环

```bash
# 手动模式：不花钱，验证管线。会在模型调用处挂起并落盘材料，
# 人工把答案粘进指定文件后重跑同一条命令续跑
python scripts/run_loop.py --run-dir ../pipeline/runs/smoke --mode manual --max-steps 1

# API 模式
python scripts/run_loop.py --run-dir ../pipeline/runs/exp04 --mode api --max-steps 3
```

### 1.5 检视结果

```bash
python scripts/inspect_run.py --run-dir ../pipeline/runs/exp04
```

输出逐轮参数、五维分数、对比判定、逐维评语、指令跟踪误差、LLM 理由、token 用量，
并做**信号诊断**：哪些维度全程恒定、总分实际被谁驱动。

### 1.6 API 能力探针（选型/排障用）

```bash
python scripts/probe_gemini.py --only 13,14 --fps 12 \
    --video-a <慢.mp4> --knee-base <k1.0.mp4> --knee-deep <k1.6.mp4>
```

---

## 2. 参数：当前开放的、未来可开放的

### 2.1 当前开放给 LLM 的——**仅 7 个生物力学中介参数**

定义与边界：`Experiment/src/motionedit/params.py`
翻译成逐帧目标曲线：`Experiment/src/motionedit/targets.py`

| 参数 | 语义 | 区间 | 恒等值 |
|---|---|---|---|
| `knee_flex_scale` | 膝屈曲角乘性缩放 | 0.5 ~ 2.0 | 1.0 |
| `hip_flex_offset` | 屈髋加性偏置（度，骨盆参考系，有符号） | −15 ~ +40 | 0 |
| `trunk_lean_offset` | 躯干前倾加性偏置（度） | −15 ~ +40 | 0 |
| `cadence_scale` | 步频乘性缩放 | 0.5 ~ 1.8 | 1.0 |
| `double_support_offset` | 双支撑相占比加性偏置 | −0.10 ~ +0.25 | 0 |
| `stride_length_scale` | 步长乘性缩放 | 0.4 ~ 1.6 | 1.0 |
| `elbow_flex_offset` | 肘屈曲加性偏置（度） | −20 ~ +45 | 0 |
| `tray_chest_distance_scale` | 托盘-胸骨水平距离缩放 | 0.5 ~ 1.5 | 1.0 |

LLM 每轮输出**完整的 7 个绝对值**（非增量），越界自动裁剪。

### 2.2 未来可开放的（当前写死）

| 层级 | 内容 | 当前状态 | 文件位置 |
|---|---|---|---|
| **loss 大类权重** | `w_task` vs `w_prior` 的相对强弱 | **写死**（自动定标 prior=0.3） | `losses.py` `LossWeights` |
| **第 0 层词汇表选择权** | 从完整生物力学特征全集中自选关注维度 | 人工预选上述 7 项 | `workflow.md` §参数分层架构 |
| **DOF mask** | 开放哪些具体关节为自由变量 | 静态 21 根核心骨骼 | `skeleton.py` `CORE_BONES`、`Skeleton.dof_mask()` |
| **best-of-N 多样性** | 同一目标下并行产出 N 组策略各异的候选 | N=1 | `outer_loop.py` `LoopConfig` |
| **上肢细粒度参数** | 肩屈曲、腕姿态、托盘水平度目标 | 不开放（水平度为 A 类硬约束） | `losses.py` `w_level` |
| **VLM 重复投票** | `vlm_repeats` | 1（A0 实测噪声极差 ≤1） | `outer_loop.py` `LoopConfig` |

> `workflow.md` 的 B 类设计原本要求把 `w_task` / `w_prior` 开放给 LLM。
> MVP 阶段未实现，**这是相对原设计最大的一处收缩**，需在下一阶段决定是否补上。

---

## 3. 损失函数的组成

实现：`Experiment/src/motionedit/losses.py`（`LossWeights` / `MotionLoss`）
求解：`Experiment/src/motionedit/solver.py`（增量旋转向量参数化 + Adam → L-BFGS）

$$L = L_{\text{task}} + w_{\text{prior}}(k) L_{\text{prior}} + w_{\text{rom}}(k) L_{\text{ROM}} + w_{\text{smooth}} L_{\text{smooth}} + w_{\text{contact}} L_{\text{contact}} + w_{\text{grip}} L_{\text{grip}} + w_{\text{level}} L_{\text{level}}$$

| 项 | 内容 | 权重 | 类别 |
|---|---|---|---|
| **task** | 膝/躯干/肘屈曲角（三点夹角）+ 根节点世界位置 + 托盘-胸骨距离 | knee 1.0 / trunk 1.0 / elbow 0.5 / root 300 | 目标 |
| **prior** | 与 A_ref 的四元数平方弦距离 | 自动定标至任务项的 30%，随迭代 ×0.1 衰减 | 正则 |
| **ROM** | swing-twist 分解 + relu 软惩罚，16 关节 | 自动定标 10%，随迭代 ×20 递增 | A 类硬约束 |
| **smooth** | 二阶差分（惩罚加速度） | 自动定标 10% | 正则 |
| **contact** | 穿地 1.0 / 防滑 1.0 / 锚定 0.5 | 自动定标 40% | A 类硬约束 |
| **grip** | 双手各自预测的托盘位姿分歧（闭链） | 1.0，位置按 1° ≙ 5mm 换算 | A 类硬约束 |
| **level** | 托盘倾角单边 hinge：`relu(tilt − 参考峰值 − 0.5°)²` | 1.0 | A 类硬约束 |

权重调参入口与全部默认值：`Experiment/src/HYPERPARAMETERS.md`。

**关键设计决策（均有实测支撑，见 §6）**：
- ROM 用 swing-twist 而非 ISB 欧拉分解
- 任务项在几何角度域比较，不反推四元数
- 时间映射暖启动（初始任务项 1828.9 → 555.4）
- 权重自动定标而非手工试凑
- 托盘位姿由双手对称最小二乘重建，不是独立自由体

---

## 4. 提示词与其实测效果

### 4.1 提示词文件位置

| 用途 | 文件 |
|---|---|
| **模板源码**（BARS 量表、观察要点、打分纪律、输出格式） | `Experiment/src/motionedit/prompts.py` |
| 每次运行实际发出的原文 | `<run-dir>/rounds/round_XXX/vlm_prompt.txt`、`llm_prompt.txt` |
| A0 探路阶段导出的静态样本 | `Experiment/pipeline/a0_probe/*.txt` |
| 模型原始返回体（含 usage、finish_reason） | `<run-dir>/rounds/round_XXX/vlm_response.json` |

### 4.2 实测效果演进

| 运行 | weight | time | space | flow | naturalness | 有候选被接受？ |
|---|---|---|---|---|---|---|
| exp02（初版 prompt） | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 | 7 / 6 / 6 | ❌ 全部回滚 |
| exp03（修打分纪律后） | **2 / 3 / 3** | **3 / 4 / 4** | 3 / 3 / 3 | 3 / 3 / 3 | 7 / 7 / 7 | ✅ **第 1 轮被接受** |

**exp02 的失效模式**：模型在评语里写出了单调递进的差异
（"缺乏负重感" → "略有放缓" → "体现出一定负重特征"），
分数却全部压在量程正中的 3 分。**评语与分数自相矛盾。**

诊断为三个根因，全在我们自己的 prompt 里：
1. BARS 结尾写着"请保留打分空间，不要在早期轮次就给出接近满分的评价"
   —— 砍掉了 4-5 档，而确实看得出在搬东西所以也不给 1-2，**只剩 3**。防见顶的措施制造了分数坍缩。
2. 只要候选的绝对分，没有任何义务让两段分数不同。
3. 自然度 10 分写的是"与真实动捕数据无法区分"，而画面是无贴图素模，**顶端在物理上不可达**。

对应修复（已落地）：删除逼中庸指令并要求分数与评语一致；
新增 `comparison` 字段逐维判定 better/worse/same 并在代码侧校验一致性；
显式声明只评动作、不评渲染质量。

**修复后 weight 与 time 开始移动，循环首次产生改进。**
space / flow / naturalness 仍恒定，但原因已改变——不再是不敢给分，
而是这两个构念过于抽象、模型无从下手（评语是"与基准基本相当"）。
最后一次修改已把五个维度**操作化为具体可观测量清单**（`prompts.py` 的 `OBSERVABLES`），
尚未验证。

---

## 5. 项目价值的再评估

### 5.1 已被证实可行的部分

| 主张 | 证据 |
|---|---|
| 可微轨迹优化能按中介参数精确编辑动捕动作 | 8 项指令跟踪指标，heavy 预设下最大误差 **8.6%**；恒等参数下总损失 0.0002 |
| VLM 能从渲染视频中读出**定量**运动差异 | 步频比值估计 2.0（真值 1.857，误差 **7.7%**）；膝屈曲差估计 20°（真值 24.5°，误差 **18%**） |
| LLM 能基于反馈做出**生物力学上正确**的决策 | 自主选择 `trunk_lean_offset` 为负（后仰配平前方载荷），**优于我们人工预设的 +8° 前倾** |
| 闭环可完整运转并产生改进 | exp03 第 1 轮被接受，总分 0.5119 → 0.6012 |
| 全链路可复现、可审计 | 每轮落盘 prompt / 请求 / 响应 / 评分 / 参数 / 特征 / 求解报告；温度 0 下实测输出逐字节确定 |

### 5.2 尚未被证实的核心主张

**"VLM 评分可作为可复现的人类评审替代协议"——这一条还没站住。**

五个维度中三个全程恒定，实际在起作用的只占总权重的 **35.7%**。
在这种信号下，"收敛"意味着什么是存疑的。这是项目方法论主张的要害，必须解决。

### 5.3 诚实的风险评估

| 风险 | 性质 |
|---|---|
| VLM 分辨力不足以驱动细粒度优化 | **高**。若观察要点清单无效，需要改渲染（纹理/光照/多机位分开提交）或换模型 |
| 7 个宏观参数不足以表达"沉重感" | **中**。模型反复提到"手臂与躯干的沉降受压感不明显"，而我们没有肩部下沉、手臂代偿类参数 |
| 自然度维度对"被编辑过"系统性扣分 | **中**。它占一半权重，若只会扣分则结构上抑制任何改动 |
| 终止条件三个阈值仍是占位值 | **低**。等分数分布稳定后标定即可 |

---

## 6. 已证实的结论（可直接写进论文）

这些都是在本阶段被**实测证伪或证实**的，每一条都有数据：

1. **ISB 固定欧拉轴序不适用于 up_bone 绑定**。对 A_ref 本身施加该方案得到伪惩罚 24803，
   最大伪越界 161.9°，而三点几何法测得的真实膝屈曲峰值仅 98.8°。改用 swing-twist 后惩罚为 0。
2. **时间映射暖启动是必需的**。无暖启动时膝关节需付出 71° 局部旋转去追 25° 的屈曲差距，
   并触发安全阀饱和；有暖启动后初始任务项 1828.9 → 555.4，耗时减半。
3. **接触项是必需的**。缺失时实测脚趾穿地 −0.024 m、全程无一帧站定、步频识别失效（0.58 vs 目标 1.27）。
4. **托盘是被双手刚性握持的，不是独立自由体**。左手→右手相对位姿抖动仅 2.08 mm / 0.05°。
   启发式重建误差 24.5 mm（与托盘半厚 24.9 mm 同量级，必然穿模），
   双手对称最小二乘降到 **0.92 mm**。
5. **腕关节在任务项中是死自由度**（实测仅动 0.82°），而手的世界朝向是整条链的乘积，
   `elbow_flex_offset` 一动前臂就转，托盘实测倾倒 **~30°**。
   加入托盘水平 hinge 后，求解器自发用腕关节补偿（肘 +25° 时腕动 6.28°，倾角守在 6.84°）。
6. **Gemini 视频默认采样率为 1 fps**（5 秒视频仅 285 VIDEO token）。
   该采样率下模型把"谁屈膝更深"**答反**；设为 12 fps 后两档难度全部答对。
   `media_resolution=HIGH` 只增加每帧空间分辨率、不增加帧数，对时间信息无改善。
7. **`interactions.create` 无法设置 temperature / seed**，必须走 `models.generate_content`。
   温度不可控即实验不可复现。
8. **宫格图在结构上无法传达速度**。按步态相位归一化抽帧会把时间信息彻底抹掉；
   同一组动作改用视频后，模型答出步频比 2.0（真值 1.857）并指出依据是"地面网格上的位移进度"。
9. **固定输出帧数是错的**。`cadence_scale<1` 会静默截断动作内容（只走完 70%），
   `>1` 会让尾部定格——快慢两侧双重标准。改为帧数随步频自适应后，任意步频下内容 100% 覆盖。
10. **LLM 给出的躯干方向优于人工预设**。胸前端重物需后仰配平，LLM 自主选择 −4°/−6°，
    而内置 `heavy` 预设的 +8° 前倾是背负重物的直觉，套在本任务上是错的。

---

## 7. 待解决的问题

### 7.1 阻塞性（必须先解决）

| # | 问题 | 现状 | 可能的解法 |
|---|---|---|---|
| P1 | **space / flow / naturalness 三维无分辨力** | 评语为"与基准基本相当" | ① 观察要点清单（已落地，待验证）② 分机位分别提交视频 ③ 提高渲染真实感 ④ 换更强模型 |
| P2 | **自然度只会扣分，占一半权重** | A_ref 7 分，候选 6-7 分 | ① 改为相对判定（候选 vs 锚点）② 下调权重（但与 workflow §4.2 冲突，需拍板） |
| P3 | **"沉重感"缺乏表达手段** | 模型反复指出"手臂与躯干的沉降受压感不明显" | 在第 0 层词汇表中补充肩部下沉、手臂承重代偿类参数 |

### 7.2 待标定

| # | 事项 | 现状 |
|---|---|---|
| C1 | `patience` / `score_target` / `improvement_margin` | 🔴 占位值 4 / 0.90 / 0.02 |
| C2 | `calibrate_ratios` 四个比例的最优配比 | 🟡 经验值，需消融实验 |
| C3 | 指令跟踪失效阈值 `rel_tol=0.15` | 🟡 拍的，实测 heavy 最大误差 8.6% |

### 7.3 设计债

| # | 事项 |
|---|---|
| D1 | `w_task` / `w_prior` 未开放给 LLM，与 workflow B 类设计不符 |
| D2 | `double_support_offset` 不在指令跟踪审计内（标量特征无法可靠衡量周期内部的时间疏密） |
| D3 | 模型版本钉不死（`gemini-3.8-flash` 的 `version` 字段是 `3.0`，无日期；3.7 已停用并自动路由到 3.8）。当前策略是每轮把 `model_version` / `response_id` / `service_tier` 写进证据包，保证漂移可事后审计 |

---

## 8. 下一阶段目标

按优先级排列，每一项都给出**可证伪的验收标准**：

### 阶段 2a · 打通评分信号（最高优先级）

1. **验证观察要点清单是否让 space / flow 动起来**
   验收：连续 3 轮中，五个维度至少 4 个出现过非零极差。
2. **若无效，改渲染或改提交形态**
   备选顺序：分机位分别提交（3 段/动作）→ 加纹理与方向性光照 → 换 Pro 级模型。
   验收同上。
3. **决定自然度的处理方式**
   需人工拍板：维持绝对打分、改为相对判定、或下调权重。

### 阶段 2b · 标定终止条件

4. 在信号打通后连续跑 10 轮以上，统计分数分布与轮间噪声，
   把 `patience` / `score_target` / `improvement_margin` 从 🔴 升级为 🔵。
   验收：阈值有实测分布支撑，且能解释为何不会被噪声误触发。

### 阶段 2c · 扩展可控性

5. 把 `w_task` / `w_prior` 开放给 LLM（补齐 workflow B 类设计）。
   验收：LLM 能在"动作不够重"与"偏离参考太远"之间做出有据可查的权衡。
6. 补充上肢承重类参数到第 0 层词汇表，回应 P3。
   验收：VLM 的 weight 维度评语不再反复指向"手臂与躯干的沉降受压感"。

### 阶段 2d · 实验设计

7. 引入对照组：人工调参基线、恒等参数基线、随机参数基线，与 agentic loop 对比。
8. 加入物理合理性自动化指标（脚部打滑量、穿地深度）作为客观量化结果，
   这类指标与求解器约束项天然对应，可直接进论文（`workflow.md` §4.6）。
9. 位置偏好偏差检验：`swap_columns_probe` 已预留接口，尚未启用。

---

## 9. 代码资产索引

```
Experiment/src/motionedit/
  第 1 层  rotation.py  skeleton.py  motion_io.py  phase.py
           features.py  rom.py  tray.py  params.py
  第 2 层  targets.py
  第 3 层  losses.py  solver.py
  第 4 层  records.py  history.py  prompts.py  clients.py
           montage.py  renderer.py  outer_loop.py  pipeline.py
Experiment/src/scripts/
  run_solve.py      单次求解
  render_motion.py  多机位渲染
  run_loop.py       闭环驱动（manual / api）
  inspect_run.py    运行检视与信号诊断
  probe_gemini.py   API 能力探针
  fix_tray_mesh.py  物体网格检查与注入
  inspect_reference.py / export_prompts.py
Experiment/src/tests/
  test_core.py  96 项   test_loop.py  149 项
Experiment/src/HYPERPARAMETERS.md   全部可调量清单与默认值
```

约 8500 行 Python。git 仓库为本地存档，无远端。

---

## 10. 给下一个接手者的提醒

本阶段抓到的每一个 bug 都有同一个特征：**不抛异常、测试全绿、结果看起来合理**。

- 托盘轴向写反 → 位姿四元数本身无异常，只有把网格变换到世界系才看得出托盘被立成了板
- `object_models` 丢失 → 渲染器静默替换成 8 顶点占位立方体，画面"有东西"
- `stride_length_scale` 失效 → 求解器选择满足 contact 而把 `task.root` 留在 50.3 不管
- VLM 打分坍缩 → JSON 合法、字段齐全、评语通顺，只是分数没有信息量

**凡是"应当成立的恒等式"，都要写成断言。**
最有价值的一条测试是"恒等参数端到端求解应还原 A_ref"——
它此前只在目标层验过、没在求解器层验过，于是 `tray_chest_distance` 用标量均值压真实波动的 bug 活了很久。
