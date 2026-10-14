"""第 4 层 · 闭环：VLM / LLM prompt 模板（Solver_Prework.md §5.5 / §5.6）。

本模块只负责**渲染纯文本**，不做任何网络调用，因此可在无 API、无 GPU 的环境
下完整测试。手动阶段（A0/A1）直接把渲染结果写文件给人用，自动阶段（B/C）
把同一份文本喂给 API —— 两条路径共用同一套模板，杜绝"手动调好的 prompt
和自动跑的 prompt 不是同一个"这类不可复现来源。

措辞纪律（2026-09 讨论结论）：
LLM 需要建立的是 "参数 A → 结果 B" 的**因果归属**，而非"这是我的功劳"的
自我认同。因此历史记录一律用**中性第三人称**（"第 N 轮采用参数 …"），
只在 system prompt 里一次性说明角色归属，避免触发沉没成本式的自我辩护。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .params import PARAM_BOUNDS, PARAM_NAMES, GaitParams
from .records import RoundRecord, VLMScore

__all__ = [
    "BARS",
    "OBSERVABLES",
    "VLM_SYSTEM",
    "LLM_SYSTEM",
    "GridSpec",
    "VideoSpec",
    "render_vlm_prompt",
    "render_vlm_prompt_round0",
    "render_llm_prompt",
    "extract_json",
]


# --------------------------------------------------------------------------- #
# 行为锚定评分量表（BARS, workflow.md §4.2）
# 高分档位刻意描述为"挑不出毛病的极端情况"，防止打分区间在前几轮就被耗尽。
# --------------------------------------------------------------------------- #
BARS: Dict[str, Dict[int, str]] = {
    "weight": {
        1: "完全看不出负重痕迹，与空手行走无异。",
        2: "有极轻微的负重暗示，但若不被提示则注意不到。",
        3: "能明确感受到在搬运重物，但沉重程度明显不足以对应目标重量。",
        4: "沉重感接近目标重量，仅个别时刻（如蹬离或转体）显得偏轻。",
        5: "与目标重量的预期高度吻合，全程挑不出明显偏轻或夸张之处。",
    },
    "time": {
        1: "节奏与负重状态完全不匹配（如负重却步频轻快跳跃）。",
        2: "节奏偏离较大，快慢与负重语义相悖。",
        3: "节奏方向正确但幅度不足或略显机械。",
        4: "节奏基本贴合负重状态，仅个别相位略显突兀。",
        5: "节奏的持续/加速特征与负重高度一致，无可指摘。",
    },
    "space": {
        1: "空间形态与负重完全无关（步幅、身体占位毫无变化）。",
        2: "空间调整方向可辨但明显不足或过度。",
        3: "步幅/重心/身体轮廓有合理调整，但不够连贯。",
        4: "空间形态贴合负重，仅个别帧的姿态略显勉强。",
        5: "步幅、重心路径与身体占位全程与负重语义高度统一。",
    },
    "flow": {
        1: "动作流动性与负重无关，或呈现无意义的断续。",
        2: "流动特征方向可辨但明显失真。",
        3: "有束缚/受控的流动感，但过渡处不够自然。",
        4: "流动性贴合负重，仅个别过渡略生硬。",
        5: "全程呈现负重应有的受控、连续流动，无可指摘。",
    },
    "naturalness": {
        1: "明显不是人类动作，姿态荒谬。",
        2: "严重违反人体动作直觉。",
        3: "可辨认为行走，但多处姿态怪异。",
        4: "整体像行走，但存在明显不合常理之处。",
        5: "基本像人走路，仍有若干处让人觉得别扭。",
        6: "较为自然，局部（如手臂或躯干）姿态略显僵硬。",
        7: "自然度良好，仅在个别帧能察觉人工痕迹。",
        8: "相当自然，需要仔细看才能发现问题。",
        9: "非常自然，几乎与真实录制的人类动作无异。",
        10: "与真实动作捕捉数据无法区分，完全挑不出毛病。",
    },
}


def _bars_block(dim: str) -> str:
    return "\n".join(f"    {k} 分：{v}" for k, v in sorted(BARS[dim].items()))


# --------------------------------------------------------------------------- #
# 观察要点（2026-10 实测后新增）
# 实测发现：space / flow 两维度在参数显著变化时仍给出"与基准基本相当"。
# 原因不是不敢给分（打分纪律修复后 weight/time 已能动），而是这两个构念
# 本身太抽象，模型无从下手。故把它们拆成具体可看的量。
# 只告诉它"看哪里"，不告诉它"应该看到什么"——后者是诱导性提问。
# --------------------------------------------------------------------------- #
OBSERVABLES: Dict[str, str] = {
    "weight": "髋部在整段中的高度起伏；支撑腿在承重时的屈曲程度；"
              "躯干相对垂直的倾角及其方向（前倾还是后仰）；"
              "上臂与躯干的夹角、肘部是否收紧；脚掌着地瞬间有无可见的缓冲下沉。",
    "time": "整段视频的总时长；走完一个完整步态周期所需的秒数；"
            "双脚同时着地的持续时长占周期的比例；摆动相与支撑相的快慢对比。",
    "space": "迈步最大时两脚之间的距离（可用地面棋盘格计数）；"
             "髋部离地高度的绝对值与波动范围；躯干倾角；"
             "双臂与躯干的开合幅度；托盘距离胸部的远近；"
             "整段在棋盘格地面上走过的总距离。",
    "flow": "关节角度随时间变化是否连续、有无突然的顿挫或抑制；"
            "重心轨迹的起伏是平滑还是冒进；重心在两腿间转移时是否干脆；"
            "抬起/落下的加减速是渐进还是突变。",
    "naturalness": "关节有无反向弯曲或超出人体范围的扭转；左右侧是否对称；"
                   "四肢与躯干的协调关系是否符合人体运动直觉；"
                   "有无抽搐、闪烁、滑步以外的怪异现象。",
}


def _observables_block() -> str:
    return "\n".join(f"- {k}：{v}" for k, v in OBSERVABLES.items())


# --------------------------------------------------------------------------- #
VLM_SYSTEM = (
    "你是一位动作质量评估专家，负责对渲染出的人体行走动作做结构化评分。\n"
    "你的评分将被用于一个自动化迭代流程，因此必须严格遵守给定的分档与输出格式。\n"
    "你只输出一个 JSON 对象，不要输出任何解释性前后文、不要使用代码块以外的散文。"
)

LLM_SYSTEM = (
    "你是一个动作生成流程中的参数决策模块。\n"
    "以下提供的历史记录，均为本流程历次决策所采用的参数及其产生的后果；"
    "你的职责是基于这些因果记录给出下一轮参数，而不是为任何一次历史决策辩护。\n"
    "你只输出一个 JSON 对象，不要输出任何解释性前后文。"
)


# --------------------------------------------------------------------------- #
class GridSpec:
    """宫格图规格。A0 阶段实测后再定稿，因此全部做成可配置项而非硬编码。

    分辨率纪律（2026-09 讨论结论）：多模态模型会把图像切成固定大小的 tile，
    2×4 宫格中单帧只占 1/8 面积，若整图被下采样，膝屈曲 98° 与 123° 的差别
    可能直接丢失。因此必须：(a) 渲染到目标模型的原生最大分辨率，
    (b) 裁掉无效背景让人物填满每格。
    """

    def __init__(self, rows: int = 2, cols: int = 4,
                 cell_px: int = 512, crop_to_subject: bool = True,
                 cameras: Sequence[str] = ("正侧面", "正前方", "45°斜前上方")):
        if rows * cols <= 0:
            raise ValueError("rows * cols 必须为正")
        self.rows, self.cols = int(rows), int(cols)
        self.cell_px = int(cell_px)
        self.crop_to_subject = bool(crop_to_subject)
        self.cameras = tuple(cameras)

    @property
    def frames_per_camera(self) -> int:
        return self.rows * self.cols

    @property
    def items_per_motion(self) -> int:
        """一个动作需要提交几份素材。"""
        return len(self.cameras)

    @property
    def media_noun(self) -> str:
        return "张宫格图"

    #: 在 ``RoundRecord.artifacts`` 中的键名，用于回取历史最佳解的素材
    artifact_key = "grids"

    @staticmethod
    def select(render) -> list:
        """从一次渲染产物中挑出要提交给 VLM 的文件。"""
        return list(render.grids)

    def layout_block(self) -> str:
        n = len(self.cameras)
        return (f"- **左列**（前 {n} 张）：当前**历史最佳解**的 {n} 个机位。\n"
                f"- **右列**（后 {n} 张）：**本轮候选解**的 {n} 个机位。\n"
                f"- 左右同一行为同一机位，可直接逐帧对比。\n"
                + self.describe())

    @property
    def grid_px(self) -> tuple:
        return (self.cols * self.cell_px, self.rows * self.cell_px)

    def describe(self) -> str:
        cams = "、".join(f"{i+1}={c}" for i, c in enumerate(self.cameras))
        return (f"每张宫格图为 {self.rows} 行 × {self.cols} 列，共 {self.frames_per_camera} 帧，"
                f"按**从左到右、从上到下**的顺序对应一个完整步态周期内的 "
                f"{self.frames_per_camera} 个关键姿态时刻（按步态相位均匀采样，非等时间间隔）。"
                f"\n共 {len(self.cameras)} 个机位，依次为：{cams}。")

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "cols": self.cols, "cell_px": self.cell_px,
                "crop_to_subject": self.crop_to_subject, "cameras": list(self.cameras),
                "grid_px": list(self.grid_px)}


class VideoSpec:
    """视频素材规格。A0 探针实测后取代宫格图成为 VLM 的默认输入形态。

    选视频而非宫格图的决定性证据（2026-10，gemini-3.8-flash 实测）：
    宫格图按步态相位归一化抽帧，**时间信息被彻底抹掉**，模型无从判断快慢；
    而同一组动作改用视频后，模型答出"乙比甲快约 2.0 倍"（真值 1.857，误差 7.7%），
    并明确指出依据是"地面网格上的位移进度"——静止相机 + 棋盘格地面的设计在此兑现。
    """

    def __init__(self, cameras: Sequence[str] = ("正侧面", "正前方", "45°斜前上方"),
                 fps: float = 24.0, stacked: bool = True):
        self.cameras = tuple(cameras)
        self.fps = float(fps)
        self.stacked = bool(stacked)

    @property
    def items_per_motion(self) -> int:
        return 1 if self.stacked else len(self.cameras)

    @property
    def media_noun(self) -> str:
        return "段视频"

    artifact_key = "video"

    def select(self, render) -> list:
        if getattr(render, "video", None) is None:
            raise ValueError(
                "渲染产物中没有视频。VideoSpec 下不得用 --no-video 渲染，"
                "否则 prompt 声称提交的是视频、实际附件却不是。")
        return [render.video]

    def describe(self) -> str:
        cams = "、".join(f"{i + 1}={c}" for i, c in enumerate(self.cameras))
        if self.stacked:
            return (f"每段视频把 {len(self.cameras)} 个机位**横向拼接**在同一画面中，"
                    f"从左到右依次为：{cams}。\n"
                    f"相机固定不动，地面为棋盘格，因此人物相对地面的位移是真实可读的。")
        return f"每个动作按机位顺序提交 {len(self.cameras)} 段视频：{cams}。"

    def layout_block(self) -> str:
        n = self.items_per_motion
        first = "第一段" if n == 1 else f"前 {n} 段"
        second = "第二段" if n == 1 else f"后 {n} 段"
        return (f"- **{first}**：当前**历史最佳解**。\n"
                f"- **{second}**：**本轮候选解**。\n"
                f"- 两者的**时长可能不同**。这是真实的动作差异（步频不同导致走完同一段"
                f"动作所需时间不同），**不是渲染错误**，请把它作为判断依据之一。\n"
                + self.describe())

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "video", "cameras": list(self.cameras),
                "fps": self.fps, "stacked": self.stacked}


# --------------------------------------------------------------------------- #
def _fmt_params(p: GaitParams) -> str:
    return json.dumps(p.to_dict(), ensure_ascii=False, indent=2)


def _fmt_score(s: Optional[VLMScore]) -> str:
    if s is None:
        return "（无评分记录）"
    return json.dumps(s.to_dict(), ensure_ascii=False, indent=2)


def _fmt_features(f: Dict[str, float]) -> str:
    if not f:
        return "（无特征记录）"
    return json.dumps({k: (round(v, 4) if isinstance(v, (int, float)) else v)
                       for k, v in f.items()}, ensure_ascii=False, indent=2)


_JUDGE_SCOPE = """【评判对象的范围】
你评判的是**动作本身**，不是画面质量。渲染用的是无贴图、无面部的素模，
这是实验设定，**不得因此扣分**。请把它当作动作捕捉数据的标准可视化方式，
只判断骨骼运动本身是否符合人体动作直觉、是否体现目标重量的语义特征。"""

_SCORING_RULES = f"""请仅评估以下维度。**不要评估物理合理性**（是否浮空、穿地、脚部打滑、
关节是否超出生理限位等），这些已由求解器的损失函数与几何检测机制保证，不属于你的职责范围；
重复评判它们会与既有机制冲突。

- Weight（1-5 分，对应"沉重感"，是本任务的核心维度）：
{_bars_block("weight")}
- Time（1-5 分）：
{_bars_block("time")}
- Space（1-5 分）：
{_bars_block("space")}
- Flow（1-5 分）：
{_bars_block("flow")}
- 自然度 naturalness（1-10 分）：
{_bars_block("naturalness")}

【打分纪律】
1. **请充分使用整个分档区间。** 分数必须与你自己的评语一致：若评语写的是
   "几乎看不出负重"，就不应给中间分；若评语写的是"自然度良好"，就不应给中等分。
   逐档对照上面的文字锚点，挑出描述最贴切的那一档，而不是凭印象取中间值。
2. **不要为了"留余地"而压低分数，也不要为了保守而向中间值靠拢。**
   分数要反映你当下看到的事实；后续轮次如果更好，自然会拿到更高的分。
3. 1 分与 5 分（1-10 档的 1 分与 10 分）都是正常可用的取值，不是禁区。

【每个维度该看哪里】
打分前请逐项比对以下可观察量，并在评语中引用你实际看到的具体现象，
避免停留在"基本相当""未见显著变化"这类笼统描述：
{_observables_block()}

{_JUDGE_SCOPE}"""

_VLM_OUTPUT_FORMAT = """请严格以如下 JSON 格式输出（不要添加任何额外字段或文字）：
{"weight": int, "time": int, "space": int, "flow": int, "naturalness": int,
 "comments": {"weight": str, "time": str, "space": str, "flow": str, "naturalness": str}}"""

_VLM_OUTPUT_FORMAT_PAIR = """请严格以如下 JSON 格式输出（不要添加任何额外字段或文字）：
{"comparison": {"weight": "better"|"worse"|"same", "time": ..., "space": ...,
                "flow": ..., "naturalness": ...},
 "weight": int, "time": int, "space": int, "flow": int, "naturalness": int,
 "comments": {"weight": str, "time": str, "space": str, "flow": str, "naturalness": str}}

字段含义与硬性要求：
- ``comparison``：逐维度判定**本轮候选解相比历史最佳解**是更好/更差/无可见差异。
  请先完成这一步，再决定分数。
- 五个整数分数是给**本轮候选解**的。
- **一致性要求（必须满足）**：某维度判为 ``better`` 时，该维度分数必须
  **严格高于**历史最佳解在该维度的分数；判为 ``worse`` 时必须**严格低于**；
  只有判为 ``same`` 时才允许相等。切勿出现"评语说有改善、分数却不变"。"""


def render_vlm_prompt(round_index: int, target_weight_kg: float,
                      best: Optional[RoundRecord],
                      grid: Optional[Any] = None) -> str:
    """常规轮次的 VLM prompt（历史最佳解在前、本轮候选解在后，同场对比）。

    ``grid`` 接受 :class:`GridSpec` 或 :class:`VideoSpec`，二者提供同一套
    ``items_per_motion`` / ``media_noun`` / ``layout_block`` 接口。

    对比锚点固定取**历史最佳解**而非紧邻上一轮：若上一轮是被回滚的失败尝试，
    以它作基准会让 VLM 评语的参照系与 LLM 的实际调参基底错位（workflow.md §4.2）。
    """
    spec = grid or GridSpec()
    n_items = 2 * spec.items_per_motion
    anchor = ""
    if best is not None:
        anchor = (f"\n作为对比锚点，历史最佳解在其对应轮次（第 {best.round_index} 轮）"
                  f"获得的完整评分为：\n{_fmt_score(best.score)}\n"
                  f"若本轮候选解确实更好，其分数不应低于上述分数；"
                  f"若更差，请如实给出更低的分数。请避免一律给出中庸评价。\n")
    return f"""以下是同一个"负重行走"任务的 {n_items} {spec.media_noun}。

【素材组织规则】
{spec.layout_block()}

【任务上下文】
任务目标：使动作看起来像是端着 {target_weight_kg:g}kg 的托盘行走。该目标全程不变。
当前为第 {round_index} 轮迭代。{anchor}
【评判维度与评分量表】
{_SCORING_RULES}

【输出格式】
{_VLM_OUTPUT_FORMAT_PAIR if best is not None else _VLM_OUTPUT_FORMAT}
"""


def render_vlm_prompt_round0(target_weight_kg: float,
                             grid: Optional[Any] = None) -> str:
    """第 0 轮专用 prompt（workflow.md §4.4）。

    必须明确告知这是未经处理的动捕真值。理由：A_ref 的自然度理应是所有候选中
    最高的，若 VLM 给出偏低分数，大概率暴露的是 VLM/prompt/渲染管线的问题，
    这构成一次**低成本的 VLM 可靠性验证**。Weight 维度仍需如实评分——
    A_ref 是空载/轻载，其 Weight 分数没有预设的"正确值"，是真正有意义的判断。
    """
    spec = grid or GridSpec()
    return f"""以下是 {spec.items_per_motion} {spec.media_noun}，来自**未经任何处理的人体动作捕捉参考数据
（ground truth，真实人类表演的录制结果）**。
{spec.describe()}

【任务上下文】
后续流程的目标是：使动作看起来像是端着 {target_weight_kg:g}kg 的托盘行走。
本轮（第 0 轮）展示的是**尚未做任何编辑**的原始参考动作，用于建立评分基线。
请注意：由于这是真实录制的人类动作，它在自然度上理应处于很高的水平；
但它的负重状态是未知的，请对 Weight 维度**如实独立判断**，不要因为它是真值就给高分。

【评判维度与评分量表】
{_SCORING_RULES}

【输出格式】
{_VLM_OUTPUT_FORMAT}
"""


# --------------------------------------------------------------------------- #
_PARAM_DOC = {
    "knee_flex_scale": "膝屈曲角乘性缩放，>1 屈膝更深",
    "trunk_lean_offset": "躯干前倾加性偏置（度），正值更前倾",
    "cadence_scale": "步频乘性缩放，>1 步数更多",
    "double_support_offset": "双支撑相占比加性偏置（归一化比例），正值双脚同时着地更久",
    "stride_length_scale": "步长乘性缩放，作用于根节点水平位移",
    "elbow_flex_offset": "肘屈曲加性偏置（度），正值肘部更收紧、托盘更贴身",
    "tray_chest_distance_scale": "托盘到胸骨的水平距离乘性缩放，<1 更贴近胸前",
}


def _param_table() -> str:
    rows = []
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        rows.append(f"- {name}：{_PARAM_DOC[name]}；取值范围 [{lo}, {hi}]")
    return "\n".join(rows)


def render_llm_prompt(round_index: int, target_weight_kg: float,
                      best: Optional[RoundRecord],
                      recent: Sequence[RoundRecord] = (),
                      last_features: Optional[Dict[str, float]] = None,
                      solver_warnings: Sequence[str] = ()) -> str:
    """LLM 调参 prompt（Solver_Prework.md §5.6）。

    传入的永远是**分项打分 + 文字评语**，绝不传加权总分（workflow.md §4.2）——
    聚合数字会让 LLM 失去"该往哪个方向调"的判断依据。
    """
    if best is None:
        base_block = ("【基底】尚无已接受的历史最佳解。请给出一组保守的初始参数，"
                      "相对恒等参数（全部为 1.0 / 0.0）做小幅调整即可。\n")
    else:
        kind = "参考动作 A_ref（未经编辑）" if best.is_reference else f"第 {best.round_index} 轮的解"
        base_block = (
            f"【历史最佳解 — 本轮参数修改的基底】来源：{kind}\n"
            f"参数：\n{_fmt_params(best.params)}\n"
            f"VLM 分项评分与评语：\n{_fmt_score(best.score)}\n"
            f"其生物力学步态特征：\n{_fmt_features(best.features)}\n"
        )

    if recent:
        blocks = []
        for r in recent:
            tag = "已回滚" if r.rolled_back else "未被接受"
            blocks.append(
                f"第 {r.round_index} 轮（{tag}）\n"
                f"采用参数：\n{_fmt_params(r.params)}\n"
                f"VLM 分项评分与评语：\n{_fmt_score(r.score)}\n"
                f"实际生物力学特征：\n{_fmt_features(r.features)}\n"
            )
        recent_block = ("【最近轮次记录 — 仅作为方向参考，不作为本轮修改基底】\n"
                        + "\n".join(blocks))
    else:
        recent_block = "【最近轮次记录】无。\n"

    feat_block = ""
    if last_features:
        feat_block = (
            "【求解器执行反馈】上一轮下达的参数，实际被求解器执行后测得的生物力学特征：\n"
            f"{_fmt_features(last_features)}\n"
            "若某个参数指令与其实际执行效果存在显著背离（例如指定膝屈曲放大而实测曲线几乎未变），"
            "请在 reasoning 中明确指出，这可能意味着求解器执行失效而非参数方向错误。\n"
        )
    warn_block = ""
    if solver_warnings:
        warn_block = "【告警】\n" + "\n".join(f"- {w}" for w in solver_warnings) + "\n"

    out_fields = ",\n ".join(f'"{n}": float' for n in PARAM_NAMES)
    return f"""任务目标：使动作看起来像是端着 {target_weight_kg:g}kg 的托盘行走。该目标全程不变。
当前需要给出第 {round_index} 轮的参数。

【可调参数说明】
{_param_table()}

{base_block}
{recent_block}
{feat_block}{warn_block}
【要求】
1. 以**历史最佳解**为基底进行修改；被回滚的轮次只用于判断"哪个方向不奏效"，不作为延拓起点。
2. 输出**完整的一组绝对数值**，而非增量或百分比调整。
3. 所有数值必须落在上述取值范围内。

请严格以如下 JSON 格式输出（不要添加任何额外文字）：
{{{out_fields},
 "reasoning": str}}
"""


# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Dict[str, Any]:
    """从模型返回的原文中稳健地抽出第一个 JSON 对象。

    兼容三种常见形态：纯 JSON、被 ```json 代码块包裹、JSON 前后有散文。
    解析失败时抛 ``ValueError`` 并附带原文片段，便于事后复盘。
    """
    s = text.strip()
    # 去掉代码块围栏
    if "```" in s:
        parts = s.split("```")
        for i in range(1, len(parts), 2):
            chunk = parts[i]
            if chunk.lstrip().lower().startswith("json"):
                chunk = chunk.lstrip()[4:]
            chunk = chunk.strip()
            if chunk.startswith("{"):
                s = chunk
                break
    start = s.find("{")
    if start < 0:
        raise ValueError(f"返回内容中找不到 JSON 对象：{text[:300]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON 解析失败：{e}；片段={s[start:i+1][:300]!r}") from e
    raise ValueError(f"JSON 对象未闭合：{text[:300]!r}")
