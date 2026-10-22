"""motionedit: 负重行走动作编辑实验的核心库。

模块划分（按依赖顺序）：

第 1 层 · 地基
- rotation  : 四元数/旋转矩阵/欧拉角/增量参数化（全部可微）
- skeleton  : 静止骨架加载、父子层级、批量前向运动学、DOF mask
- motion_io : pkl 动作数据读写与张量化
- phase     : 步态相位识别（足部接触 → 归一化相位）
- features  : 生物力学步态特征提取
- rom       : 关节活动度约束（swing-twist 软惩罚）
- tray      : 托盘位姿更新（双手驱动）
- params    : 7 个中介参数的定义与校验

第 2 层 · 目标
- targets   : 参数 → 逐帧目标曲线（时间重映射 + 姿态/根节点/托盘目标）

第 3 层 · 求解
- losses    : 五项损失（task/prior/ROM/smooth/contact）+ 退火 + 自动定标
- solver    : 求解器主体（增量旋转向量参数化 + Adam → L-BFGS）

第 4 层 · 闭环
- records   : 单轮记录与证据包（可复现性的落盘基础）
- history   : 历史最佳锚点 / 回滚语义 / 终止条件
- prompts   : VLM & LLM prompt 模板（BARS 量表、中性措辞历史）
- clients   : 模型调用客户端（手动 / API 双实现，同一接口）
- montage   : 宫格拼图与留底视频（仅依赖 numpy + opencv）
- renderer  : 多机位相位抽帧渲染（宫格用跟随相机，留底视频用静止相机+棋盘格）
- outer_loop: 外层循环主体（渲染与求解以注入点接入）

注：第 4 层中 ``montage``（需 opencv）与实机渲染器（需 pytorch3d）
只能在配置好的 Linux 环境运行；其余模块在纯 CPU 环境即可完整单测。
"""

from .params import GaitParams, DEFAULT_PARAMS, PARAM_BOUNDS

__all__ = ["GaitParams", "DEFAULT_PARAMS", "PARAM_BOUNDS"]
__version__ = "0.4.0"
