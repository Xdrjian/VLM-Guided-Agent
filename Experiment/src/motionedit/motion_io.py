"""HUMOTO pkl 动作数据的读写与张量化。

pkl 结构（Solver_Prework.md §0）::

    {
      'armature': [ {bone_name: [qw,qx,qy,qz,tx,ty,tz]}, ... ],   # 逐帧
      'objects' : { 'draw_organizer_tray': [[qw,...,tz], ...] },  # 逐帧位姿
      'object_models': { 'draw_organizer_tray': {'mesh': (verts, faces)} },  # 可选，静态几何
    }

关于 ``object_models``：该字段**可选**，只在渲染时使用。渲染管线的分支是
"有则用内嵌网格，无则去外部 obj 库找，再找不到就生成 8 顶点的占位立方体"——
最后这条路径**不会报错**，只会静默地把真实物体换成一个盒子。因此本模块在
读写时必须原样透传该字段，否则求解器每输出一次动作，物体几何就丢一次。
详见 ``scripts/fix_tray_mesh.py``。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

__all__ = ["MotionSequence", "load_motion", "save_motion", "FPS"]

#: HUMOTO 数据集采样帧率（Solver_Prework.md §2.1）
FPS: float = 24.0


@dataclass
class MotionSequence:
    """一段动作序列的张量化表示。

    Attributes
    ----------
    pose          : ``[T, B, 7]`` 逐帧逐骨骼的局部 ``[quat, trans]``
    bone_names    : 骨骼顺序，与 ``pose`` 的第 1 维对应
    objects       : ``{物体名: [T, 7]}`` 物体的世界位姿
    object_models : ``{物体名: {'mesh': (verts, faces)}}`` 静态几何，仅渲染用；
                    以原始 numpy 形式透传，不参与任何计算
    fps           : 帧率
    """

    pose: Tensor
    bone_names: List[str]
    objects: Dict[str, Tensor]
    fps: float = FPS
    meta: Optional[dict] = None
    object_models: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.pose.shape[0])

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def device(self) -> torch.device:
        return self.pose.device

    def to(self, device: str | torch.device) -> "MotionSequence":
        return MotionSequence(
            pose=self.pose.to(device),
            bone_names=self.bone_names,
            objects={k: v.to(device) for k, v in self.objects.items()},
            fps=self.fps,
            meta=self.meta,
            object_models=self.object_models,
        )

    def slice(self, start: int, end: int) -> "MotionSequence":
        """按帧区间 ``[start, end)`` 截取。物体几何是静态的，原样保留。"""
        return MotionSequence(
            pose=self.pose[start:end],
            bone_names=self.bone_names,
            objects={k: v[start:end] for k, v in self.objects.items()},
            fps=self.fps,
            meta=self.meta,
            object_models=self.object_models,
        )

    def clone(self) -> "MotionSequence":
        return MotionSequence(
            pose=self.pose.clone(),
            bone_names=list(self.bone_names),
            objects={k: v.clone() for k, v in self.objects.items()},
            fps=self.fps,
            meta=dict(self.meta) if self.meta else None,
            object_models=self.object_models,
        )


# --------------------------------------------------------------------------- #
def load_motion(
    path: str | Path,
    bone_order: Optional[Sequence[str]] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    fps: float = FPS,
) -> MotionSequence:
    """加载 pkl 动作数据。

    Parameters
    ----------
    bone_order : 指定骨骼顺序（通常传 :attr:`Skeleton.bone_names` 以保证 FK 对齐）。
                 为 ``None`` 时沿用 pkl 首帧字典的键顺序。
    """
    path = Path(path)
    if path.is_dir():                       # 支持传目录（同名 pkl）
        path = path / f"{path.name}.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    armature = data["armature"]
    if not armature:
        raise ValueError(f"empty armature in {path}")

    names = list(bone_order) if bone_order is not None else list(armature[0].keys())
    missing = [n for n in names if n not in armature[0]]
    if missing:
        raise KeyError(f"{len(missing)} bones missing in pkl, e.g. {missing[:3]}")

    pose = torch.tensor(
        [[frame[n] for n in names] for frame in armature], dtype=dtype, device=device
    )

    objects = {
        k: torch.tensor(v, dtype=dtype, device=device)
        for k, v in data.get("objects", {}).items()
    }

    # 物体几何原样透传（numpy），不转张量：它只被渲染器消费，不参与计算
    object_models = data.get("object_models") or {}

    return MotionSequence(pose=pose, bone_names=names, objects=objects, fps=fps,
                          object_models=object_models)


def save_motion(seq: MotionSequence, path: str | Path) -> Path:
    """把 :class:`MotionSequence` 写回 HUMOTO pkl 格式（可直接喂给官方渲染脚本）。

    ``object_models`` 若存在则一并写回。**不要省略**——缺失时渲染器会静默地
    用 8 顶点占位立方体替代真实物体，画面看上去"有东西"，实则是假的。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pose = seq.pose.detach().cpu().tolist()
    armature = [
        {name: frame[i] for i, name in enumerate(seq.bone_names)} for frame in pose
    ]
    objects = {k: v.detach().cpu().tolist() for k, v in seq.objects.items()}

    payload: Dict[str, Any] = {"armature": armature, "objects": objects}
    if seq.object_models:
        payload["object_models"] = seq.object_models

    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path
