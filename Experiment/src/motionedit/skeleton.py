"""静止骨架加载、层级结构与批量前向运动学。

FK 约定（与 ``humoto/human_model/human_model.py`` 的 ``compute_bone_transforms`` 一致）::

    world[root]  = Rest[root] @ Pose[root]
    world[child] = world[parent] @ inv(Rest[parent]) @ Rest[child] @ Pose[child]

其中 ``Rest[b]`` 为 JSON 中该骨骼的 4x4 静止世界变换，``Pose[b]`` 为 pkl 中该骨骼
相对静止姿态的局部 4x4 变换（由 ``[w,x,y,z,tx,ty,tz]`` 转换而来）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .rotation import quat_trans_to_matrix

__all__ = ["Skeleton", "CORE_BONES", "END_EFFECTORS"]


#: Solver_Prework.md §1.1 / §4.3 的可优化自由度白名单（21 根核心骨骼）
CORE_BONES: tuple[str, ...] = (
    "mixamorig:Hips",
    "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
    "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase",
    "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
    "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
    "mixamorig:Neck",
)

#: 常用末端点（接触约束 / 特征提取用）
END_EFFECTORS: Dict[str, str] = {
    "left_foot": "mixamorig:LeftFoot",
    "left_toe": "mixamorig:LeftToeBase",
    "right_foot": "mixamorig:RightFoot",
    "right_toe": "mixamorig:RightToeBase",
    "left_hand": "mixamorig:LeftHand",
    "right_hand": "mixamorig:RightHand",
}


@dataclass
class Skeleton:
    """静止骨架 + 层级 + FK。

    Attributes
    ----------
    bone_names : 按拓扑序排列的骨骼名（父骨骼一定排在子骨骼之前）
    parent_idx : 每根骨骼的父骨骼下标，根骨骼为 ``-1``
    rest       : ``[B, 4, 4]`` 静止世界变换
    rest_inv   : ``[B, 4, 4]`` 静止世界变换的逆
    """

    bone_names: List[str]
    parent_idx: Tensor           # [B] int64
    rest: Tensor                 # [B, 4, 4]
    rest_inv: Tensor             # [B, 4, 4]
    vertices: Optional[Tensor] = None
    faces: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        load_mesh: bool = False,
    ) -> "Skeleton":
        """从 ``human_model_*_zup.json`` / ``*_yup.json`` 加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bones = data["bones"]
        order = _topological_order(bones)
        name_to_idx = {n: i for i, n in enumerate(order)}

        rest = torch.tensor(
            [bones[n]["matrix"] for n in order], dtype=dtype, device=device
        )
        parent_idx = torch.tensor(
            [name_to_idx[bones[n]["parent"]] if bones[n]["parent"] else -1 for n in order],
            dtype=torch.long,
            device=device,
        )

        verts = faces = None
        if load_mesh:
            verts = torch.tensor(data["vertices"], dtype=dtype, device=device)
            faces = _triangulate(data["faces"], device=device)

        return cls(
            bone_names=order,
            parent_idx=parent_idx,
            rest=rest,
            rest_inv=torch.linalg.inv(rest.double()).to(dtype),
            vertices=verts,
            faces=faces,
        )

    # ------------------------------------------------------------------ #
    @property
    def num_bones(self) -> int:
        return len(self.bone_names)

    @property
    def device(self) -> torch.device:
        return self.rest.device

    def index(self, bone: str) -> int:
        return self.bone_names.index(bone)

    def indices(self, bones: Sequence[str]) -> Tensor:
        return torch.tensor([self.index(b) for b in bones], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ #
    def dof_mask(self, active: Sequence[str] = CORE_BONES) -> Tensor:
        """返回 ``[B]`` 的 0/1 mask（Solver_Prework.md §4.3 的显式接口）。"""
        mask = torch.zeros(self.num_bones, dtype=torch.bool, device=self.device)
        for b in active:
            if b in self.bone_names:
                mask[self.index(b)] = True
        return mask

    # ------------------------------------------------------------------ #
    def forward_kinematics(self, pose: Tensor) -> Tensor:
        """批量 FK。

        Parameters
        ----------
        pose : ``[T, B, 7]``（四元数+平移）或 ``[T, B, 4, 4]`` 的局部姿态。

        Returns
        -------
        ``[T, B, 4, 4]`` 各骨骼的世界变换（顺序同 :attr:`bone_names`）。
        """
        if pose.shape[-1] == 7 and pose.dim() == 3:
            pose = quat_trans_to_matrix(pose)
        if pose.shape[-2:] != (4, 4):
            raise ValueError(f"unexpected pose shape {tuple(pose.shape)}")

        T = pose.shape[0]
        rest = self.rest.to(pose.dtype)
        rest_inv = self.rest_inv.to(pose.dtype)

        # local_to_parent[b] = inv(Rest[parent]) @ Rest[b] @ Pose[b]（根骨骼取 Rest[b] @ Pose[b]）
        world: List[Tensor] = [torch.empty(0)] * self.num_bones
        for b in range(self.num_bones):
            p = int(self.parent_idx[b])
            if p < 0:
                world[b] = rest[b].expand(T, 4, 4) @ pose[:, b]
            else:
                delta = rest_inv[p] @ rest[b]
                world[b] = world[p] @ (delta.expand(T, 4, 4) @ pose[:, b])
        return torch.stack(world, dim=1)

    def joint_positions(self, pose: Tensor) -> Tensor:
        """FK 后各骨骼原点的世界坐标，``[T, B, 3]``。"""
        return self.forward_kinematics(pose)[..., :3, 3]


# --------------------------------------------------------------------------- #
def _topological_order(bones: dict) -> List[str]:
    """按父子关系排序，保证父骨骼下标小于子骨骼。"""
    children: Dict[Optional[str], List[str]] = {}
    for name, info in bones.items():
        children.setdefault(info["parent"], []).append(name)

    order: List[str] = []
    stack = list(reversed(children.get(None, [])))
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children.get(node, [])))

    if len(order) != len(bones):          # 存在孤立节点（父骨骼名不在表内）时兜底
        order += [n for n in bones if n not in set(order)]
    return order


def _triangulate(faces: Sequence[Sequence[int]], device) -> Tensor:
    """把可能包含四边形的面片统一三角化。"""
    tris: List[List[int]] = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
    return torch.tensor(tris, dtype=torch.long, device=device)
