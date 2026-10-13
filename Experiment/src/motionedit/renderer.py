"""多机位渲染器：产出 VLM 评分用宫格图 + 人看的留底视频。

设计要点（2026-09 讨论定稿，勿重新推翻）
------------------------------------------------
1. **两类输出用两套相机**，因为它们的读者不同：

   ======== ============ ============ ==========================================
   输出     相机          地面          理由
   ======== ============ ============ ==========================================
   宫格图   跟随相机      纯白          VLM 只评"重量感/时间/空间/流畅/自然度"五项，
                                      不评脚滑与穿模；把人物放大到填满画格，
                                      才能让 98° 与 123° 的膝屈曲差别活到
                                      模型的 tile 下采样之后。
   留底视频 静止相机      棋盘格        位移与脚滑必须肉眼可见（答辩/自查用）。
                                      跟随相机会把打滑完全藏起来。
   ======== ============ ============ ==========================================

2. **不依赖 ``humoto/render_humoto_pytorch3d.py`` 与 ``load_humoto.py``**。
   已确认用户 Linux 机上的 ``humoto/`` 与本仓库版本不同（多了 ``-s`` 参数和
   ``create_fallback_object_model``）。本模块只用 ``utils/pytorch3d_render_helper.py``
   里的稳定原语和 ``human_model.py``，动作数据一律走本仓库的 :mod:`motion_io`。

3. **物体网格缺失时直接报错，不静默降级**。用户版本的 loader 在缺 mesh 时会画一个
   8 顶点占位立方体——画面"有东西"但是假的，这种失败必须响亮。

4. pytorch3d 只在真正渲染时才 import，保证本模块在无 GPU 的 Windows 上可被导入
   （``tests/test_loop.py`` 依赖这一点）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from . import montage
from .motion_io import MotionSequence, load_motion
from .outer_loop import RenderOutput
from .phase import ContactConfig, detect_gait_phase, sample_frames_by_phase
from .prompts import GridSpec
from .rotation import quat_trans_to_matrix
from .skeleton import END_EFFECTORS, Skeleton

__all__ = [
    "CameraSpec", "DEFAULT_CAMERAS", "RenderConfig",
    "MotionRenderer", "render_motion",
]


# --------------------------------------------------------------------------- #
_HELPER_CACHE: Dict[str, object] = {}


def _helpers(repo_root: Path):
    """惰性加载 ``humoto/utils/pytorch3d_render_helper.py``。

    只取其中的稳定原语（棋盘格、渲染器装配、批量渲染），不碰用户 Linux 机上
    已经分叉的 ``load_humoto.py`` / ``render_humoto_pytorch3d.py``。
    """
    key = str(repo_root)
    mod = _HELPER_CACHE.get(key)
    if mod is None:
        import sys
        p = str(Path(repo_root) / "humoto")
        if p not in sys.path:
            sys.path.insert(0, p)
        from utils import pytorch3d_render_helper as mod  # type: ignore
        _HELPER_CACHE[key] = mod
    return mod


@dataclass(frozen=True)
class CameraSpec:
    """相对被摄者**行进方向**定义的机位，而非世界坐标固定机位。

    这样无论源动作朝哪个方向走，三个机位的语义（正侧面 / 正前方 / 斜前上方）
    都保持一致——否则不同轮次之间的图不可比，VLM 的对比评分就失效了。

    Attributes
    ----------
    azimuth_deg : 0° = 正侧面（站在行进方向左手边），90° = 正前方（迎面）
    elev_deg    : 仰角，0° 为水平视线，正值为俯视
    distance    : 相机到注视点的距离（米）
    height_bias : 注视点相对髋部的竖直偏移（米），略微抬高可兼顾头与脚
    """

    name: str
    azimuth_deg: float
    elev_deg: float
    distance: float = 3.2
    height_bias: float = 0.05


#: 与 ``prompts.GridSpec.cameras`` 的默认顺序严格对应，不得错位
DEFAULT_CAMERAS: Tuple[CameraSpec, ...] = (
    CameraSpec("正侧面", azimuth_deg=0.0, elev_deg=4.0, distance=3.2),
    CameraSpec("正前方", azimuth_deg=90.0, elev_deg=6.0, distance=3.4),
    CameraSpec("45°斜前上方", azimuth_deg=45.0, elev_deg=22.0, distance=3.6),
)


@dataclass
class RenderConfig:
    """渲染参数。默认值按 RTX 5070 Laptop 8 GB 显存留了余量。"""

    #: 骨架/蒙皮模型 json（必须与求解时用的同一个文件）
    human_model_json: Path = Path("humoto/human_model/human_model_up_bone_zup.json")
    tray_name: str = "draw_organizer_tray"
    cameras: Tuple[CameraSpec, ...] = DEFAULT_CAMERAS
    #: 宫格单帧的渲染分辨率（正方形，之后裁剪并缩放到 cell_px）
    grid_render_px: int = 1200
    #: 留底视频单机位分辨率 (H, W)
    video_render_px: Tuple[int, int] = (720, 1280)
    #: 每批渲染帧数，显存不够就调小
    batch_size: int = 8
    video_batch_size: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #: 视频每隔几帧取一帧（1 = 全帧率）
    video_stride: int = 1
    write_video: bool = True
    write_gif: bool = False
    #: GIF 横向最大宽度（三机位拼接后的总宽）
    gif_max_width: int = 1920
    human_color: Tuple[float, float, float] = (144 / 255, 210 / 255, 236 / 255)
    tray_color: Tuple[float, float, float] = (0.85, 0.62, 0.32)
    contact: ContactConfig = field(default_factory=ContactConfig)


# --------------------------------------------------------------------------- #
def _facing_direction(hips_xy: Tensor) -> Tensor:
    """从髋部水平轨迹估计行进方向（单位向量，世界 XY 平面内）。

    行走序列首选首尾位移；若位移过小（原地动作），退化为 PCA 主方向；
    再退化则取世界 +X。返回 ``[2]``。
    """
    disp = hips_xy[-1] - hips_xy[0]
    if float(disp.norm()) > 0.25:
        return disp / disp.norm()

    centered = hips_xy - hips_xy.mean(dim=0, keepdim=True)
    if float(centered.norm()) > 1e-6:
        # 2x2 协方差的主特征向量
        cov = centered.T @ centered
        evals, evecs = torch.linalg.eigh(cov.double())
        v = evecs[:, int(evals.argmax())].to(hips_xy.dtype)
        if float(v.norm()) > 1e-6:
            return v / v.norm()
    return torch.tensor([1.0, 0.0], dtype=hips_xy.dtype, device=hips_xy.device)


def _camera_offset(cam: CameraSpec, forward_xy: Tensor) -> Tensor:
    """把机位规格转换成"相机相对注视点的位置偏移"，``[3]``（Z-up）。"""
    f = torch.tensor([forward_xy[0], forward_xy[1], 0.0],
                     dtype=forward_xy.dtype, device=forward_xy.device)
    # 左手边向量 = up × forward
    left = torch.tensor([-f[1], f[0], 0.0], dtype=f.dtype, device=f.device)

    az = math.radians(cam.azimuth_deg)
    el = math.radians(cam.elev_deg)
    horiz = math.cos(az) * left + math.sin(az) * f
    horiz = horiz / horiz.norm()
    dir3 = math.cos(el) * horiz
    dir3 = torch.stack([dir3[0], dir3[1], torch.tensor(
        math.sin(el), dtype=f.dtype, device=f.device)])
    return dir3 / dir3.norm() * cam.distance


# --------------------------------------------------------------------------- #
class MotionRenderer:
    """实现 :class:`outer_loop.Renderer` 协议。

    典型用法::

        r = MotionRenderer(RenderConfig(device="cuda"), repo_root=REPO)
        out = r(motion_path, out_dir=Path("run/round_003"), grid=GridSpec(), tag="cand")

    重型资源（蒙皮模型、渲染器）按需构建并缓存，同一实例可跨轮次复用。
    """

    def __init__(self, cfg: Optional[RenderConfig] = None,
                 repo_root: Optional[Path] = None,
                 skeleton: Optional[Skeleton] = None):
        self.cfg = cfg or RenderConfig()
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self._model = None
        self._skel = skeleton
        self._renderers: Dict[Tuple[int, int], object] = {}

    # -- 惰性资源 ---------------------------------------------------------- #
    @property
    def model_path(self) -> Path:
        p = Path(self.cfg.human_model_json)
        return p if p.is_absolute() else self.repo_root / p

    @property
    def skeleton(self) -> Skeleton:
        if self._skel is None:
            self._skel = Skeleton.from_json(self.model_path)
        return self._skel

    def _human_model(self):
        if self._model is None:
            import sys
            hm_dir = self.repo_root / "humoto"
            if str(hm_dir) not in sys.path:
                sys.path.insert(0, str(hm_dir))
            from human_model.human_model import HumanModelDifferentiable  # type: ignore
            self._model = HumanModelDifferentiable(
                character_data_path=str(self.model_path), device=self.cfg.device)
        return self._model

    def _renderer(self, image_size: Tuple[int, int]):
        key = tuple(image_size)  # type: ignore[assignment]
        if key not in self._renderers:
            h = _helpers(self.repo_root)
            self._renderers[key] = h.setup_renderer(
                image_size=image_size, device=torch.device(self.cfg.device), y_up=False)[0]
        return self._renderers[key]

    # -- 几何准备 ---------------------------------------------------------- #
    def _skin(self, seq: MotionSequence, frames: Sequence[int]) -> Tuple[Tensor, Tensor]:
        """蒙皮出指定帧的人体网格，返回 ``(verts[B,N,3], faces[F,3])``。"""
        model = self._human_model()
        dev = torch.device(self.cfg.device)
        pose = seq.pose[list(frames)].to(dev)              # [B, Bn, 7]
        mats = quat_trans_to_matrix(pose)                  # [B, Bn, 4, 4]
        params = {name: mats[:, i] for i, name in enumerate(seq.bone_names)}
        with torch.no_grad():
            verts, _ = model(params)
        return verts, model.triangulated_faces_torch

    def _objects(self, seq: MotionSequence, frames: Sequence[int]) -> Dict[str, tuple]:
        """把物体静态网格按每帧位姿变换到世界坐标。

        缺少 ``object_models`` 时**抛错**：静默画占位立方体是本项目已经踩过的坑。
        """
        dev = torch.device(self.cfg.device)
        idx = list(frames)
        out: Dict[str, tuple] = {}
        for name, pose in seq.objects.items():
            model = seq.object_models.get(name)
            if model is None:
                raise RuntimeError(
                    f"动作文件缺少物体 '{name}' 的网格（object_models）。"
                    f"请先运行 scripts/fix_tray_mesh.py --donor Experiment/assets/object_meshes.npz "
                    f"--target <pkl> 注入；缺失时渲染会静默退化为占位立方体。")
            mesh = model["mesh"] if isinstance(model, dict) else model
            v = torch.as_tensor(np.asarray(mesh[0]), dtype=torch.float32, device=dev)
            f = torch.as_tensor(np.asarray(mesh[1]), dtype=torch.long, device=dev)

            m = quat_trans_to_matrix(pose[idx].to(dev))          # [B, 4, 4]
            R, t = m[:, :3, :3], m[:, :3, 3]
            vw = v.unsqueeze(0) @ R.transpose(1, 2) + t[:, None]  # [B, N, 3]
            color = torch.tensor(self.cfg.tray_color, dtype=torch.float32, device=dev)
            out[name] = (vw, f.unsqueeze(0), color)
        return out

    def _phase_frames(self, seq: MotionSequence, n: int) -> List[int]:
        joints = self.skeleton.joint_positions(seq.pose)          # [T, B, 3]
        lt = joints[:, self.skeleton.index(END_EFFECTORS["left_toe"])]
        rt = joints[:, self.skeleton.index(END_EFFECTORS["right_toe"])]
        gp = detect_gait_phase(lt, rt, fps=seq.fps, cfg=self.cfg.contact, up_axis=2)
        return sample_frames_by_phase(gp.phase, n)

    # -- 相机 -------------------------------------------------------------- #
    def _look_at(self, centers: Tensor, offset: Tensor):
        """给定每帧注视点与固定偏移，生成 pytorch3d 的 ``(R, T)``。"""
        from pytorch3d.renderer import look_at_view_transform  # type: ignore
        eye = centers + offset
        up = torch.zeros_like(centers)
        up[:, 2] = 1.0
        return look_at_view_transform(eye=eye, at=centers, up=up)

    def _cameras(self, R, T, image_size: Tuple[int, int]):
        from pytorch3d.renderer import PerspectiveCameras  # type: ignore
        return PerspectiveCameras(
            focal_length=1.73, principal_point=((0.0, 0.0),),
            image_size=(image_size,), R=R, T=T, device=torch.device(self.cfg.device))

    # -- 渲染主体 ---------------------------------------------------------- #
    def _render(self, seq: MotionSequence, frames: Sequence[int],
                centers: Tensor, offset: Tensor,
                image_size: Tuple[int, int], ground, batch: int) -> np.ndarray:
        h = _helpers(self.repo_root)
        dev = torch.device(self.cfg.device)
        renderer = self._renderer(image_size)
        objs = self._objects(seq, frames)

        out: List[np.ndarray] = []
        for s in range(0, len(frames), batch):
            sl = slice(s, min(s + batch, len(frames)))
            hv, hf = self._skin(seq, list(frames)[sl])
            og = {k: (v[0][sl], v[1], v[2]) for k, v in objs.items()}
            R, T = self._look_at(centers[sl].to(dev), offset.to(dev))
            cams = self._cameras(R, T, image_size)
            img = h.render_frames(renderer, cams, hv, hf, og, device=dev, ground=ground)
            out.append((img[..., :3] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy())
            del img
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)

    # -- Renderer 协议 ----------------------------------------------------- #
    def __call__(self, motion_path: Optional[Path], out_dir: Path,
                 grid: GridSpec, tag: str) -> RenderOutput:
        if motion_path is None:
            raise ValueError("motion_path 不能为空：渲染器不负责生成动作")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seq = load_motion(motion_path, bone_order=self.skeleton.bone_names)
        hips = self.skeleton.joint_positions(seq.pose)[:, self.skeleton.index("mixamorig:Hips")]
        forward = _facing_direction(hips[:, :2])

        n = grid.frames_per_camera
        frames = self._phase_frames(seq, n)
        cams = self.cfg.cameras[:len(grid.cameras)] or DEFAULT_CAMERAS

        meta = {
            "motion": str(motion_path),
            "num_frames": seq.num_frames,
            "grid_frames": frames,
            "forward_xy": [float(forward[0]), float(forward[1])],
            "cameras": [c.name for c in cams],
            "device": self.cfg.device,
        }

        # ---- 宫格图：跟随相机 + 纯白背景 ---- #
        gp = self.cfg.grid_render_px
        grids: List[Path] = []
        for ci, cam in enumerate(cams):
            offset = _camera_offset(cam, forward)
            centers = hips[frames].clone()
            centers[:, 2] += cam.height_bias
            imgs = self._render(seq, frames, centers, offset, (gp, gp),
                                ground=None, batch=self.cfg.batch_size)
            sheet = montage.make_grid(list(imgs), rows=grid.rows, cols=grid.cols,
                                      cell_px=grid.cell_px, crop=grid.crop_to_subject)
            p = out_dir / f"{tag}_cam{ci + 1}_{_slug(cam.name)}.png"
            _imwrite(p, sheet)
            grids.append(p)

        # ---- 留底视频：静止相机 + 棋盘格地面 ---- #
        video = gif = None
        if self.cfg.write_video:
            vf = list(range(0, seq.num_frames, max(1, self.cfg.video_stride)))
            ground = _chessboard(torch.device(self.cfg.device), self.repo_root)
            # 注视点固定在整段轨迹的中心，相机因此静止：位移与脚滑才看得见
            center = ((hips.min(0).values + hips.max(0).values) / 2).clone()
            center[2] = float(hips[:, 2].mean())
            span = float((hips.max(0).values - hips.min(0).values)[:2].norm())
            centers = center.unsqueeze(0).expand(len(vf), 3)

            views = []
            for cam in cams:
                far = CameraSpec(cam.name, cam.azimuth_deg, cam.elev_deg,
                                 distance=cam.distance + span / 2.0)
                offset = _camera_offset(far, forward)
                views.append(self._render(seq, vf, centers, offset,
                                          self.cfg.video_render_px, ground,
                                          batch=self.cfg.video_batch_size))
            merged = [montage.hstack_views([v[i] for v in views]) for i in range(len(vf))]
            fps = seq.fps / max(1, self.cfg.video_stride)
            video = montage.write_video(merged, out_dir / f"{tag}_video.mp4", fps=fps)
            if self.cfg.write_gif:
                gif = montage.write_gif(merged, out_dir / f"{tag}_video.gif", fps=fps,
                                        max_width=self.cfg.gif_max_width)
            meta["video_frames"] = len(vf)

        return RenderOutput(grids=grids, video=video, gif=gif, meta=meta)


# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    keep = "".join(ch if ch.isalnum() else "_" for ch in s)
    return keep.strip("_") or "cam"


def _imwrite(path: Path, img: np.ndarray) -> Path:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(img[..., :3]),
                                             cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"写图失败：{path}")
    return path


def _chessboard(device: torch.device, repo_root: Path):
    return _helpers(repo_root).create_chessboard_mesh(
        board_size=20, square_size=0.5, device=device, y_up=False)


def render_motion(motion_path: Path, out_dir: Path, *,
                  cfg: Optional[RenderConfig] = None,
                  grid: Optional[GridSpec] = None,
                  tag: str = "motion",
                  repo_root: Optional[Path] = None) -> RenderOutput:
    """一次性函数式入口（脚本用）。"""
    return MotionRenderer(cfg, repo_root=repo_root)(
        Path(motion_path), Path(out_dir), grid or GridSpec(), tag)
