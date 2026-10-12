"""把物体网格（``object_models``）注入到缺失该字段的 pkl 中。

背景：物体的**逐帧位姿**在 ``pkl['objects']``（一直都有），物体的**几何体**在
``pkl['object_models']``（可选）。渲染管线的分支逻辑是：

    if 'object_models' in data:   # 优先：用 pkl 内嵌网格
        ...
    else:                          # 兜底：去 HUMOTO_OBJECT_DIR 找 .obj；
                                   # 再找不到则生成 8 顶点的占位立方体

不同版本的 ``extract_pk_data.py`` 对是否默认提取网格的行为不一致，导致同一条
流水线可能产出缺 ``object_models`` 的 pkl。此时渲染**不会报错**，而是静默地
画出一个占位立方体——这是最危险的失败模式：画面里"有托盘"，但它是假的。

本脚本把已验证含真实网格的 pkl 作为供体，将 ``object_models`` 原样复制到目标
pkl。物体网格是静态的（只有位姿逐帧变），因此直接复制在几何上是正确的。

**推荐做法**：先用 ``--export`` 把网格导出成仓库内的 ``.npz`` 资产，之后一律以
该 npz 为供体。这样流水线不再依赖任何外部机器上的 pkl，成为自包含的单一真相来源。

用法::

    # 一次性：从供体 pkl 导出网格资产到仓库
    python scripts/fix_tray_mesh.py --donor <含网格的.pkl> --export assets/object_meshes.npz

    # 之后：用仓库内的 npz 修复任意 pkl（单文件或整个目录）
    python scripts/fix_tray_mesh.py --donor assets/object_meshes.npz --target <目录> --recursive

    # 只检查不修改
    python scripts/fix_tray_mesh.py --check <目录> --recursive
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: fallback 占位立方体的顶点数——用于识别"假托盘"
FALLBACK_VERT_COUNT = 8


def _load(path: Path) -> Dict[str, Any]:
    """读取 pkl，或读取 npz 资产并伪装成 ``{'object_models': ...}`` 结构。"""
    if path.suffix.lower() == ".npz":
        return {"objects": {}, "object_models": _load_npz(path)}
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_npz(path: Path) -> Dict[str, Any]:
    """从 npz 资产还原 ``object_models``。"""
    data = np.load(path, allow_pickle=False)
    names = sorted({k.rsplit("__", 1)[0] for k in data.files})
    out: Dict[str, Any] = {}
    for n in names:
        v, f = data[f"{n}__verts"], data[f"{n}__faces"]
        out[n] = {"mesh": (v, f)}
    return out


def export_npz(models: Dict[str, Any], path: Path) -> Path:
    """把 ``object_models`` 存成仓库可入库的 npz 资产。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {}
    for name, entry in models.items():
        v, f = entry["mesh"]
        arrays[f"{name}__verts"] = np.asarray(v, dtype=np.float32)
        arrays[f"{name}__faces"] = np.asarray(f, dtype=np.int64)
    np.savez_compressed(path, **arrays)
    return path


def _mesh_shape(entry: Any) -> tuple:
    """从 ``object_models[name]`` 取出 (顶点数, 面片数)，取不到返回 (0, 0)。"""
    try:
        v, f = entry["mesh"]
        return len(v), len(f)
    except (KeyError, TypeError, ValueError):
        return 0, 0


def inspect(path: Path) -> Dict[str, Any]:
    """检查一个 pkl（或 npz 资产）的物体几何状况。"""
    d = _load(path)
    models = d.get("object_models")
    # npz 资产没有位姿轨道，用网格名代替物体名
    objects = list(d.get("objects", {}).keys()) or list((models or {}).keys())
    info: Dict[str, Any] = {
        "path": path,
        "objects": objects,
        "has_models": models is not None,
        "meshes": {},
        "status": "",
        "suspicious": [],
    }
    if models is None:
        info["status"] = "MISSING"        # 渲染时会退化成占位立方体
        return info

    suspicious = []
    for name in objects:
        nv, nf = _mesh_shape(models.get(name))
        info["meshes"][name] = (nv, nf)
        if nv == 0:
            suspicious.append(f"{name}: 无网格")
        elif nv <= FALLBACK_VERT_COUNT:
            suspicious.append(f"{name}: 仅 {nv} 顶点，疑似占位立方体")
    info["status"] = "FALLBACK?" if suspicious else "OK"
    info["suspicious"] = suspicious
    return info


def report(paths: List[Path]) -> int:
    bad = 0
    for p in paths:
        info = inspect(p)
        mark = {"OK": "✓", "MISSING": "✗", "FALLBACK?": "!"}[info["status"]]
        detail = ", ".join(f"{k}: {v[0]} 顶点/{v[1]} 面片"
                           for k, v in info["meshes"].items()) or "—"
        print(f"  [{mark}] {p.name}")
        print(f"        物体轨道: {info['objects']}")
        print(f"        内嵌网格: {detail}")
        for s in info.get("suspicious", []):
            print(f"        ⚠ {s}")
        if info["status"] != "OK":
            bad += 1
    return bad


def inject(donor: Path, target: Path, backup: bool = True,
           force: bool = False) -> bool:
    """把 ``donor`` 的 ``object_models`` 复制进 ``target``。返回是否发生了修改。"""
    d_info = inspect(donor)
    if d_info["status"] != "OK":
        raise ValueError(
            f"供体 {donor} 自身的网格不可用（状态 {d_info['status']}），拒绝作为来源。")

    t = _load(target)
    t_info = inspect(target)
    if t_info["status"] == "OK" and not force:
        print(f"  [跳过] {target.name} 已有合法网格")
        return False

    donor_models = _load(donor)["object_models"]
    missing = [n for n in t.get("objects", {}) if n not in donor_models]
    if missing:
        raise KeyError(
            f"供体缺少目标所需的物体网格：{missing}（供体含 {list(donor_models)}）")

    if backup:
        bak = target.with_suffix(target.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(target, bak)

    # 只复制目标实际用到的物体，避免带入无关几何
    t["object_models"] = {n: donor_models[n] for n in t["objects"]}
    with open(target, "wb") as f:
        pickle.dump(t, f)

    shapes = ", ".join(f"{n}: {_mesh_shape(t['object_models'][n])[0]} 顶点"
                       for n in t["objects"])
    print(f"  [已注入] {target.name} ← {shapes}")
    return True


def _collect(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*.pkl" if recursive else "*.pkl"
    return sorted(p for p in root.glob(pattern) if not p.name.endswith(".bak"))


def main() -> int:
    ap = argparse.ArgumentParser(description="检查/修复 pkl 中缺失的物体网格")
    ap.add_argument("--donor", type=Path, help="含真实网格的供体（.pkl 或 .npz）")
    ap.add_argument("--target", type=Path, help="待修复的 pkl 或目录")
    ap.add_argument("--export", type=Path, metavar="NPZ",
                    help="把供体的网格导出为仓库内 npz 资产（一次性操作）")
    ap.add_argument("--check", type=Path, help="只检查不修改（pkl/npz 或目录）")
    ap.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    ap.add_argument("--force", action="store_true", help="即使已有网格也覆盖")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = ap.parse_args()

    if args.check:
        paths = _collect(args.check, args.recursive)
        if not paths:
            print(f"未找到 pkl：{args.check}")
            return 2
        print(f"检查 {len(paths)} 个文件：")
        bad = report(paths)
        print(f"\n{len(paths) - bad} 个正常，{bad} 个需要修复。")
        print("说明：状态 MISSING 或 FALLBACK? 的文件在渲染时会画出占位立方体，"
              "而非真实物体——这不会报错，必须靠本检查发现。")
        return 1 if bad else 0

    if args.export:
        if not args.donor:
            ap.error("--export 需要配合 --donor 指定来源")
        info = inspect(args.donor)
        if info["status"] != "OK":
            print(f"供体网格不可用（状态 {info['status']}），拒绝导出。")
            report([args.donor])
            return 1
        out = export_npz(_load(args.donor)["object_models"], args.export)
        size_kb = out.stat().st_size / 1024
        print(f"已导出网格资产：{out}  ({size_kb:.1f} KB)")
        for k, v in info["meshes"].items():
            print(f"  {k}: {v[0]} 顶点 / {v[1]} 面片")
        print("\n该文件可直接入库，之后用它作为 --donor，流水线不再依赖外部 pkl。")
        return 0

    if not args.donor or not args.target:
        ap.error("需要同时提供 --donor 与 --target，或使用 --check / --export")

    print("供体：")
    report([args.donor])

    paths = _collect(args.target, args.recursive)
    if not paths:
        print(f"未找到目标 pkl：{args.target}")
        return 2
    print(f"\n处理 {len(paths)} 个目标：")
    changed = 0
    for p in paths:
        try:
            if inject(args.donor, p, backup=not args.no_backup, force=args.force):
                changed += 1
        except (ValueError, KeyError) as e:
            print(f"  [失败] {p.name}: {e}")

    print(f"\n完成：{changed} 个已注入。")
    if changed:
        print("建议复核：python scripts/fix_tray_mesh.py --check <目录> --recursive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
