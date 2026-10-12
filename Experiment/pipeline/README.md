# pipeline 目录说明

存放 HUMOTO 原始数据经过官方三步转换流程后，每一阶段的产物，按**阶段 → 序列名**两级组织。转换流程与命令细节见 `Solver_Prework.md` 第0节，脚本本身在 `humoto/scripts/`。

## 结构与各阶段含义

```
pipeline/
├── raw/<sequence_name>/
│   ├── <sequence_name>.fbx    原始下载的 HUMOTO 动作数据（未处理）
│   └── <sequence_name>.yaml   序列元数据（任务描述等）
├── scale_cleared/<sequence_name>/
│   └── <sequence_name>.fbx    第1步产物：烘焙根骨骼缩放后的 FBX（clear_human_scale.py）
└── up_bone/<sequence_name>/
    ├── <sequence_name>.fbx    第2步产物：重定向到 up_bone 骨架后的 FBX（transfer_human_model.py）
    └── <sequence_name>.pkl    第3步产物（★项目实际使用的最终数据源）：提取出的骨骼动作pkl（extract_pk_data.py）
```

## 当前已处理的序列

- `carry_organizer_with_both_hands_at_chest_height-436`：唯一的参考动作 A_ref（端空盘子行走），三阶段均已跑通。

## 新增序列时的操作顺序

1. 把原始 `.fbx`（+可选 `.yaml`）放入 `raw/<new_sequence_name>/`
2. 激活 `Experiment/venv`，依次运行 `clear_human_scale.py` → `transfer_human_model.py` → `extract_pk_data.py`（用法见 `humoto/README.md`），产物分别写入 `scale_cleared/` 与 `up_bone/`
3. 下游代码只需读取 `up_bone/<sequence_name>/<sequence_name>.pkl`
