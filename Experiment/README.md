# Experiment 目录说明

本目录存放本项目**自己生成/持有**的实验数据与运行环境，与 `humoto/`（第三方官方工具代码，克隆自 HUMOTO 仓库，不要在此目录下修改其内部结构）区分开。

## 结构

```
Experiment/
├── pipeline/   数据转换流水线的各阶段产物（详见 pipeline/README.md）
└── venv/       Python 3.11 + bpy==4.3.0 虚拟环境，专用于运行 humoto/scripts/ 的官方数据转换脚本
```

## venv 说明

- **用途**：仅用于运行 `humoto/scripts/` 下的三步官方转换脚本（`clear_human_scale.py` → `transfer_human_model.py` → `extract_pk_data.py`），把原始 HUMOTO 数据转换为项目使用的 up_bone 骨架 + pkl 格式。
- **当前状态**：`carry_organizer_with_both_hands_at_chest_height-436` 这条序列的转换已经完成，最终产物是 `pipeline/up_bone/.../*.pkl`，后续求解器开发**不需要**再次激活这个环境。
- **体积较大（约800MB）**：如果后续不需要转换新的动作序列（如切换到不同的参考动作），可以安全删除此环境以节省空间；未来若需要处理新序列，可参照 `humoto/README.md` 的 Installation 步骤重新创建（`conda create -n humoto python=3.10` 或用 venv + 对应依赖版本）。
