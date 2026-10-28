# MCTS-LLM Alpha挖掘框架

当前默认接入 NeuroII 的 Qwen 思考模型与 Langfuse，运行方式及预算说明见 [接入文档](docs/neuroii_langfuse.md)。

基于LLM的蒙特卡洛树搜索框架，用于量化金融中的公式化Alpha因子挖掘。

## 概述

本项目实现了一个先进的Alpha因子发现系统，结合了：
- 蒙特卡洛树搜索（MCTS）进行系统化探索
- 大语言模型（GPT-4）进行创造性公式生成
- 跨有效性、稳定性、换手率、多样性和过拟合的多维度评估
- 频繁子树分析（FSA）进行模式规避

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd mcts-llm-alpha

# 开发模式安装
pip install -e ".[dev]"

# 复制环境变量
cp .env.example .env
# 编辑.env添加您的OPENAI_API_KEY和QLIB_PROVIDER_URI
```

## 快速开始

### 1. 安装Qlib（必需）

系统需要Qlib来访问真实市场数据：

```bash
# 安装Qlib
pip install qlib

# 下载CSI300数据（中国A股）
python -m qlib.run.get_data qlib_data --target_dir G:/workspace/qlib_bin/qlib_bin --region cn
```

⚠️ **重要**：如果没有正确安装Qlib，系统将使用模拟数据，导致评估结果不准确。详细设置说明请参考 [QLIB_SETUP.md](QLIB_SETUP.md)。

### 2. 配置环境变量

编辑`.env`文件：

```bash
OPENAI_API_KEY=your_api_key_here
QLIB_PROVIDER_URI=G:/workspace/qlib_bin/qlib_bin
```

### 3. 运行系统

```bash
# 论文设置（需要 2011-01-01 至 2024-11-30 的 Qlib 数据）
python run_experiment.py --search-count 1000

# 本地 cn_data（截至 2020-09）的冒烟配置
python run_experiment.py --config configs/live_neuroii.yaml

# 中断后恢复
python run_experiment.py --resume results/<run_id>/checkpoint.pkl --output-dir results/<run_id>
```

每次运行在 `results/<run_id>/` 下写出 `run_manifest.json`、`search_trees.json`、
`alpha_repository.csv`、`final_alphas_top{10,50,100}.csv`（含测试期指标）和 `checkpoint.pkl`；
实验账本默认追加到 `logs/mcts_experiments.jsonl`，作为跨运行的历史记忆。

## 实现要点

与论文的逐项对应、信息流设计和所有自行补充的解释见
[docs/search_feedback_and_experiments.md](docs/search_feedback_and_experiments.md)。

- 多棵小树：每棵树初始预算 B=3，出现新最高分时 +1；树结束后按附录 F.1 的绝对标准入库。
- 每次扩展：按弱项 softmax 选维度 → LLM 读结构化上下文（SearchBrief）生成改进建议 →
  按建议改写画像 → 生成符号公式和 3 组参数 → 校验/执行失败时带错误信息纠错 →
  LLM 过拟合审阅 → 改进总结。
- 算子集合采用论文附录 D，翻译为 Qlib 表达式执行；FSA 从有效因子库挖掘 top-3 闭合子结构。

## 项目结构

```
mcts-llm-alpha/
├── src/mcts_llm_alpha/
│   ├── config/         # 配置（默认值即论文设置）
│   ├── formula/        # 论文算子语言：解析、校验、Qlib 翻译
│   ├── evaluation/     # Qlib 行情、论文指标、有效因子库与入库标准
│   ├── llm/            # 传输、提示词、SearchBrief、LLM 角色
│   ├── mcts/           # 节点、多树搜索、FSA、实验账本
│   └── experiment.py   # 完整实验流程（训练期搜索 + 测试期评估）
├── configs/            # 本地数据配置
├── legacy/             # 已被取代的旧实现（仅供对照，可删除）
├── tests/              # 离线测试（脚本化 LLM + 合成数据）
└── docs/
```

## 开发

```bash
# 运行测试
pytest

# 格式化代码
black src tests
isort src tests

# 代码检查
ruff check src tests

# 类型检查
mypy src
```

## 许可证

[您的许可证]