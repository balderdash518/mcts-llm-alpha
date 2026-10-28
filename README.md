# MCTS-LLM Alpha Mining Framework

当前默认接入 NeuroII 的 Qwen 思考模型与 Langfuse，运行方式及预算说明见 [接入文档](mcts-llm-alpha/docs/neuroii_langfuse.md)。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An advanced quantitative alpha factor mining framework that combines Monte Carlo Tree Search (MCTS) with Large Language Models (LLMs) to discover formulaic alpha factors for stock prediction.

## 🌟 Overview

This project implements the cutting-edge algorithm described in the paper "Formulaic Alpha Mining using Monte Carlo Tree Search with LLM". It uses GPT-4 to generate and refine mathematical formulas that predict stock returns, evaluating them across multiple quality dimensions:

- **Effectiveness**: Signal strength and prediction accuracy
- **Stability**: Robustness across market conditions
- **Turnover**: Trading frequency (lower is better)
- **Diversity**: Uniqueness from existing factors
- **Overfitting**: Generalization capability

## 🚀 Key Features

- **Two-Step Generation Process**: High-level alpha portrait → Symbolic formula with parameters
- **Multi-dimensional Evaluation**: Comprehensive assessment across 5 quality dimensions
- **Virtual Action Mechanism**: Enables expansion of internal nodes in MCTS
- **Relative Ranking System**: Dynamic scoring based on repository performance
- **Few-shot Learning**: Context-aware formula refinement using successful examples
- **Real Market Data**: Integration with Qlib for CSI300 universe evaluation

## 📋 Requirements

- Python 3.8+
- OpenAI API key (GPT-4 access)
- Qlib with CSI300 market data
- CUDA-capable GPU (optional, for faster Qlib computations)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/dtbtc/mcts-llm-alpha.git
cd mcts-llm-alpha
```

2. **Set up Python environment**
```bash
conda create -n alphagen_dev python=3.8
conda activate alphagen_dev
```

3. **Install the package**
```bash
cd mcts-llm-alpha
pip install -e ".[dev]"
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY and QLIB_PROVIDER_URI
```

5. **Download Qlib data (CSI300)**
```bash
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data/cn_data --region cn
```

## 🎯 Quick Start

### Basic MCTS Search
```bash
# Run with default settings (50 iterations)
mcts-llm-alpha search --iterations 50

# Run with custom configuration
mcts-llm-alpha search --config config/experiment.yaml

# Resume from checkpoint
mcts-llm-alpha resume --checkpoint checkpoints/mcts_checkpoint_20240101_120000.pkl
```

### Direct Python Script
```bash
cd mcts-llm-alpha
python run_search.py --iterations 10
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

## 📊 Example Output

```
[5] 开始MCTS搜索...
============================================================

训练过程说明:
  1. LLM生成初始Alpha画像（描述）
  2. 将画像转换为符号公式
  3. 评估多组参数，选择最优
  4. 开始MCTS树搜索:
     - 选择：使用UCT算法选择节点
     - 扩展：LLM针对特定维度优化公式
     - 评估：计算5个维度的得分
     - 回传：更新树节点统计

============================================================

🎯 最佳发现公式:
公式: Rank((Mean($volume, 5) / Mean($volume, 20)) * Corr($close, $volume, 15), 10)
最佳分数: 6.842

📚 Alpha仓库统计:
入库因子数: 12
仓库平均分: 5.73
仓库最高分: 6.84
```

## 🏗️ Architecture

```
mcts-llm-alpha/
├── src/mcts_llm_alpha/
│   ├── config/          # Configuration management
│   ├── llm/            # LLM integration (OpenAI, prompts)
│   ├── mcts/           # Core MCTS algorithm
│   ├── formula/        # Formula processing pipeline
│   ├── evaluation/     # Multi-dimensional evaluation
│   ├── data/           # Qlib data provider wrapper
│   └── cli.py          # Command-line interface
├── tests/              # Comprehensive test suite
├── notebooks/          # Example notebooks
└── docs/              # Documentation
```

## 📈 Performance & Results

The system has been tested on CSI300 universe with the following improvements:
- Fixed relative ranking mechanism to correctly implement paper's formula
- Enhanced formula diversity through structural changes (not just parameters)
- Improved cold-start handling with progressive threshold adjustment
- Optimized caching for 10x faster repeated evaluations

## 🔧 Configuration

Key parameters in `config/default.yaml`:
- `max_iterations`: Total MCTS search iterations (default: 50)
- `exploration_constant`: UCT exploration vs exploitation (default: 1.0)
- `dimension_temperature`: Softmax temperature for dimension selection (default: 10.0)
- `effectiveness_threshold`: Minimum effectiveness score (default: 3.0)
- `diversity_threshold`: Minimum diversity score (default: 2.0)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest -v --cov=mcts_llm_alpha --cov-report=html

# Run specific test
pytest tests/test_mcts_core.py::test_node_selection
```

## 📚 Documentation

- [Architecture Overview](mcts-llm-alpha/docs/architecture.md)
- [MCTS Algorithm Details](Artical/MTCS_LLM_Alpha.md)
- [中文文档](Artical/alpha_mtcs_cn.md)
- [Fix History](mcts-llm-alpha/docs/MCTS_LLMALPHA_FIX_PLAN.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the paper "Formulaic Alpha Mining using Monte Carlo Tree Search with LLM"
- Uses [Qlib](https://github.com/microsoft/qlib) for market data and backtesting
- Powered by OpenAI's GPT-4 for formula generation

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system requires real market data and OpenAI API access. Simulation mode has been removed to ensure authentic alpha factor discovery.