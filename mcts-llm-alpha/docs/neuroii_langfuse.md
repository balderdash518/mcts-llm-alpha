# NeuroII / Qwen / Langfuse 接入

默认配置与 `src/mcts_llm_alpha/config/neuroii_qwen.yaml` 使用：

- Endpoint：`https://neuroii.com/v1`
- Model：`Qwen/Qwen3.8-27B-FP8`
- 服务端上下文窗口：32768 tokens，输入、思考、最终答案共用。
- `chat_template_kwargs.enable_thinking: true`，流式输出，600 秒网络超时。
- 输出上限 32768，实际每次按 `min(上限, 窗口 - 估算输入 - 256)` 分配。没有设置固定的小型 thinking budget，也没有向该路由添加未经验证的 `reasoning_effort` 参数。
- 当前搜索顺序调用模型；同一时间最多一个请求。服务端显示的 3.79× 不作为整数并发数。

## 本地配置与运行

Python 3.11+ 可使用固定版本的 Langfuse SDK 5.0.4。当前仓库根目录 `.venv` 是独立的 Python 3.12 环境，已安装接口、追踪、Qlib 0.9.7 与离线测试依赖。

凭据写在项目子目录 `mcts-llm-alpha/.env`，已被 Git 忽略。需要 `NEUROII_API_KEY`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`。程序始终从脚本旁的 `.env` 加载，系统环境变量优先。自定义地址不能隐式使用 `OPENAI_API_KEY`；通过 `llm.api_key_env` 明确选择独立凭据。

从仓库根目录运行接口检查（真实 API 请求，不需要市场数据）：

```powershell
.\.venv\Scripts\python.exe .\mcts-llm-alpha\check_connection.py
```

完整搜索还需要 Microsoft Qlib（发行包 `pyqlib`）、匹配的依赖与真实行情数据，然后设置 `QLIB_PROVIDER_URI`：

```powershell
.\.venv\Scripts\python.exe .\mcts-llm-alpha\run_search.py --iterations 10
```

本文使用已接入新客户端的 `run_search.py`；历史 `mcts-llm-alpha` CLI 仍依赖旧接口，尚未迁移。配置文件也可通过 `--config` 指定。`llm.langfuse` 支持 `on/auto/off`：on 缺失配置时明确失败；auto 在没有凭据时关闭追踪；off 完全不启用。

## 上下文处理

不再提前按 650 字、5 条或固定的 12000 字切断上下文。当前节点、父节点、兄弟、子节点、路径及描述完整渲染；历史实验检索数量可配置（默认 30）。信号矩阵不进入提示词。

估算提示词足够小时不压缩任何记录；长提示词会先减少输出预算。只有连最少 4096 tokens 输出预留也无法保留时，才从关系/实验列表中移除整条记录，并标出省略数量，保留当前候选、错误、父节点和评估协议。这里的 4096 是触发精简的最低预留，不是固定输出上限；可调整 `min_output_tokens`。字符估算不是服务端 tokenizer 的精确计数，中文按较高开销估算。强制上下文仍超窗时明确报错，不切坏公式。

200-token 审阅限制已经移除，初始画像、精炼、公式转换、审阅和总结共用动态调用通道。服务端返回的思考片段与最终答案分离，仅最终答案进入 JSON/公式解析。长预算只允许模型多思考，不会强制它耗尽窗口，也不保证更好的因子。

## 失败与追踪

连接、超时、429、5xx 会退避重试；鉴权/其他请求错误不盲目重试。中断的流丢弃后整体重试；`finish_reason=length`、空答案、未完成输出明确失败，不能作为完整候选返回。SDK 重试关闭，避免与应用重试叠加。

使用 Langfuse 官方 OpenAI SDK 集成捕获模型调用、输入、原始输出和 token usage。搜索初始化和每次 MCTS 迭代分别是 agent trace，同次运行共享 session；其中嵌套 evidence retriever、参数试评 evaluator、候选 evaluator 和具名 generation。调用链同时记录估算预算、精简数量、重试信息；迭代记录选中公式、候选评分、奖励及最佳结果。评估输出不包含整个信号矩阵。

结束时 flush。追踪更新/导出故障不重复触发付费模型调用。配置/密钥不进入观测 payload。真实服务端返回的 usage 是准数；预算估算不代替 usage。

## 验证范围

离线回归涵盖完整历史、超窗精简、动态预算、中文估算、stream usage 尾块、思考分离、中断恢复、输出截断拒绝、鉴权错误和独立密钥解析。真实连通检查覆盖 NeuroII 与 Langfuse；这不等于运行了行情回测。

参考：[Langfuse OpenAI 集成](https://langfuse.com/integrations/model-providers/openai-py)、[追踪结构规范](https://langfuse.com/docs/observability/best-practices)、[Qwen vLLM 接口](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)。

## 完整真实数据实验（2026-10-24）

本机可用行情截至 2020-09-25，所以 `configs/live_neuroii.yaml` 明确使用 2018-01-01 至 2020-09-24，分割日 2019-07-01。该实验是初始生成加 3 次 MCTS 精炼；预算增量为零，确保按指定规模结束。

```powershell
.\.venv\Scripts\python.exe .\mcts-llm-alpha\run_experiment.py --config .\mcts-llm-alpha\configs\live_neuroii.yaml
```

新入口在数据初始化之前建立 `run-alpha-experiment` 根 observation，下面嵌套数据初始化、模型生成、参数试评、证据检索、迭代及入库决策。整个实验共用一条 trace 和一个 session。失败时也保存 `run_manifest.json` 并 flush。数据实际提供的字段作为模型调用约束，避免引用缺失的 `$vwap`。

每次结果写到独立目录：`run.log`、`run_manifest.json`、`experiments.jsonl`、`search_tree.json`、`alpha_repository.csv`。完成后可从服务端回读并核验轨迹：

```powershell
.\.venv\Scripts\python.exe .\mcts-llm-alpha\audit_experiment_trace.py <结果目录>
```

核验脚本使用官方 `@langfuse/cli` 1.2.4 与 observations v2 API，检查 session、完整根节点、MCTS 迭代、入库决策、每次模型调用的输入/输出及 token usage。直接在当前密钥对应的项目查看，避免打开历史测试项目的链接。
