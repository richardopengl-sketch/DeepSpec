# DSpark 代码阅读：从缓存到训练，再到验证

本文整理公开 DSpark 源码的阅读顺序、训练与验证链路，以及与原生 MTP 的区别。
这是个人 fork 的学习文档，不包含其他项目的数据、checkpoint 或协作状态。

- 源码基准：`deepseek-ai/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`；本机 checkout 已核对。
- 论文：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)。
- 性质：已有研究的阅读导航，不是运行教程或执行授权。只做静态阅读和链接检查，没有安装、加载权重、推理或训练。
- 适用范围：固定版本的公开源码。Qwen3、Qwen3.5 与 Qwen3.8 Flash Next 的架构和缓存契约不同，不能仅替换模型名称就视为完成适配。
- 发布状态：2026-11-12 整理文档；没有因此启动新的训练、推理、测试或测速。

## 1. 路径规则：从哪里点开才正确

本文放在 **DeepSpec 仓库根目录**，与 [README.md](./README.md)、[train.py](./train.py) 同级。所有本地相对链接都以本文所在目录为基准，不以终端当前目录或 ms-swift 工作区为基准。

- 本仓库源码用 `./deepspec/...`、`./config/...`、`./scripts/...`。
- 例如这里的数据说明链接是 [scripts/data/README.md](./scripts/data/README.md)，不是 `../scripts/data/README.md`。
- 不把聊天中的 `C:/.../file.py:行号` 直接当作 Markdown 文件路径，也不把 `#L25` 当成本地 Python 文件的通用跳转约定。下面本地链接打开文件，关键符号单独标明；固定版本的 GitHub 链接用于独立阅读。
- 本文不依赖相邻的本地研究目录，不复制其他项目的数据或冻结源码。
- 若单独分享本文，优先使用每行的“固定源码”链接；本地源码链接需要保留完整 DeepSpec 目录布局。

## 2. 推荐阅读顺序

| 顺序 | 本地入口 | 主要看什么 | 固定源码 |
|---|---|---|---|
| 1 | [项目说明](./README.md) | 区分数据准备、训练和评测三段；默认例子不是 Flash Next。 | [README](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md) |
| 2 | [数据准备说明](./scripts/data/README.md) | prompts、target 重新生成的回答、target cache 是不同产物；不要自动执行其中命令。 | [数据说明](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/README.md) |
| 3 | [parser.py](./deepspec/data/parser.py) | `conversations` 如何进入 tokenizer/template、截断和监督 mask。 | [parser](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/parser.py) |
| 4 | [target_cache_dataset.py](./deepspec/data/target_cache_dataset.py) | manifest、二进制索引、字段长度、`CacheDataset` / `CacheCollator`。不要把 embeddings 当作完整缓存契约。 | [缓存协议与读取](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/target_cache_dataset.py) |
| 5 | [DSpark Qwen3-4B 配置](./config/dspark/dspark_qwen3_4b.py) | `trainer_cls`、target 层选择、block size、loss 系数；这是源码学习样例，不是正式目标配置。 | [配置](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py) |
| 6 | [train.sh](./scripts/train/train.sh) → [train.py](./train.py) | shell 传入配置与缓存路径；Python 按可见 GPU 数 spawn，再调用配置指定的 trainer。不是标准 torchrun 入口。 | [launcher](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/train/train.sh) / [Python 入口](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/train.py) |
| 7 | [base_trainer.py](./deepspec/trainer/base_trainer.py) | `build_models`、数据加载、优化与保存。缓存存在不等于启动训练时无需 target 权重。 | [BaseTrainer](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/base_trainer.py) |
| 8 | [dspark_trainer.py](./deepspec/trainer/dspark_trainer.py) | `Qwen3DSparkTrainer.run_batch` 将哪些 batch 字段传入模型，再怎样计算 loss。 | [DSpark trainer](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/dspark_trainer.py) |
| 9 | [Qwen3 modeling.py](./deepspec/modeling/dspark/qwen3/modeling.py)、[common.py](./deepspec/modeling/dspark/common.py)、[markov_head.py](./deepspec/modeling/dspark/markov_head.py) | 多层 target 特征、并行 draft backbone、anchor/mask、串行 head 与输出对齐。 | [模型](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py) / [公共逻辑](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py) / [Markov head](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py) |
| 10 | [loss.py](./deepspec/modeling/dspark/loss.py) | `_build_loss_weight_mask`、`_collect_local_terms`、`compute_dspark_loss`；区分 CE、概率 L1/TV、confidence 监督及归约。 | [loss](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py) |
| 11 | [eval.py](./eval.py)、[DSpark evaluator](./deepspec/eval/dspark/evaluator.py)、[draft_ops.py](./deepspec/eval/dspark/draft_ops.py) | proposal 怎样产生、何时截断、怎样交给 verifier；不要把训练里的代理指标等同于实际延迟。 | [评测入口](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py) / [evaluator](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/evaluator.py) / [proposal](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py) |

## 3. 把训练链路连起来

```text
target 生成的回答
    -> parser/tokenizer/template
    -> target cache
    -> CacheDataset / CacheCollator
    -> Qwen3DSparkTrainer.run_batch
    -> Qwen3DSparkModel
    -> compute_dspark_loss
```

在已检查的公开实现中：

- 缓存协议版本为 2。记录包含 `input_ids`、`attention_mask`、`loss_mask`、`target_hidden_states`、`target_last_hidden_states`；索引还记录样本、序列长度及各字段 offset。缓存协议规定 hidden 为 BF16、token 为 int32、mask 为 uint8，不能据此推断其他实现的文件格式。
- `run_batch` 明确传入 `input_ids`、`target_hidden_states`、`loss_mask`、`target_last_hidden_states`。不要把“缓存存了 attention mask”误写成“这个 trainer 原样向模型传了 attention mask”。
- 示例配置选取多个 target 层。仅有一个普通 embedding 矩阵，不等于满足此输入契约。
- `BaseTrainer.build_models` 仍调用 `AutoModelForCausalLM.from_pretrained`，将 target 放到 CPU，取得 embedding/head 后初始化并冻结 draft 中对应部分，再释放 target。因此 **有缓存也不能推断无需权重或 CPU 内存**。
- loss 的远 horizon 权重来自 `exp(-position / loss_decay_gamma)`，叠加实际有效 mask。CE 用 token labels；概率 L1/TV 和论文式 confidence 另依赖对齐的 target 分布。不要把三者的输入需求混为一谈。

这些是源码事实，不是本机训练成功的证据。本文没有执行上面的链路。

## 4. 与原生 MTP 的区别

| 已有研究结论 | 与原生 MTP-only 的关系 |
|---|---|
| DSpark 的多层特征投影、并行 backbone、Markov/RNN head | 是新 drafter，不是原生 MTP 换一个 loss；不能宣称现有 Flash Next 入口可直接运行它。 |
| horizon 加权 CE | 可作为原生 MTP 的独立消融假设；不是已验证的 NES 收益。 |
| 概率 L1/TV | 可能保留原生 MTP 结构，但增加 target 分布或精确重建该分布所需的状态/head。 |
| confidence head | 新增可训练模块；如果用于动态决定验证长度，还需要 serving 接入。 |
| 跨请求验证长度调度、异步生产调度 | 属于 serving 工程，不能由离线 embeddings 替代。已追踪的公开 proposal 是 batch size 1 路径，不等同论文完整生产调度。 |

不同模型的原生 MTP 对输入层位、归一化、状态结构和展开方式有各自要求。
公开 DSpark 的实现不能替代对实际 target、tokenizer、embedding、head 和 checkpoint 身份的核对。

## 5. 原生 MTP 的拟议最小对照

以下是实验设计示例，不是本仓库新增的可执行实现或已测结论。

四组要明确区分：

1. **MTP off**：同一固定、任务适配过的 target，关闭 MTP。
2. **Unchanged vanilla MTP**：指定起始 checkpoint 的 MTP 权重，不做本轮微调；不据此推断历史训练 loss。
3. **均匀 CE 微调**：相同起点，显式设定均匀 horizon 权重。
4. **衰减 CE 微调**：其余条件与均匀组相同，只改变 horizon 相对权重。

均匀和衰减两组固定数据成员与顺序、tokenization、有效 mask、状态边界、参数分区、unroll、优化器、学习率、种子和训练预算。至少两个 horizon 实际存在有效监督；仅配置 unroll >= 2 不够。

拟议的共同 masked reduction 是：

```text
L = lambda * sum(w[d] * m[d,t] * CE[d,t]) / sum(w[d] * m[d,t])
```

这里的 `m` 是 shift、边界、ignore 处理后的实际有效 mask。分子和分母必须在一致的分布式归约/梯度累积范围内计算；零有效监督必须显式停止或报告，不能伪造分母。该定义不声称等同现有 bridge 默认 reduction，也不声称已经实现。

纯 CE 不要求完整 target 分布；TV/confidence 是后续可选路线。固定 K 的比较只适用于 MTP 组，MTP off 没有 draft K。指标包括 draft-token acceptance、每次 target step 推进量、完整端到端延迟和吞吐；不是用户接受建议的比例。样本量不足时不宣称可靠 P95。

## 6. 接入其他模型前的检查边界

当前只读检查应分别核对：

- **身份**：target checkpoint/revision、task adaptation、tokenizer/template；正式目标不能默默更换。
- **状态**：embeddings 实际含义、提取层、norm 前后、shape/dtype、序列覆盖、与 token 的 shift 约定。
- **对齐**：token IDs、position、attention/loss mask、packing/文档边界、EOS 和尾部监督。
- **入口与分集**：配套 loader/trainer 实际读取什么；train/test 文件和成员分别核对，不合并、不把 test 用于训练或校准。

结果只列“可直接复用、存在差异、最少缺项”。现有 validation-only 记录不能自动升级为未触碰的最终独立评测集。

本说明不授权下载大批缓存、安装、加载模型、forward/backward 或训练；字段检查通过也不等于真实模型链路通过。
