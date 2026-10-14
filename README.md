# LIS-VLA: Latent Imagination Search VLA

> 在 [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002) 之上，把 “感知 → 动作” 的单向流程改造成 **隐空间内的闭环搜索**：一次 backbone 前向，生成 N 个候选动作块，在隐空间里“想象”各自的未来，用语言指令打分，选最优的执行。

![LIS-VLA architecture](rynnvla-002/assets/lis_vla_architecture.png)

## 核心思想

| 模块 | 图中位置 | 实现（`rynnvla-002/model/lis/modules.py`） |
|---|---|---|
| Encoder → $z_t$ | Input Layer | Chameleon backbone（RynnVLA-002）。$z_t$ = `ObservationEncoder`：对当前帧各视角 **VQ token 的输入嵌入** 做 Perceiver 式注意力池化，得到 Q 个 latent tokens（默认 16×512）。它只依赖观测本身，因此同一个编码器也能给出未来帧的目标 $z_{t+k}$。 |
| Action Proposer | System 2 | `FlowMatchingProposer`：rectified-flow 头，条件为 RynnVLA 动作头的 action-query 特征 + backbone 上下文 tokens $c_t$；Euler 10 步并行采样 N 个动作块 $A^1..A^N$。候选 #0 固定为 System 1（原 L1 动作头）的确定性输出，保证搜索结果不劣于“不搜索”的打分。 |
| Latent World Model | System 2 | `LatentWorldModel`：$\hat z_{t+k} = z_t + D([z_t; c_t; A])$，Transformer 动力学头；输出层零初始化（未训练时等价于“世界不变”）。不生成像素，k = H（动作块长度）。 |
| Language-Guided Evaluator | System 2 | `LanguageGuidedEvaluator`：$R = \cos(\phi(\hat z_{t+k}), \psi(g))$，$g$ 为 backbone 对指令 token 的隐状态池化。 |
| Selection | Decision | $A^* = \arg\max_i \; R^i - \text{Cost}(A^i)$，可选 CEM 精炼若干轮；执行整块（与 RynnVLA-002 一致的 action chunking）。 |

### 搜索目标

$$
A_t^* = \arg\max_{A \in \{A^i\}_{i=1}^N} \; \mathrm{Sim}\big(\phi(\hat z_{t+k}),\, \psi(g)\big) - \mathrm{Cost}(A), \qquad \hat z_{t+k} = \mathcal{WM}(z_t, c_t, A)
$$

$$
\mathrm{Cost}(A) = \lambda_s \frac{1}{H-1}\sum_{i}\lVert a_{i+1}^{\text{arm}} - a_i^{\text{arm}}\rVert^2 + \lambda_p \frac{1}{H}\lVert A - A^{\text{S1}}\rVert^2
$$

说明：当前世界模型是确定性的，因此原式中的期望 $\mathbb{E}_{\hat z\sim\mathcal{WM}}$ 退化为单次预测；平滑项不作用于夹爪维度（夹爪本来就需要跳变）。$A^{\text{S1}}$ 是 System 1 的输出，$\lambda_p$ 默认 0。

### 训练目标（全部在 latent 空间）

LIS 训练样本把“执行动作块后到达的未来帧”拼到 VLA 序列后面，一条序列同时监督所有模块：

```
human: What action should the robot take to {task}? <|state|> <|image|>×(his·V)
gpt  : <|action|>×H  <|image|>×V        ← t+H 时刻的第三视角 + 腕部视角
```

| 损失 | 作用 | 默认权重 |
|---|---|---|
| `lis_fm` | flow-matching：$\lVert v_\theta(x_\tau,\tau)-(a-x_0)\rVert^2$，τ 取 logit-normal | 1.0 |
| `lis_wm` | $1-\cos(\hat z_{t+H}, \mathrm{sg}[z_{t+H}])$，逐 latent token（JEPA 式 stop-grad 目标） | 1.0 |
| `lis_wm_nce` | batch 内 InfoNCE：预测必须能认出自己的真实未来（防表征坍塌） | 0.1 |
| `lis_var` | VICReg 方差项（防坍塌） | 0.1 |
| `lis_lang_nce` | 多正例 InfoNCE：真实/想象的未来 latent ↔ 指令嵌入（同指令样本互为正例） | 0.1 |
| `lis_prog` | 进度排序：$R(z_{t+H}, g) > R(z_t, g)$ | 0.05 |
| `lis_rank` | 想象中的动作排序：专家动作块的想象未来 > 反事实动作块（加噪 / 时间反转 / batch 内错配）的想象未来 | 0.1 |

所有损失通过模型返回的 `additional_loss_dict` 进入原版 trainer，**trainer 本身未修改**，日志里会直接看到上述各项。原版的 next-token CE（含 t+H 帧图像生成）和连续动作 L1 损失保持不变。

## 相比 RynnVLA-002 的改动

新增文件：

| 文件 | 内容 |
|---|---|
| `rynnvla-002/model/lis/tokens.py` | 解析 RynnVLA 的 token 序列（指令 / 状态 / 当前帧 / 动作块 / 未来帧的位置） |
| `rynnvla-002/model/lis/modules.py` | Observation/Context/Language 编码器、Proposer、Latent World Model、Evaluator、损失、搜索 |
| `rynnvla-002/model/modeling_lis_vla.py` | `ChameleonXLLMXForConditionalGeneration_LIS`、`LISVLAConfig`、`load_lis_model` |
| `rynnvla-002/pretrain_solver_lis_vla.py` | 训练入口（继承原 solver，新增全部 `--lis_*` 参数） |
| `rynnvla-002/eval_solver_libero_lis.py`, `libero_util/lis_utils.py` | LIBERO 评测（搜索 + 每步搜索日志 `lis_search_log.jsonl`） |
| `rynnvla-002/data/lis_conv_generation.py`, `data/pretoken_lis_model.py` | LIS 训练数据生成与预分词 |
| `rynnvla-002/exps_lis/*.sh`, `configs/lis/*.yaml` | 数据准备 / 训练 / 评测脚本 |
| `rynnvla-002/tests/test_lis_vla.py` | CPU 冒烟测试（tiny Chameleon） |

对原有行为的两处修正（均在子类中实现，原文件未改）：

1. **Attention mask**：原 `generate_att_mask_3` 以“最后一张图”为参照决定动作块之间是否互相屏蔽；LIS 样本在动作块之后还有未来帧，会导致训练时动作块互相可见、与推理不一致。LIS 版本以“第一个动作块之前的最后一张图”为参照——对原有的 VLA 样本、世界模型样本、推理 prompt **结果完全相同**（有测试验证），只在 LIS 样本上生效。
2. **Action head padding**：原 `ActionHead` 在 batch 内上下文长度不同时会让 action query 注意到零填充位置；LIS 版本对填充位置做了 mask。参数名与形状完全一致，**原版 checkpoint 可直接加载**，且 batch=1 推理时与原版数值一致（有测试验证）。

## 使用

环境与 RynnVLA-002 相同（`requirements.txt`，torch 2.2.0 / transformers 4.43.0），先按 [README_RynnVLA-002.md](README_RynnVLA-002.md) 完成 LIBERO 数据的 Step 1（重新生成图像/动作/状态），并下载 tokenizer 与 RynnVLA-002 checkpoint 到 `rynnvla-002/ckpts/`。

### 1. 准备 LIS 数据

```bash
cd rynnvla-002/exps_lis
bash prepare_data.sh goal 5        # goal/object: H=5；spatial/10: H=10
```

会生成 LIS 样本并预分词；若原版世界模型数据和 VLA 数据的 token 已存在，会一并混入 `processed_data/concate_tokens/libero_goal_..._lis_awm.json`。

### 2. 训练

```bash
# 冻结 RynnVLA-002，只训练 System 2（最省算力，推荐先跑）
bash train_lis.sh goal 5 lis_only ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
# 同时微调连续动作头
bash train_lis.sh goal 5 lis_and_head ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
# 全量微调（与 RynnVLA-002 训练设置一致）
bash train_lis.sh goal 5 all ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
```

从不含 LIS 权重的 checkpoint 启动时，`load_lis_model` 会自动初始化 `lis.*` 参数。`--lis_detach_backbone true`（默认）表示 LIS 损失不回传到 backbone。

### 3. 评测

```bash
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9            # N=16 搜索
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9 32 2        # N=32 + 2 轮 CEM
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9 16 0 0      # 关闭搜索 = System 1 消融
```

### Python 接口

```python
from model import load_lis_model
model = load_lis_model(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
chunk = model.generate_lis_action(input_ids, generation_config, num_candidates=16, cem_iters=0)  # (H, 7), 归一化空间
model.last_search_info   # best_score / best_semantic / best_cost / score_std / prior_score ...
```

## 规模与开销

- System 2 参数量：**约 50.1M**（7B 维度下：obs 9.0M / ctx 5.3M / lang 5.3M / proposer 13.7M / world model 12.9M / evaluator 3.9M），约为 backbone 的 0.7%。
- 搜索 **不需要额外的 backbone 前向**，也不解码像素：所有候选在 512 维 latent 空间里批量前向。
- 参考：在 2 线程 CPU、fp32 下单次规划 N=16 约 0.43 s，N=32 约 0.74 s，N=16+2 轮 CEM 约 0.63 s；GPU 上的延迟尚未实测。

## 测试

```bash
cd rynnvla-002
python -m pytest -q tests/test_lis_vla.py
```

覆盖：序列解析；attention mask 与原版一致性及 LIS 样本上的修正；训练前向/反向（所有 LIS 模块均有梯度）；60 步内 `lis_fm` / `lis_wm` / `lis_lang_nce` 下降；端到端 `generate_lis_action`（含 CEM 与关闭搜索）；动作头与原版 `ActionHead` 的 state-dict 兼容与数值一致；从不含 LIS 权重的 checkpoint 加载。

## 当前状态与建议的消融

代码已在 tiny 配置上端到端跑通，**尚未在 LIBERO 上用 7B 模型训练或评测，目前没有任何成功率结果**。建议的实验：

1. System 1（`SEARCH=0`）vs. LIS（同一 checkpoint），N ∈ {4, 8, 16, 32}；
2. Evaluator 去掉 `lis_rank` / `lis_prog` / `lis_lang_nce` 各一项；
3. 隐空间搜索 vs. 像素空间（用 RynnVLA-002 自带的世界模型生成图像再打分）的成功率与延迟对比；
4. `lis_only` vs. `all` 训练范围。

注意：Evaluator 只从专家演示中学习（专家动作块 vs. 反事实动作块），它学到的是“像专家那样推进任务”的偏好，在演示分布之外的可靠性需要实验验证；`last_search_info["score_std"]` 很小时说明候选之间区分度不足。

## 致谢与许可

基于 [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002)（Apache-2.0），原 README 保留为 [README_RynnVLA-002.md](README_RynnVLA-002.md)。本仓库同样以 Apache-2.0 发布，见 `LICENSE` 与 `NOTICE`。
