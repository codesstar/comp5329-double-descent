# 项目计划书
# Positional Encoding and Length Generalization in Transformers

**课程：** COMP5329 / COMP4329 Deep Learning  
**截止：** 2026-05-24 23:59 (Sydney Time)  
**小组：** 3人  
**平台：** Mac mini M4 Pro 48GB · PyTorch MPS  
**计划书版本：** v1.0 · 2026-05-09

---

## 1. 研究概述

### 1.1 核心问题

> 当 Transformer 在固定长度 L_train=128 上训练，推理时遇到 L_test > L_train 的序列，四种主流位置编码（Sinusoidal、Learnable、RoPE、ALiBi）的长度外推能力有何本质差异？差异的根源能否通过注意力模式定量解释？

### 1.2 研究动机

大语言模型在实际部署中常常需要处理比训练时更长的序列。位置编码是影响这一能力的核心组件，但现有研究缺乏在相同受控条件下对四种主流方案的系统性机制分析——大多数对比只报告性能数字，不解释"为什么"。本研究填补这一空白。

### 1.3 四个核心假设

| 编号 | 假设 | 验证方式 |
|------|------|----------|
| H1 | ALiBi 和 RoPE 的外推 perplexity 显著优于 Sinusoidal 和 Learnable | 实验1 泛化曲线 |
| H2 | Sinusoidal 超出训练长度后 attention entropy 崩溃 | 实验2 entropy 分析 |
| H3 | RoPE base frequency 越小，外推衰减越快 | 实验3 ablation |
| H4 | Learnable 在训练长度内最好，超出后立即失效 | 实验1 + 实验2 |

---

## 2. 资源需求

### 2.1 硬件资源

| 资源 | 配置 | 说明 |
|------|------|------|
| CPU | M4 Pro (12核) | 数据预处理 |
| GPU/MPS | M4 Pro GPU (20核) | 模型训练推理 |
| 内存 | 48GB Unified Memory | 可同时加载4个模型 |
| 磁盘 | 约 2GB | 数据集 + checkpoint + 结果 |

### 2.2 软件依赖

```
Python          3.11+
PyTorch         2.3+  (MPS backend)
transformers    4.40+ (tokenizer)
datasets        2.19+ (WikiText-2)
matplotlib      3.8+  (可视化)
seaborn         0.13+ (热力图)
numpy           1.26+
tqdm            4.66+
wandb           0.17+ (可选，实验追踪)
```

### 2.3 时间资源估算（Mac mini M4 Pro）

**单次训练（1个模型，WikiText-2，L=128，20 epochs）：**
- 每 epoch：约 90-120 秒
- 20 epochs：约 30-40 分钟
- 4个模型串行：约 2-2.5 小时

**所有实验总时间估算：**

| 实验 | 训练时间 | 评估时间 | 合计 |
|------|----------|----------|------|
| 实验1：4模型泛化曲线 | 2-2.5h | 30min | ~3h |
| 实验2：Attention 可视化 | 无（用实验1模型）| 20min | ~20min |
| 实验3：RoPE ablation（4个base）| 2h | 20min | ~2.5h |
| 实验4：训练长度敏感性（可选） | 2h | 20min | ~2.5h |
| **总计（不含可选）** | | | **~6h** |
| **总计（含可选）** | | | **~8.5h** |

> **注：** MPS 有时性能波动，建议预留 1.5x buffer，总计约 9-13 小时机器时间。

---

## 3. 代码架构

```
homework/
├── src/
│   ├── model.py              # Transformer 主体（不含位置编码）
│   ├── pe/
│   │   ├── __init__.py
│   │   ├── sinusoidal.py     # 固定三角函数编码
│   │   ├── learnable.py      # 可学习 embedding
│   │   ├── rope.py           # 旋转位置编码
│   │   └── alibi.py          # 线性偏置注意力
│   ├── data.py               # WikiText-2 加载 + tokenize
│   ├── train.py              # 训练主循环
│   ├── evaluate.py           # perplexity 评估
│   └── visualize.py          # attention entropy + 图表
├── experiments/
│   ├── exp1_generalization.py   # 实验1：泛化曲线
│   ├── exp2_attention.py        # 实验2：attention 分析
│   ├── exp3_rope_ablation.py    # 实验3：RoPE base ablation
│   ├── exp4_train_length.py     # 实验4：训练长度敏感性
│   └── run_all.sh               # 一键运行所有实验
├── results/
│   ├── checkpoints/          # 模型权重
│   ├── metrics/              # JSON 格式结果
│   └── figures/              # 生成的图表
├── paper/
│   └── (LaTeX 模板文件)
├── PROJECT_PLAN.md
└── README.md
```

---

## 4. 详细实验流程

---

### 阶段 0：环境搭建（预计 30 分钟）

**步骤：**
1. 创建 Python 虚拟环境
2. 安装所有依赖
3. 验证 MPS 可用：`torch.backends.mps.is_available()`
4. 设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`（防止 MPS 不支持的算子静默失败）
5. 下载 WikiText-2 数据集（约 12MB）

**验收标准：**
- `python -c "import torch; print(torch.backends.mps.is_available())"` 输出 `True`
- 数据集加载成功，能看到 train/valid/test split 大小

---

### 阶段 1：基础组件实现（预计 2-3 小时）

#### 1.1 数据处理（`src/data.py`）
- 加载 WikiText-2（HuggingFace datasets）
- 使用 GPT-2 tokenizer（BPE，词汇表 50257）
- 按固定长度截断/拼接成序列
- 输出：`DataLoader`，返回 `(input_ids, target_ids)` 对

**验收标准：**
- `batch = next(iter(train_loader))` 形状为 `(batch_size, seq_len)`
- 没有 padding token 泄漏到 target

#### 1.2 Transformer 主体（`src/model.py`）
- 6层，8头，d_model=512，FFN=2048
- 位置编码作为**可替换模块**注入（策略模式）
- ALiBi 特殊：不注入 embedding，而是修改 attention bias
- 使用手写 attention（不用 `F.scaled_dot_product_attention`，规避 MPS 兼容性问题）

**验收标准：**
- `model(x).shape == (batch, seq_len, vocab_size)` ✓
- 四种编码都能通过 forward pass，无报错

#### 1.3 四种位置编码

**Sinusoidal（`src/pe/sinusoidal.py`）**
- 公式：PE(pos, 2i) = sin(pos / 10000^(2i/d))
- 可生成任意长度（测试时直接扩展）
- **外推机制：** 测试时直接延长 PE 矩阵即可

**Learnable（`src/pe/learnable.py`）**
- `nn.Embedding(max_len, d_model)`，max_len=512
- 训练时只有前 128 个位置有梯度
- **外推机制：** 超出 max_len 的位置用最后一个 embedding（截断外推）

**RoPE（`src/pe/rope.py`）**
- θ_i = base^(-2i/d)，默认 base=10000
- 在 Q、K 上应用旋转矩阵，不改变 V
- **外推机制：** 旋转矩阵可直接外推到任意位置

**ALiBi（`src/pe/alibi.py`）**
- 不添加 position embedding
- 在 attention score 上加线性偏置：score += m_h * |i - j|
- m_h 按头序号以几何级数递增：m_h = -2^(-8h/n_heads)
- **外推机制：** 偏置天然支持任意位置

**验收标准（每种编码）：**
- 前向传播无报错
- 输出 embedding 形状正确
- 梯度可以正常反传

#### 1.4 训练循环（`src/train.py`）
- AdamW 优化器，lr=3e-4，weight_decay=0.01
- Cosine LR schedule with warmup（2000 steps）
- 每 epoch 记录 train loss
- 保存最低 valid loss 的 checkpoint
- 支持从 checkpoint 恢复

**验收标准：**
- 20 epoch 后 train loss 从 ~10 降到 ~3.5 左右（正常语言模型学习曲线）
- 四个模型的 valid perplexity 在训练长度内都收敛到合理值（< 100）

---

### 阶段 2：实验1 — 泛化曲线（预计 3-4 小时机器时间）

**目标：** 核心结果图。横轴测试序列长度，纵轴 perplexity，四条曲线。

**流程：**
1. 用 L_train=128 训练四个模型（各 20 epochs）
2. 分别在 L_test ∈ {64, 128, 192, 256, 384, 512} 上评估
3. 计算 perplexity = exp(cross-entropy loss)
4. 生成折线图，保存到 `results/figures/exp1_generalization.pdf`

**关键实现细节：**
- 评估时只改序列长度，不重新加载模型
- Learnable 在 L_test > 128 时用最后一个 position embedding 填充
- 每个测试长度跑完整 test set，取平均

**预期结果：**
```
L_test=128（训练内）：四种编码 perplexity 相近（80-120范围）
L_test=256（2x）：
  ALiBi:      ~120-140  (轻微上升)
  RoPE:       ~130-160  (轻微上升)
  Sinusoidal: ~200-400  (明显上升)
  Learnable:  ~400-800  (急剧上升)
L_test=512（4x）：
  ALiBi:      ~150-200
  RoPE:       ~180-250
  Sinusoidal: ~500+
  Learnable:  ~1000+
```

**验收标准：**
- 四条曲线数据完整（6个点）
- H1 和 H4 的方向性结论可以从图中明确读出
- 图表清晰，有误差范围（跑3次取平均）

---

### 阶段 3：实验2 — Attention Entropy 分析（预计 1 小时）

**目标：** 解释"为什么"某些编码外推失败，是论文 insight 的核心。

**流程：**
1. 加载实验1训练好的四个模型
2. 准备一批测试样本（L=512）
3. 提取每一层、每个头的 attention weight 矩阵
4. 计算 attention entropy：H(a) = -Σ_j a_ij * log(a_ij + ε)
5. 绘制：
   - **图A：** 各编码 × 各层的平均 entropy 热力图
   - **图B：** 对比 L=128 vs L=512 的 entropy 变化（Sinusoidal 应该崩溃）

**预期结果：**
- Sinusoidal/Learnable 在 L=512 时某些层 entropy 接近 0（注意力集中到少数位置，信息崩溃）
- ALiBi/RoPE 的 entropy 在 L=512 时仍保持相对均匀
- 这与 H2 一致，并能直观解释 perplexity 崩溃的原因

**验收标准：**
- 热力图生成，每个格子有数值
- 能清晰看出 Sinusoidal 的 entropy 在长序列时更低
- 图可以直接放入论文

---

### 阶段 4：实验3 — RoPE Base Frequency Ablation（预计 2-3 小时机器时间）

**目标：** 深挖机制，展示研究深度，直接对应 H3。

**流程：**
1. 训练4个 RoPE 模型，只改 base：{100, 1000, 10000, 100000}
2. 同样在 L_test ∈ {128, 256, 384, 512} 评估
3. 生成折线图：横轴测试长度，不同曲线对应不同 base
4. 讨论：为什么 LLaMA 选 10000？更大的 base 真的更好吗？

**预期结果：**
- base=100：外推快速崩溃（高频过于密集）
- base=10000：标准配置，中等外推
- base=100000：最佳外推（低频旋转更平滑）
- 存在 base 和训练内性能的 trade-off

**验收标准：**
- 四条曲线趋势清晰
- 能从结果推导出 base frequency 选择原则
- 结论与 H3 一致（或提供有趣的反例）

---

### 阶段 5：实验4 — 训练长度敏感性（可选，时间够才做）

**目标：** 补充实验，回答"外推比例"这个问题。

**流程：**
1. 固定 L_test=256
2. 改变 L_train ∈ {64, 128, 256}
3. 对 ALiBi 和 RoPE 分别跑
4. 看外推比例（2x vs 4x）对性能的影响

**验收标准：**
- 生成 2×2 的结果表格（2种编码 × 3种训练长度）

---

### 阶段 6：论文写作（预计 3-4 天）

**结构与字数规划（目标 7 页，双栏）：**

| 章节 | 目标字数 | 主要内容 |
|------|----------|----------|
| Abstract | 150词 | 问题、方法、主要发现 |
| 1. Introduction | 400词 | 动机、现有研究缺口、我们的贡献 |
| 2. Related Work | 500词 | 四种编码原始工作、已有外推研究 |
| 3. Methodology | 600词 | 模型架构、实验设置、评估指标 |
| 4. Experiments | 800词 | 四个实验结果 + 分析（含图表） |
| 5. Conclusion | 200词 | 总结、局限性、未来工作 |
| References | — | 目标 15-20 篇 |

**必须包含的图表（共 5 张）：**
1. 实验1：泛化曲线折线图（核心图）
2. 实验2A：Attention entropy 热力图
3. 实验2B：Entropy 随序列长度变化
4. 实验3：RoPE base ablation 曲线
5. 模型架构示意图（方法论部分）

**关键引文（必引）：**
- Vaswani et al. (2017) — Attention is All You Need
- Devlin et al. (2019) — BERT
- Su et al. (2021) — RoFormer (RoPE)
- Press et al. (2022) — ALiBi (Train Short, Test Long)
- Kazemnejad et al. (2023) — Impact of Positional Encoding on Length Generalization

---

## 5. 完整时间线

```
2026-05-09 (今天)
  ├── 确认计划书
  ├── 搭建环境
  └── 实现数据加载 + 模型骨架

2026-05-10
  ├── 实现四种位置编码
  ├── 验证前向传播
  └── 跑通单个模型训练

2026-05-11
  ├── 实验1：训练四个模型（~2.5h 机器时间）
  └── 生成泛化曲线图

2026-05-12
  ├── 实验2：Attention entropy 分析（~1h）
  └── 实验3：RoPE ablation（~2.5h 机器时间）

2026-05-13
  ├── 实验4（可选）
  ├── 整理所有结果图
  └── 开始写 Related Work

2026-05-15
  ├── 写 Introduction + Methodology
  └── 把图表嵌入 LaTeX

2026-05-17
  ├── 写 Experiments + Conclusion
  └── 初稿完成

2026-05-19
  ├── 内部审阅，互相修改
  └── 检查格式、引用、页数

2026-05-21
  ├── 最终润色
  └── 生成 PDF，检查排版

2026-05-23
  └── 提交到 OpenReview（留出 24h buffer）
```

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| MPS 不支持某些操作导致报错 | 高 | 中 | 设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`；手写 attention 规避 SDPA |
| 训练不收敛（loss 不下降）| 中 | 高 | 先用 CPU 跑 2 epoch 验证代码正确性；降低 lr |
| ALiBi 实现有 bug | 中 | 中 | 参考官方实现对照输出；先在小模型上验证 |
| 四种编码训练内 perplexity 差异太大（不公平比较）| 低 | 高 | 统一用相同超参数训练到收敛；若收敛速度不同，以 valid perplexity 最优点为准 |
| 结果与预期相反（如 Sinusoidal 外推更好）| 低 | 低 | 这反而是有趣的 negative result，详细分析原因即可高分 |
| 时间不够做实验4 | 中 | 低 | 实验4本来就是可选的，不做不影响论文质量 |
| 论文超过 8 页 | 中 | 中 | 优先保留核心图表；附录放详细实验配置 |

---

## 7. 验收标准总览

### 代码验收
- [ ] 四种位置编码全部实现，前向传播无报错
- [ ] WikiText-2 数据加载正常
- [ ] 训练20 epoch，valid perplexity 收敛到 < 100
- [ ] 四个模型 checkpoint 保存成功

### 实验验收
- [ ] 实验1：6个测试长度 × 4种编码的完整结果表
- [ ] 实验2：Attention entropy 热力图生成
- [ ] 实验3：4种 base frequency 的 RoPE 对比曲线
- [ ] 所有图表为 PDF 格式，分辨率满足论文要求

### 论文验收
- [ ] 使用官方 LaTeX 模板
- [ ] 主体 6-8 页（双栏）
- [ ] 5 张图表全部嵌入
- [ ] 引用 >= 12 篇，格式正确
- [ ] 四条假设 H1-H4 在论文中都有对应分析
- [ ] Abstract 清晰陈述主要发现

---

## 8. 分工建议

| 角色 | 负责内容 |
|------|----------|
| 成员A（实验主力）| 代码框架、训练脚本、实验1+2 |
| 成员B（实验辅助）| RoPE/ALiBi 实现、实验3+4、图表生成 |
| 成员C（写作主力）| Introduction、Related Work、Methodology、Conclusion |
| 全组 | Experiments 分析、内部审阅、最终检查 |

---

*计划书 v1.0 · 如有调整请同步更新本文件*
