# COMP5329 Assignment 2 — 研究计划

**课题：** Positional Encoding and Length Generalization in Transformers  
**假设：** 不同位置编码在训练短序列、推理长序列时存在系统性差异，且差异根源可通过 attention 模式解释  
**截止：** 论文 2026-05-24 23:59  
**平台：** Mac mini M4 Pro 48GB · PyTorch MPS  
**人员：** 3人

---

## 一、研究问题与假设

### 核心问题
> 当 Transformer 在固定长度 L_train 上训练，推理时遇到 L_test > L_train 的序列，四种主流位置编码（Sinusoidal、Learnable、RoPE、ALiBi）的泛化能力有何本质差异？差异的根源能否通过注意力模式量化解释？

### 具体假设（每条都要被实验验证/证伪）
- **H1：** ALiBi 和 RoPE 的 perplexity 随序列长度增长显著优于 Sinusoidal 和 Learnable
- **H2：** Sinusoidal 在超出训练长度后 attention entropy 会崩溃（注意力集中到少数位置）
- **H3：** RoPE 的 base frequency 越小，外推到更长序列时性能衰减越快
- **H4：** Learnable 在训练长度内性能最好，但超出训练长度后立即失效

---

## 二、模型设计

自建小型 Transformer，确保唯一变量是位置编码：

```
- 层数：6
- 注意力头数：8
- 隐藏维度：512
- FFN 维度：2048
- Dropout：0.1
- 参数量：约 25M（M4 Pro 可轻松容纳）
```

四个模型完全相同，只换位置编码模块。

---

## 三、实验设计

### 数据集
- **WikiText-2**（语言建模，标准 benchmark）
- 训练序列长度：**128 tokens**（截断/填充）
- 测试序列长度：**64 / 128 / 192 / 256 / 384 / 512**

### 实验1：泛化曲线（核心实验）
- 指标：Perplexity（越低越好）
- 输出：每种编码 × 每个测试长度 → 折线图
- 预期：RoPE/ALiBi 曲线平缓，Learnable/Sinusoidal 急剧上升

### 实验2：Attention 模式分析（可视化出图）
- 对比各编码在 L_test=512 时各层的 attention map
- 计算 attention entropy：H = -Σ a_ij * log(a_ij)
- 分析：熵崩溃意味着注意力失效

### 实验3：RoPE Base Frequency Ablation
- 固定架构，测试 base ∈ {100, 1000, 10000, 100000}
- 验证 H3，同时解释为什么 LLaMA 用 10000

### 实验4：训练长度敏感性（可选，时间够就做）
- 固定 L_test=256，改变 L_train ∈ {64, 128, 256}
- 看外推比例（L_test/L_train）对结果的影响

---

## 四、论文结构

```
1. Abstract         — 问题、方法、主要发现（150词）
2. Introduction     — 为什么位置编码外推重要，现有研究缺口
3. Related Work     — 四种编码的原始论文，已有对比研究
4. Methodology      — 模型架构、实验设置、评估指标
5. Experiments      — 四个实验的结果与分析
6. Conclusion       — 总结发现，局限性，未来方向
7. References
```

目标：**7页主体 + 附录（额外实验细节）**

---

## 五、时间线

| 日期 | 任务 | 负责 |
|------|------|------|
| 5月9日 | 搭建代码框架，跑通 Sinusoidal baseline | 全组 |
| 5月11日 | 实现四种位置编码，跑实验1 | 实验为主 |
| 5月13日 | 实验2（attention 可视化） | 实验为主 |
| 5月14日 | 实验3（ablation） | 实验为主 |
| 5月16日 | 开始写论文 Introduction + Related Work | 写作为主 |
| 5月18日 | 写 Methodology + Experiments | 全组 |
| 5月20日 | 初稿完成，内部审阅 | 全组 |
| 5月22日 | 修改润色，格式检查 | 全组 |
| 5月23日 | 最终检查，提交到 OpenReview | 全组 |

---

## 六、代码结构

```
homework/
├── src/
│   ├── model.py          # Transformer 主体
│   ├── pe/
│   │   ├── sinusoidal.py
│   │   ├── learnable.py
│   │   ├── rope.py
│   │   └── alibi.py
│   ├── data.py           # WikiText-2 数据加载
│   ├── train.py          # 训练主循环
│   ├── evaluate.py       # 评估 + perplexity 计算
│   └── visualize.py      # Attention entropy 可视化
├── experiments/
│   ├── run_all.sh        # 一键跑全部实验
│   └── configs/          # 各实验配置文件
├── results/              # 实验结果 JSON + 图
├── paper/                # LaTeX 论文
│   └── (LaTeX template files)
└── PLAN.md
```

---

## 七、关键引用文献

1. Vaswani et al. (2017) — Attention is All You Need（Sinusoidal）
2. Devlin et al. (2018) — BERT（Learnable）
3. Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding
4. Press et al. (2021) — Train Short, Test Long: ALiBi
5. Kazemnejad et al. (2023) — The Impact of Positional Encoding on Length Generalization in Transformers

---

## 八、评分策略

| 评分维度 | 我们的应对 |
|----------|-----------|
| 新颖性(25) | 系统性四方案对比 + attention entropy 机制分析，不是简单性能比较 |
| 方法严谨性(30) | 控制变量严格，有 ablation，有统计可重复性 |
| 表达清晰(20) | 结构清晰，图表驱动，每个假设都有对应结果 |
| 文献定位(20) | 引用原始论文 + 近期外推研究，明确说明我们的贡献 |
| 格式(5) | LaTeX 模板，严格控制在 8 页内 |
