# CS336 Assignment 1 实现细节文档（简洁版）

本文档说明本仓库实现思路，重点覆盖 tokenizer/BPE、Transformer 基础模块、训练工具函数。

## 1. Tokenizer 与 BPE

实现文件：`cs336_basics/tokenizer.py`。

### 1.1 预分词
- 使用 GPT-2 风格正则 `PAT` 做 pre-tokenization。
- 普通文本先按 special token（若提供）切段，再对普通片段做正则切分。
- special token 使用“按长度降序”的正则匹配，保证重叠 token 时优先最长匹配。

### 1.2 编码（encode / encode_iterable）
- `encode(text)` 直接复用 `encode_iterable([text])`，保证逻辑一致。
- `encode_iterable(iterable)` 逐块处理输入并 `yield` token id，不构建全量缓存，内存占用稳定。
- 普通 token 的字节序列通过 `_bpe` 按 merge-rank 迭代合并，直到无可合并 pair。

### 1.3 解码
- `decode(ids)` 将 token id 映射回 bytes 后拼接，再按 UTF-8（`errors="replace"`）解码。

### 1.4 BPE 训练（train_bpe）
- 使用双向链表节点表示每个预分词 token 的字节序列，支持原地合并。
- 通过：
  - `stats[(a,b)]` 维护 pair 频次；
  - `indices[(a,b)]` 维护 pair 出现位置；
  - 堆 `HeapItem` 维护“最高频 + 词典序 tie-break”的候选 pair。
- 每次选中最佳 pair 后：
  1) 新建 token id 与 merge 记录；
  2) 仅局部更新受影响邻接 pair 统计；
  3) 执行链表节点物理合并。
- 最后追加 special tokens 到词表，不参与普通 merge。

### 1.5 GPT-2 文件加载
- `from_files` 支持从 `vocab.json` + `merges.txt` 恢复 tokenizer。
- 使用 `bytes_to_unicode` 的反向映射把可见字符还原回原始 byte。

## 2. Transformer 基础模块

实现文件：`cs336_basics/transfomer.py`。

- `Linear` / `Embedding`：基础参数层，支持权重加载。
- `RMSNorm`：按最后一维做 RMS 归一化，内部升到 `float32` 增强数值稳定性。
- `SwiGLU`：`w2(silu(w1(x)) * w3(x))`。
- `scaled_dot_product_attention`：支持 mask，softmax 前做缩放。
- `multihead_self_attention(_with_rope)`：
  - 先将 QKV 权重 reshape 为多头；
  - 生成 causal mask；
  - 拼接各头输出后做 output projection。
- `TransformerBlock`：pre-norm + 残差结构（attention 子层 + FFN 子层）。
- `TransformerLM`：token embedding + 多层 block + final norm + lm head。

## 3. 训练与优化工具

实现文件：`cs336_basics/lmtrain.py`。

- `cross_entropy`：稳定版本（减去样本级 max logit）。
- `AdamW`：手写优化器，包含 bias-correction 与 decoupled weight decay。
- `learning_rate_schedule`：线性 warmup + cosine decay + 最小学习率下限。
- `gradient_clipping`：全局 L2 norm 裁剪。
- `data_loader`：从 1D token 数据采样 `(x,y)`，满足 `y = x` 右移一位。
- `save_checkpoint` / `load_checkpoint`：序列化模型、优化器与迭代步。

## 4. 兼容你的运行环境（ARM + CentOS + 2xA100-40G）

- 本次实现均为 PyTorch 原生算子，无 x86 特定依赖。
- 多卡训练可后续在训练脚本层加 DDP/FSDP；本作业核心函数不依赖单卡假设。
- 建议：
  - 固定 `torch`, `cuda` 版本组合；
  - tokenizer 训练放 CPU，模型训练放 GPU；
  - 使用 bf16/fp16 + grad accumulation 控制显存。
