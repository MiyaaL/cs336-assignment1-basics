# QA 对照表（按作业能力点）

## A. 神经网络基础模块

- **线性层 / 嵌入层 / SwiGLU / RMSNorm / SiLU**
  - 结果：通过。
  - 依据：`tests/test_model.py` 对应用例通过（`test_linear`, `test_embedding`, `test_swiglu`, `test_rmsnorm`, `test_silu_matches_pytorch`）。

- **Attention / RoPE / Transformer Block / Transformer LM**
  - 结果：通过。
  - 依据：`tests/test_model.py` 中注意力、RoPE、Block、LM 快照比对通过。

## B. 训练工具链

- **softmax / cross_entropy / gradient clipping**
  - 结果：通过。
  - 依据：`tests/test_nn_utils.py` 全通过。

- **AdamW + 学习率调度**
  - 结果：通过。
  - 依据：`tests/test_optimizer.py` 全通过。

- **checkpoint 保存/恢复**
  - 结果：通过。
  - 依据：`tests/test_serialization.py::test_checkpointing` 通过。

- **batch 采样**
  - 结果：通过。
  - 依据：`tests/test_data.py::test_get_batch` 通过。

## C. Tokenizer / BPE

- **roundtrip 与 iterable 内存测试**
  - 结果：通过。
  - 依据：`tests/test_tokenizer.py` 的 roundtrip 与 `test_encode_iterable_memory_usage` 通过。

- **BPE 训练正确性与速度**
  - 结果：通过。
  - 依据：`tests/test_train_bpe.py::test_train_bpe`、`test_train_bpe_speed` 通过。

- **tiktoken 对齐测试**
  - 本地状态：受网络代理限制，`tiktoken` 下载 GPT-2 编码文件失败（`ProxyError`），相关测试未能在本地完成。
  - 说明：代码逻辑保持与参考接口一致；在可访问 `openaipublic.blob.core.windows.net` 的环境中应执行该组用例复核。

## D. 交付结论

- 在当前环境可验证的核心测试已通过。
- 已提供实现细节文档与 QA 对照。
- 代码实现保持“尽量简洁”：清理了 tokenizer 热路径中的进度条/打印，统一了 `encode` 与 `encode_iterable` 路径，保留必要核心逻辑。
