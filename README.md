# DNA-4: Neural Data Compression with RWKV-7

**DNA-4** (Deep Neural Arithmetic coding, 4th generation) 是一个基于 **RWKV-7** 语言模型的端到端神经网络数据压缩系统。它利用经过训练的语言模型对 token 序列进行自适应熵编码，并结合 NF4 量化、种子正交旋转优化和算术编码（Arithmetic Coding），实现高压缩比的无损数据压缩。

> 本项目基于 [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) 框架开发，并集成了 [NNCP](https://github.com/fabricebellard/nncp) 风格的字典预处理。

---

## 📌 核心特性

- **端到端神经网络压缩**：训练 RWKV-7 语言模型预测 token 分布，通过算术编码实现近熵率无损压缩
- **NF4 量化 + 种子正交旋转**：模型权重经旋转优化后量化为 4-bit，显著减小模型存储体积
- **自适应在线微调**：压缩时对模型进行在线 fine-tuning，提升长序列的预测精度
- **算术编码流式处理**：C++ 实现的高性能流式算术编码器，支持无限追加编码与解码
- **NNCP 字典预处理**：基于 Fabrice Bellard 的 NNCP 预处理器进行字典编码，进一步降低熵率
- **CLI 压缩/解压工具链**：CLI 编解码 token 流；字节级恢复由随附的 `preprocess` 工具完成

---

## 🏗️ 项目结构

```
.
├── train.py                    # 模型训练入口（PyTorch Lightning）
├── dna4_cli.py                 # 压缩 / 解压缩 CLI 工具
├── preprocess                  # 已编译的 NNCP 字典预处理器
├── preprocess.c                # 预处理器源码；二进制不兼容时用于重新编译
├── demo-training-prepare.sh    # 初始化模型权重脚本
├── demo-training-run.sh        # 启动训练脚本
│
├── src/
│   ├── training/
│   │   ├── model.py            # RWKV-7 训练模型定义（x070 架构）
│   │   └── cuda/
│   │       ├── wkv7_cuda.cu    # WKV7 CUDA 前向/反向内核
│   │       └── wkv7_op.cpp     # CUDA 算子注册
│   │
│   ├── inference/
│   │   ├── batch_inf.py        # RWKV-7 批量推理引擎（含 CUDA 加速）
│   │   └── cuda/
│   │       ├── rwkv7_fast_ops_bf16.cpp/.cu   # 推理快速算子
│   │       └── rwkv7_wkv_fp32_v2.cpp/.cu     # WKV 推理内核
│   │
│   ├── AC/                     # 算术编码器（C++ / PyTorch JIT 扩展）
│   │   ├── ac_wrapper_torch.cpp     # PyBind11 流式编码/解码绑定
│   │   ├── ArithmeticCoder.cpp/.hpp # 算术编码核心实现
│   │   └── BitIoStream.cpp/.hpp     # 位 I/O 流
│   │
│   ├── model_codec.py          # 模型 NF4 量化压缩 / 解压
│   ├── nf4_rotation_optimizer.py   # 正交旋转种子搜索
│   ├── dna4_engine.py          # DNA-4 核心压缩/解压引擎
│   ├── dataset.py              # 训练数据集加载
│   ├── binidx.py               # MMap 索引数据集读取
│   └── trainer.py              # 训练回调 & 权重初始化
│
├── data/                       # 原始文件与预处理后的 token 文件（本地生成）
├── enwik9.dna4/                # enwik9 压缩归档
│   ├── dna4_compressed_model.ac    # 量化模型权重（算术编码）
│   ├── dna4_compressed_model.meta  # 模型元信息
│   ├── dna4_stream.ac              # 数据流（算术编码）
│   ├── nncp.dict                   # 字典文件
│   └── rotation_seeds.pt           # 旋转种子
│
├── out/                        # 训练输出（模型检查点）
└── logs/                       # 训练/压缩日志（非解压必需）
```

---

## 🔧 环境依赖

### 硬件要求

- **GPU**：NVIDIA GPU（推荐 CUDA Compute Capability ≥ 7.0，即 V100 / A100 / RTX 30xx 及以上）
- **内存**：≥ 16 GB（训练时视模型大小和 batch size 而定）
- **CUDA**：需安装 CUDA Toolkit（用于 JIT 编译 CUDA 内核）

### 软件依赖

| 依赖 | 用途 |
|------|------|
| Python ≥ 3.8 | 运行环境 |
| PyTorch ≥ 2.0 | 深度学习框架 |
| PyTorch Lightning | 训练框架 |
| DeepSpeed | 训练脚本使用的优化器 |
| bitsandbytes | NF4 量化 |
| SciPy | 正交旋转生成 |
| NumPy | 数值计算 |
| tqdm | 旋转种子搜索进度显示 |
| GCC | 编译 C 预处理器 & CUDA JIT |

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 第 1 步：准备 NNCP 预处理器

release 附带可直接执行的 `preprocess`。先确认当前设备可以运行它：

```bash
./preprocess -h
```

若因系统架构、动态链接库或可执行格式不兼容而无法运行，再使用随附源码重新编译：

```bash
gcc -O3 -Wall -DCONFIG_STANDALONE preprocess.c -o preprocess -lm
```

### 第 2 步：准备数据

将原始文本数据通过 NNCP 预处理器转换为 token 序列：

```bash
# 编码：构建字典 + 将文本转换为 token
./preprocess c <dict_file> <input_file> <output_file> <n_words> <min_freq>

# 示例：enwik9
./preprocess c enwik9.dna4/nncp.dict enwik9 data/enwik9_tokens.bin 16384 512

```

解码（恢复原始文本）：

```bash
./preprocess d <dict_file> <input_file> <output_file>
./preprocess d data/my_dict.bin data/enwik9_tokens.bin enwik9_restored
```

### 第 3 步：训练 RWKV-7 模型

**3a. 初始化模型权重：**

```bash
bash demo-training-prepare.sh
```

此脚本会根据配置生成 `rwkv-init.pth` 到输出目录。随附脚本的默认配置为：
- `N_LAYER`：5
- `N_EMBD`：512
- `CTX_LEN`：上下文长度（默认 2048）
- `VOCAB_SIZE`：16,384

**3b. 启动训练：**

```bash
bash demo-training-run.sh
```

可在脚本中调整：
- `M_BSZ`：micro batch size（默认 24，减小以省显存）
- `LR_INIT` / `LR_FINAL`：学习率
- `GRAD_CP`：梯度检查点（1 = 省显存但慢，0 = 快但费显存）
- `EPOCH_SAVE`：每多少 epoch 保存检查点

训练完成后，模型检查点保存在 `out/` 目录下。

### 第 4 步：压缩数据

```bash
python dna4_cli.py compress \
    --input_file data/enwik9_tokens.bin \
    --model_path out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-200.pth \
    --archive_dir enwik9.dna4 \
    --seed_trials 100 \
    --device cuda
```

压缩参数说明：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_file` | 预处理后的 token 文件 (.bin) | 必填 |
| `--model_path` | 训练好的 RWKV 模型路径 (.pth) | 必填 |
| `--archive_dir` | 压缩输出目录 | `enwik9.dna4` |
| `--seed_trials` | 正交旋转种子搜索次数 | 20 |
| `--n_layer` | 模型层数 | 5 |
| `--n_embd` | 嵌入维度 | 512 |
| `--vocab_size` | 词表大小 | 16384 |
| `--batch_size_inf` | 推理 batch size | 4096 |
| `--chunk_size` | 分块大小 | 512 |

> 压缩时会自动执行：旋转种子优化 → 模型 NF4 量化 → 在线微调推理 → 算术编码输出

### 第 5 步：解压数据

```bash
python dna4_cli.py decompress \
    --archive_dir enwik9.dna4 \
    --total_tokens 200608961 \
    --device cuda
```

解压参数说明：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--archive_dir` | 压缩包目录 | `enwik9.dna4` |
| `--output_file` | 恢复的 token 文件路径 | `data/enwik9_restored.bin` |
| `--raw_output_file` | 恢复的原始字节文件路径 | 默认移除 `--output_file` 的 `.bin` 后缀 |
| `--preprocess_bin` | 已编译的 `preprocess` 可执行文件路径 | release 根目录的 `preprocess`，其次为 `PATH` |
| `--total_tokens` | 原始 token 总数（必须精确匹配） | 200608961 |
| `--verify` | 解压时验证模型一致性 | False |
| `--model_path` | 原始模型路径（仅验证时需要） | — |

解压命令会自动调用 `preprocess d` 完成原始字节恢复；`--output_file` 保留 token 流，`--raw_output_file` 指定最终文件。

### 跨设备解压验证

`DNA_RWKV_release_v1_decoder_verification.zip` 是精简的解压验证包，不含原始数据、训练检查点、日志和训练入口。解压后，在包根目录执行：

```bash
pip install -r requirements.txt
./preprocess -h
sha256sum -c ARCHIVE_SHA256SUMS
python dna4_cli.py decompress --archive_dir enwik9.dna4 --total_tokens 200608961 --device cuda
sha256sum data/enwik9_restored
```

如果 `./preprocess -h` 无法执行，使用 `gcc -O3 -Wall -DCONFIG_STANDALONE preprocess.c -o preprocess -lm` 重建后再继续。首次运行会为本机编译 CUDA 与算术编码扩展。预期的最终字节文件 SHA-256 为 `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc`。

---

## 🧠 技术原理

### 压缩流程

```
原始文本 → NNCP字典编码 → token序列 → RWKV-7预测概率分布 → 算术编码 → 压缩流
                                      ↑
                               模型NF4量化 + 在线微调
```

1. **NNCP 字典预处理**：基于频率的字典构建算法，将常见词组编码为单个 token，降低序列熵率
2. **RWKV-7 语言模型**：基于线性注意力机制的 RNN 架构，在训练和推理中均具有 O(T) 复杂度，适合长序列建模
3. **NF4 量化 + 种子正交旋转**：
   - 通过搜索最优旋转种子，最小化权重矩阵的峰度（kurtosis），使量化误差更均匀
   - 旋转后的权重使用 4-bit NormalFloat 量化，配合算术编码进一步压缩
4. **自适应在线微调**：在压缩过程中，对量化后的模型在当前数据块上进行微调训练，提升预测精度
5. **算术编码**：将模型输出的概率分布转换为二进制位流，实现近熵率编码

### 解压流程

```
压缩流 → 算术解码 → token序列 → NNCP字典解码 → 原始文本
            ↑
     量化模型反量化恢复
```

解压是压缩的精确逆过程，保证无损恢复。

### 压缩归档结构

一个 `.dna4` 归档目录包含：

| 文件 | 说明 |
|------|------|
| `dna4_compressed_model.ac` | NF4 量化后的模型权重（算术编码） |
| `dna4_compressed_model.meta` | 模型元信息（pickle 格式：量化状态、旋转种子等） |
| `dna4_stream.ac` | 数据流的算术编码输出 |
| `nncp.dict` | NNCP 字典文件 |
| `rotation_seeds.pt` | 正交旋转种子配置（种子也写入模型元数据） |

---

## ⚙️ 模型架构 (RWKV-7 x070)

本项目使用 **RWKV-7** (代号 x070) 架构，核心组件：

- **RWKV_Tmix_x070**：时间混合层，包含 LoRA 风格的 decay/gate/AAA 模块、value residual、GroupNorm
- **RWKV_CMix_x070**：通道混合层（FFN），使用 ReLU² 激活函数
- **WindBackstepping**：自定义 CUDA 内核，实现 WKV7 的高效前向/反向传播
- **权重绑定**：可选的 embedding ↔ head 权重共享

已验证的模型配置：

| 配置 | 层数 | 维度 | 词表大小 | 上下文长度 |
|------|------|------|----------|------------|
| 小模型 | 2 | 256 | 4096 | 2048 |
| 中模型 | 3 | 256 | 4096 | 2048 |
| 中大模型 | 4 | 128 | — | 2048 |
| 大模型 | 5 | 512 | 16384 | 2048 |

---

## 📊 性能参考

压缩日志保存在 `logs/batch_inf/` 目录，记录了压缩过程中的 BPC (bits per character) 和预估压缩大小。

---

## 📝 注意事项

1. **`--total_tokens` 必须精确**：解压时必须提供与原始文件完全一致的 token 总数，否则解码结果会错位
2. **`magic_prime` 计算**：训练时需计算 `magic_prime`（最大的满足 3n+2 且小于 `datalen/ctxlen-1` 的素数），可用 [dcode.fr](https://www.dcode.fr/prime-numbers-search) 辅助计算
3. **显存管理**：减小 `--micro_bsz` / `--batch_size_inf`、开启 `--grad_cp` 可降低显存占用
4. **检查点恢复**：训练器自动加载输出目录中最新的 `rwkv-*.pth`，支持断点续训
5. **CUDA JIT 编译**：首次运行时会自动编译 CUDA 内核和算术编码器，可能需要几分钟

---

## 📄 许可证

- **项目代码**：遵循 RWKV-LM 项目许可证
- **NNCP 预处理器** (`preprocess.c`)：Copyright (c) 2018-2021 Fabrice Bellard，遵循其原始许可证

---

## 🙏 致谢

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — RWKV 语言模型框架
- [NNCP](https://github.com/fabricebellard/nncp) — Fabrice Bellard 的神经网络数据压缩
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — NF4 量化实现
