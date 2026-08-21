# DNA-4: Neural Data Compression with RWKV-7

DNA-4, short for **Dynamical Neural Adaptation from a 4-bit Prior**, is an end-to-end neural data compressor based on an RWKV-7 recurrent language model. It converts an input file into dictionary tokens, predicts the token stream with a trained model, and uses arithmetic coding for lossless compression. The stored model prior uses seeded orthogonal rotations and 4-bit NF4 quantization.

The project is based on [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) and includes a standalone NNCP-style dictionary preprocessor derived from [NNCP](https://github.com/fabricebellard/nncp).

## Features

- RWKV-7 neural prediction with streaming arithmetic coding
- Seeded orthogonal rotation and 4-bit NF4 model quantization
- Synchronized online adaptation during compression and decompression
- NNCP-style dictionary preprocessing for byte streams
- A complete CLI pipeline for model decoding, token decoding, and raw-file restoration
- CUDA kernels for RWKV-7 training and inference

## Repository Layout

```text
.
├── train.py                    # RWKV-7 training entry point
├── dna4_cli.py                 # Compression and decompression CLI
├── preprocess                  # Prebuilt NNCP-style preprocessor
├── preprocess.c                # Preprocessor source for rebuilding the binary
├── demo-training-prepare.sh    # Create the initial model weights
├── demo-training-run.sh        # Train the RWKV-7 prior
├── demo-dna4.sh                # Compress with a trained prior
├── demo-model-config.sh        # Shared enwik9 model configuration
├── requirements.txt            # Python dependencies
├── src/
│   ├── AC/                     # C++ arithmetic coder and PyTorch bindings
│   ├── inference/              # RWKV-7 inference implementation and CUDA kernels
│   ├── training/               # RWKV-7 training model and CUDA kernels
│   ├── dna4_engine.py          # Adaptive coding engine
│   ├── model_codec.py          # NF4 model codec
│   └── nf4_rotation_optimizer.py
├── data/                       # Local input and token files; ignored by Git
├── out/                        # Local checkpoints; ignored by Git
├── logs/                       # Local logs; ignored by Git
└── enwik9.dna4/                # Local compressed archive; ignored by Git
```

The repository intentionally does not track raw data, token files, model checkpoints, compressed archives, logs, caches, or other generated files. The prebuilt `preprocess` executable is tracked; `preprocess.c` is provided as a fallback for systems on which that binary cannot run.

## Requirements

The compression and inference pipeline requires an NVIDIA GPU and a CUDA toolchain for the first-run JIT compilation. Training additionally uses PyTorch Lightning and DeepSpeed.

The demo scripts do not create or activate a Python environment. Create one
with your preferred environment manager, activate it in the current shell, and
then run the scripts from the repository root. For example, with Conda:

```bash
conda create -n dna-rwkv python=3.12.1 -y
conda activate dna-rwkv

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0
python -m pip install \
    pytorch-lightning==1.9.5 \
    deepspeed==0.18.4 \
    bitsandbytes==0.49.0 \
    scipy==1.13.1 \
    numpy==1.26.4 \
    ninja==1.11.1.1 \
    tqdm
```

`dna-rwkv` is only an example environment name; choose any name appropriate to
your machine. The important requirement is that the intended environment is
already active before invoking a demo script.

### Validated Reproduction Environment

The included demo scripts verify Python and PyTorch at startup. Their supported
reproduction environment is:

| Component | Version |
|---|---|
| Operating system | Linux |
| Python | 3.12.1 |
| PyTorch | 2.7.0+cu126 |
| PyTorch Lightning | 1.9.5 |
| DeepSpeed | 0.18.4 |
| bitsandbytes | 0.49.0 |
| SciPy | 1.13.1 |
| NumPy | 1.26.4 |
| Ninja | 1.11.1.1 |

Check the active environment before running a demo:

```bash
python - <<'PY'
import sys
import torch

print(sys.executable)
print(sys.version)
print(torch.__version__)
assert sys.version_info[:3] == (3, 12, 1)
assert torch.__version__ == "2.7.0+cu126"
PY
```

`requirements.txt` lists the broad runtime dependencies, but it does not pin a
reproducible CUDA/PyTorch stack. Use the commands above for the demo
configuration. Record the exact source revision with `git rev-parse HEAD` when
comparing results.

An identical GPU model is not required for training, compression,
decompression, or lossless restoration. However, GPU architecture, CUDA,
driver, PyTorch, and DeepSpeed versions can change floating-point evaluation
and cause training trajectories or compressed archive bytes to diverge.

Stage I initialization runs on CPU with its own fixed seed. Stage III training
uses a separate fixed seed and deterministic settings. To compare exact
checkpoints, also match the CPU software stack, GPU model, driver, and PyTorch
version; cross-hardware bitwise identity is not guaranteed.

Lossless recovery is a separate property: a valid archive must restore the original input exactly on any supported compatible system, even when its generated archive bytes differ from an archive produced elsewhere.

The tested model configuration is RWKV-7 x070 with 5 layers, embedding size 512, vocabulary size 16,384, context length 2,048, and head size 64.

## 1. Prepare the Preprocessor

The release includes an executable `preprocess`. Try it first:

```bash
./preprocess -h
```

If the binary is incompatible with the target operating system or CPU architecture, rebuild it from the included source:

```bash
gcc -O3 -Wall -DCONFIG_STANDALONE preprocess.c -o preprocess -lm
```

## 2. Tokenize the Input

The DNA-4 CLI consumes the preprocessed token file, not the original raw file. Create the dictionary and token stream with:

```bash
mkdir -p data enwik9.dna4
./preprocess c enwik9.dna4/nncp.dict enwik9 data/enwik9_tokens.bin 16384 512
```

The command creates:

- `enwik9.dna4/nncp.dict`: the dictionary required for final restoration
- `data/enwik9_tokens.bin`: the big-endian 16-bit token stream used by training and compression

The generic command form is:

```bash
./preprocess c <dictionary> <input> <token_output> <n_words> <min_freq>
```

## 3. Train the RWKV-7 Prior

Ensure the environment described above is active, and run these commands from
the repository root. Model and data settings shared by all demos are in
`demo-model-config.sh`.

Create the initial weights:

```bash
bash demo-training-prepare.sh
```

This creates:

```text
out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-init.pth
```

Train the prior:

```bash
bash demo-training-run.sh
```

`train.py` automatically resumes the highest numbered `rwkv-*.pth` in the
project directory during Stage III. To begin a fresh run from `rwkv-init.pth`,
use a project directory containing only that initial checkpoint; preserve any
older checkpoints elsewhere rather than deleting them.

Training checkpoints and logs are written to the same project directory:

```text
out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/
├── rwkv-init.pth
├── rwkv-0.pth, rwkv-20.pth, ...
├── rwkv-final.pth
└── train_log.txt
```

The training script uses the 5-layer, 512-dimensional configuration. It is intended to be edited for a different dataset, model size, or hardware setup.

## 4. Compress

After training the checkpoint selected by `MODEL_PATH` in `demo-dna4.sh`, run:

```bash
bash demo-dna4.sh
```

The demo uses the shared `PROJ_DIR` configuration, expects the dictionary at
`enwik9.dna4/nncp.dict`, and currently selects `rwkv-200.pth`. Change
`MODEL_PATH`, `ARCHIVE_DIR`, or `SEED_TRIALS` in `demo-dna4.sh` deliberately if
your trained checkpoint or archive name differs.

The equivalent direct CLI invocation is:

```bash
python dna4_cli.py compress \
    --input_file data/enwik9_tokens.bin \
    --model_path out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-200.pth \
    --archive_dir enwik9.dna4 \
    --seed_trials 100 \
    --weight_tying 1 \
    --device cuda
```

The compression pipeline performs the following steps:

1. Reuses the dictionary in `archive_dir`.
2. Searches rotation seeds for eligible model tensors.
3. Quantizes and arithmetic-codes the model weights.
4. Reloads the quantized model so encoder and decoder start from the same state.
5. Runs synchronized adaptive prediction and arithmetic-codes the token stream.

The archive directory receives:

```text
enwik9.dna4/
├── nncp.dict
├── rotation_seeds.pt
├── dna4_compressed_model.ac
├── dna4_compressed_model.meta
└── dna4_stream.ac
```

Existing rotation seeds and model codec files are reused by default. Use `--force_rotation` or `--force_model_codec` to regenerate them.

## 5. Decompress and Restore the Raw File

The default output locations are under `data/`:

```bash
python dna4_cli.py decompress \
    --archive_dir enwik9.dna4 \
    --total_tokens 200608961 \
    --device cuda
```

The CLI first decodes the token stream and then automatically runs `preprocess d`. It creates:

```text
data/enwik9_restored.bin  # decoded token stream
data/enwik9_restored      # restored raw file
```

To choose different paths:

```bash
python dna4_cli.py decompress \
    --archive_dir enwik9.dna4 \
    --output_file data/restored_tokens.bin \
    --raw_output_file data/restored.raw \
    --total_tokens 200608961 \
    --device cuda
```

If the preprocessor is not in the project root or on `PATH`, specify it explicitly:

```bash
python dna4_cli.py decompress \
    --archive_dir enwik9.dna4 \
    --preprocess_bin ./preprocess \
    --total_tokens 200608961 \
    --device cuda
```

`--total_tokens` must exactly match the number of tokens in the compressed input. A mismatch changes the lane layout and prevents correct recovery.

## Generated Files

During normal use, files are generated in these locations:

| Path | Purpose |
|---|---|
| `data/` | Raw input, tokenized input, decoded tokens, and restored raw output |
| `out/<project>/` | Initial weights and training checkpoints |
| `<archive>.dna4/` | Dictionary, model payload, rotation seeds, and coded data stream |
| `logs/batch_inf/` | Compression and decompression logs |
| PyTorch extension cache | JIT-built CUDA and arithmetic-coder extensions |

All of these generated project files are ignored by Git. Only source code, scripts, documentation, dependency metadata, and the prebuilt preprocessor are tracked.

## Implementation Overview

The stored prior is built from an offline-trained RWKV-7 model. Eligible large tensors are transformed with seeded orthogonal rotations and quantized blockwise to NF4. Small or sensitive tensors are retained without NF4 quantization. The resulting model indices are arithmetic-coded together with the quantization metadata.

During data coding, the quantized model is restored in BF16. The token stream is arranged into parallel lanes and processed in chunks. The current model distribution is converted to integer cumulative frequencies for the arithmetic coder. After each coding window, encoder and decoder apply the same online update to the same known token window, so no adapted weights need to be transmitted.

The final token stream is converted back to the original byte stream using the retained dictionary.

## Notes

- The first execution JIT-compiles CUDA and arithmetic-coder extensions and may take several minutes.
- Reduce the training micro-batch or inference batch size if GPU memory is insufficient.
- Activate the required Python environment yourself before running any demo;
  the scripts intentionally do not assume a Conda installation path or an
  environment name.
- Change the shared model/data settings in `demo-model-config.sh`; keep
  training and compression-specific settings in their respective demo scripts.
- The current implementation targets CUDA execution and is not a CPU-only Hutter Prize submission.

## License and Acknowledgements

The project follows the RWKV-LM project license. The NNCP preprocessor source is Copyright (c) 2018--2021 Fabrice Bellard and follows its original license.

This project builds on:

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [NNCP](https://github.com/fabricebellard/nncp)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
