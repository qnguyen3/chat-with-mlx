<div align="center">

# Native RAG on MacOS and Apple Silicon with MLX 🧑‍💻

[![version](https://badge.fury.io/py/chat-with-mlx.svg)](https://badge.fury.io/py/chat-with-mlx)
[![downloads](https://img.shields.io/pypi/dm/chat-with-mlx)](https://pypistats.org/packages/chat-with-mlx)
[![license](https://img.shields.io/pypi/l/chat-with-mlx)](https://github.com/qnguyen3/chat-with-mlx/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/chat-with-mlx)](https://badge.fury.io/py/chat-with-mlx)
</div>

This repository showcases a Retrieval-augmented Generation (RAG) chat interface with support for multiple open-source models.

![chat_with_mlx](assets/chat-w-mlx.gif)

## Features

- **Chat with your Data**: `doc(x), pdf, txt` and YouTube video via URL.
- **Multilingual**: Chinese 🇨🇳, English🏴, French🇫🇷, German🇩🇪, Hindi🇮🇳, Italian🇮🇹, Japanese🇯🇵,Korean🇰🇷, Spanish🇪🇸, Turkish🇹🇷 and Vietnamese🇻🇳
- **Easy Integration**: Easy integrate any HuggingFace and MLX Compatible Open-Source Model.

## Installation and Usage

### Easy Setup

- Install Pip
- Install: `pip install chat-with-mlx`
- Note: Setting up this way is really hard if you want to add your own model (which I will let you add later in the UI), but it is a fast way to test the app.

### Manual Pip Installation

```bash
git clone https://github.com/qnguyen3/chat-with-mlx.git
cd chat-with-mlx
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Manual Conda Installation

```bash
git clone https://github.com/qnguyen3/chat-with-mlx.git
cd chat-with-mlx
conda create -n mlx-chat python=3.11
conda activate mlx-chat
pip install -e .
```

#### Usage

- Start the app: `chat-with-mlx`

## Supported Models

- Google Gemma-7b-it, Gemma-2b-it
- Mistral-7B-Instruct, OpenHermes-2.5-Mistral-7B, NousHermes-2-Mistral-7B-DPO
- Mixtral-8x7B-Instruct-v0.1, Nous-Hermes-2-Mixtral-8x7B-DPO
- Quyen-SE (0.5B), Quyen (4B)
- StableLM 2 Zephyr (1.6B)
- Vistral-7B-Chat, VBD-Llama2-7b-chat, vinallama-7b-chat

## Add Your Own Models

### Solution 1

This solution only requires you to add your own model with a simple .yaml config file in `chat_with_mlx/models/configs`

`examlple.yaml`:

```yaml
original_repo: google/gemma-2b-it # The original HuggingFace Repo, this helps with displaying
mlx-repo: mlx-community/quantized-gemma-2b-it # The MLX models Repo, most are available through `mlx-community`
quantize: 4bit # Optional: [4bit, 8bit]
default_language: multi # Optional: [en, es, zh, vi, multi]
```

After adding the .yaml config, you can go and load the model inside the app (for now you need to keep track the download through your Terminal/CLI)

### Solution 2

Do the same as Solution 1. Sometimes, the `download_snapshot` method that is used to download the models are slow, and you would like to download it by your own.

After the adding the .yaml config, you can download the repo by yourself and add it to `chat_with_mlx/models/download`. The folder name MUST be the same as the orginal repo name without the username (so `google/gemma-2b-it` -> `gemma-2b-it`).

A complete model should have the following files:

- `model.safetensors`
- `config.json`
- `merges.txt`
- `model.safetensors.index.json`
- `special_tokens_map.json` - this is optinal by model
- `tokenizer_config.json`
- `tokenizer.json`
- `vocab.json`

## Known Issues

- You HAVE TO unload a model before loading in a new model. Otherwise, you would need to restart the app to use a new model, it would stuck at the old one.
- When the model is downloading by Solution 1, the only way to stop it is to hit `control + C` on your Terminal.
- If you want to switch the file, you have to manually hit STOP INDEXING. Otherwise, the vector database would add the second document to the current database.
- You have to choose a dataset mode (Document or YouTube) in order for it to work.

## WHY MLX?

MLX is an array framework for machine learning research on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy.  MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API.  MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
   dynamically. Changing the shapes of function arguments does not trigger
   slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
   (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
   is the *unified memory model*. Arrays in MLX live in shared memory.
   Operations on MLX arrays can be performed on any of the supported
   device types without transferring data.

## Acknowledgement

I would like to send my many thanks to:

- The Apple Machine Learning Research team for the amazing MLX library.
- LangChain and ChromaDB for such easy RAG Implementation
- People from Nous, VinBigData and Qwen team that helped me during the implementation.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qnguyen3/chat-with-mlx&type=Date)](https://star-history.com/#qnguyen3/chat-with-mlx&Date)
