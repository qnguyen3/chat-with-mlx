<div align="center">

# Native RAG on Apple Sillicon Mac with MLX 🧑‍💻

[![version](https://badge.fury.io/py/chat-with-mlx.svg)](https://badge.fury.io/py/chat-with-mlx)
[![downloads](https://img.shields.io/pypi/dm/chat-with-mlx)](https://pypistats.org/packages/chat-with-mlx)
[![license](https://img.shields.io/pypi/l/chat-with-mlx)](https://github.com/qnguyen3/chat-with-mlx/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/chat-with-mlx)](https://badge.fury.io/py/chat-with-mlx)
</div>

This repository showcases a Retrieval-augmented Generation (RAG) chat interface with support for multiple open-source models.

![chat_with_mlx](assets/Logo.png)

## Features

- **Privacy-enhanced AI**: Chat with your favourite models and data securely.
- **MLX Playground**: Your all in one LLM Chat UI for Apple MLX
- **Easy Integration**: Easy integrate any HuggingFace and MLX Compatible Open-Source Models.
- **Default Models**: Llama-3, Phi-3, Yi, Qwen, Mistral, Codestral, Mixtral, StableLM (along with Dolphin and Hermes variants)

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

## Add Your Model

Please checkout the guide [HERE](ADD_MODEL.MD)

## Known Issues

- When the model is downloading by Solution 1, the only way to stop it is to hit `control + C` on your Terminal.
- If you want to switch the file, you have to manually hit STOP INDEXING. Otherwise, the vector database would add the second document to the current database.
- You have to choose a dataset mode (Document or YouTube) in order for it to work.
- **Phi-3-small** can't do streaming in completions

## Why MLX?

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
- All contributors

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qnguyen3/chat-with-mlx&type=Date)](https://star-history.com/#qnguyen3/chat-with-mlx&Date)
