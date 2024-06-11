# Add Your Own Models

## Auto

Use the **Model Manager** tab and fill the in all necessary information such as:
- Original repo of the model
- MLX repo of the model
- Quantization of the MLX model
- Supported Language (for RAG)

After filling in all the required fields, hit **Add Model** then **Quit** and restart the program to use the newly added model.

## Manual

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
- `special_tokens_map.json` - this is optional by model
- `tokenizer_config.json`
- `tokenizer.json`
- `vocab.json`