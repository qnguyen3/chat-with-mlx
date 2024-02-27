from mlx_lm import convert

upload_repo = "ontocord/vinallama-7b-chat-mlx-4bit"

convert("vilm/vinallama-7b-chat", quantize=True, upload_repo=upload_repo)