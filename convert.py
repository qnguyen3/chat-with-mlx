from mlx_lm import convert

upload_repo = "Viet-Mistral/Vistral-7B-Chat-mlx-4bit"

convert("Viet-Mistral/Vistral-7B-Chat", quantize=True, upload_repo=upload_repo)