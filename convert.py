from mlx_lm import convert

upload_repo = "vilm/Quyen-v0.1-mlx-4bit"

convert("vilm/Quyen-v0.1", quantize=True, upload_repo=upload_repo)