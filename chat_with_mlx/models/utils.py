import os
import yaml

def get_yaml_files(directory):
    yaml_files = []
    for entry in os.listdir(directory):
        if entry.endswith('.yaml'):
            yaml_files.append(os.path.join(directory, entry))
    return yaml_files

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_yaml(yaml_path):
    config = load_yaml_config(yaml_path)
    return {f'{config["original_repo"]}': config["mlx-repo"]}

def model_info():
    model_list = {}
    yaml_files = get_yaml_files('chat_with_mlx/models/configs/')
    for file in yaml_files:
        model_dict = process_yaml(file)
        model_list.update(model_dict)

    return model_list