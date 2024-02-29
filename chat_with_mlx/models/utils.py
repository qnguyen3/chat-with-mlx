import os
import yaml

flags = {'vi': "🇻🇳", 'en': "🏴󠁧󠁢󠁥󠁮󠁧󠁿", 'zh': "🇨🇳", 'es': "🇪🇸", 'tr': "🇹🇷", 'ja': "🇯🇵", 'kr': "🇰🇷", 'in': "🇮🇳", 'de': "🇩🇪", 'fr': "🇫🇷", 'it': "🇮🇹", 'multi': "🌍"}

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
    final_config = config["original_repo"]
    model_lang = ''
    model_quant = ''
    if 'default_language' in config:
        model_lang = config['default_language']
    if 'quantize' in config:
        model_quant = config['quantize']

    if model_lang != '' and model_quant != '':
        final_config += f'- ({flags[model_lang]}, {model_quant})'
    elif model_lang != '' and model_quant == '':
        final_config += f'- ({flags[model_lang]})'
    elif model_lang == '' and model_quant != '':
        final_config += f'- ({model_quant})'
    else:
        final_config = final_config
        
    return {f'{config["original_repo"]}': config["mlx-repo"]}, {f'{config["original_repo"]}': yaml_path}, {f'{final_config}': config["original_repo"]}, {f'{final_config}': config["mlx-repo"]}

def model_info():
    model_list = {}
    yml_list = {}
    final_cfg_list = {}
    mlx_config_list = {}
    directory_path = os.path.dirname(os.path.abspath(__file__))
    yaml_files = get_yaml_files(f'{directory_path}/configs')
    for file in yaml_files:
        model_dict, yml_path, final_cfg, mlx_config = process_yaml(file)
        model_list.update(model_dict)
        yml_list.update(yml_path)
        final_cfg_list.update(final_cfg)
        mlx_config_list.update(mlx_config)

    return model_list, yml_list, final_cfg_list, mlx_config_list