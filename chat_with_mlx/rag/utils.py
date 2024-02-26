from ..models.utils import load_yaml_config
from .rag_prompt import *


def get_prompt(yaml_path):
    config = load_yaml_config(yaml_path)
    if 'default_language' in config:
        lang = config['default_language']
        if lang == 'vi':
            return rag_prompt_default_vi, rag_prompt_history_default_vi
    else:
        return rag_prompt_default_en, rag_prompt_history_default_en
