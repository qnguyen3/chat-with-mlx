from ..models.utils import load_yaml_config
from .rag_prompt import *

prompt_dict = {'en': [rag_prompt_default_en, rag_prompt_history_default_en],
               'vi': [rag_prompt_default_vi, rag_prompt_history_default_vi],
               'es': [rag_prompt_default_es, rag_prompt_history_default_es],
               'zh': [rag_prompt_default_zh, rag_prompt_history_default_zh],
               'multi': [rag_prompt_default_en, rag_prompt_history_default_en]
               }




def get_prompt(yaml_path, lang):
    config = load_yaml_config(yaml_path)
    sys_prompt = None
    if 'system_prompt' in config:
        sys_prompt = config['system_prompt']
    if lang == 'default':
        if 'default_language' in config:
            lang = config['default_language']
            return prompt_dict[lang], sys_prompt
        else:
            return prompt_dict['en'], sys_prompt
    elif lang == 'English':
         return prompt_dict['en'], sys_prompt
    elif lang == 'Vietnamese':
        return prompt_dict['vi'], sys_prompt
    elif lang == 'Spanish':
        return prompt_dict['es'], sys_prompt
    elif lang == 'Chinese':
        return prompt_dict['zh'], sys_prompt
    else:
        return prompt_dict['en'], sys_prompt
