from chat_with_mlx.models.utils import load_yaml_config
from chat_with_mlx.rag.rag_prompt import (
    rag_prompt_default_de,
    rag_prompt_default_en,
    rag_prompt_default_es,
    rag_prompt_default_fr,
    rag_prompt_default_in,
    rag_prompt_default_it,
    rag_prompt_default_ja,
    rag_prompt_default_kr,
    rag_prompt_default_tr,
    rag_prompt_default_vi,
    rag_prompt_default_zh,
    rag_prompt_history_default_de,
    rag_prompt_history_default_en,
    rag_prompt_history_default_es,
    rag_prompt_history_default_fr,
    rag_prompt_history_default_in,
    rag_prompt_history_default_it,
    rag_prompt_history_default_ja,
    rag_prompt_history_default_kr,
    rag_prompt_history_default_tr,
    rag_prompt_history_default_vi,
    rag_prompt_history_default_zh,
)

prompt_dict = {
    "en": [rag_prompt_default_en, rag_prompt_history_default_en],
    "vi": [rag_prompt_default_vi, rag_prompt_history_default_vi],
    "es": [rag_prompt_default_es, rag_prompt_history_default_es],
    "zh": [rag_prompt_default_zh, rag_prompt_history_default_zh],
    "tr": [rag_prompt_default_tr, rag_prompt_history_default_tr],
    "ja": [rag_prompt_default_ja, rag_prompt_history_default_ja],
    "kr": [rag_prompt_default_kr, rag_prompt_history_default_kr],
    "in": [rag_prompt_default_in, rag_prompt_history_default_in],
    "de": [rag_prompt_default_de, rag_prompt_history_default_de],
    "fr": [rag_prompt_default_fr, rag_prompt_history_default_fr],
    "it": [rag_prompt_default_it, rag_prompt_history_default_it],
    "multi": [rag_prompt_default_en, rag_prompt_history_default_en],
}

lang_dict = {
    "English": "en",
    "Vietnamese": "vi",
    "Spanish": "es",
    "Chinese": "zh",
    "Turkish": "tr",
    "Japanese": "ja",
    "Korean": "kr",
    "Indian": "in",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Multilingual": "multi",
}


def get_prompt(yaml_path, lang):
    config = load_yaml_config(yaml_path)
    sys_prompt = None
    if "system_prompt" in config:
        sys_prompt = config["system_prompt"]
    if lang == "default":
        if "default_language" in config:
            lang = config["default_language"]
            return prompt_dict[lang], sys_prompt
        else:
            return prompt_dict["en"], sys_prompt
    else:
        try:
            return prompt_dict[lang_dict[lang]], sys_prompt
        except KeyError:
            raise ValueError(f"Language '{lang}' is not supported.") from None
