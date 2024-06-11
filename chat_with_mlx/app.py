import argparse
import os
import re
import time
from typing import Iterable, List, Tuple
import json
import yaml
import signal

from mlx_lm import load, stream_generate, generate

import gradio as gr
from huggingface_hub import snapshot_download
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    YoutubeLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

from chat_with_mlx import __version__
from chat_with_mlx.models.utils import model_info, flags, recommended_usage
from chat_with_mlx.rag.utils import get_prompt, lang_dict

css = """
.message-row.bubble.user-row.svelte-gutj6d {
  background-color: #007aff;
  color: #fff;
  border-radius: 50px;
  max-width: 80%;
  align-self: flex-end;
}

.message.user.svelte-gutj6d.message-fit.message-bubble-border {
  background-color: #007aff;
  padding: 5px 15px 5px 20px;
  border: 1px solid #007aff;
  border-radius: 35px 35px 0px 35px;
}

.message-row.bubble.user-row.svelte-gutj6d .md.svelte-1k4ye9u.chatbot.prose p {
  color: #fff;
  font-size: 14px;
}

.lg.primary.svelte-cmf5ev {
  background-color: #007aff;
  color: #fff;
}

#component-34 {
  font-size: 16px;
}

.lg.stop.red-btn.svelte-cmf5ev {
  background-color: #eb4034;
  color: #fff;
}

.message-row.bubble.bot-row.svelte-gutj6d {
  color: #fff;
  border-radius: 50px;
  max-width: 80%;
  align-self: flex-start;
}

.message.bot.svelte-gutj6d.message-fit.message-bubble-border {
  background-color: #e9e9eb;
  padding: 5px 15px 5px 20px;
  border: 1px solid #e9e9eb;
  border-radius: 35px 35px 35px 0px;
}

.message-row.bubble.bot-row.svelte-gutj6d .md.svelte-1k4ye9u.chatbot.prose p {
  color: #000000;
  font-size: 14px;
}
"""


os.environ["TOKENIZERS_PARALLELISM"] = "False"
SUPPORTED_LANG = [
    "default",
    "English",
    "Spanish",
    "Chinese",
    "Vietnamese",
    "Japanese",
    "Korean",
    "Indian",
    "Turkish",
    "German",
    "French",
    "Italian",
]
model_dicts, yml_path, cfg_list, mlx_config = model_info()
model_list = list(cfg_list.keys())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
emb = None
vectorstore = None
model_load_status = False
prev_model = ""

class GusStyle(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.indigo, #colors.Color(c50="#e0f2ff",c100="#b3e5fc",c200="#81d4fa",c300="#4fc3f7",c400="#29b6f6",c500="#03a9f4",c600="#039be5",c700="#0288d1",c800="#0277bd",c900="#01579b",c950="#004f73",),
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter Tight"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

def terminate_process():
    os.kill(os.getpid(), signal.SIGTERM)

def load_model(model_name, lang):
    global process, rag_prompt, rag_his_prompt, sys_prompt, default_lang, model_load_status, local_model_dir, directory_path, cfg_list, prev_model
    default_lang = "default"
    prompts, sys_prompt = get_prompt(f"{yml_path[cfg_list[model_name]]}", lang)
    rag_prompt, rag_his_prompt = prompts[0], prompts[1]
    model_name_list = cfg_list[model_name].split("/")
    directory_path = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(
        directory_path, "models", "download", model_name_list[1]
    )
    if not os.path.exists(local_model_dir):
        gr.Info(f'Model {model_name} is downloading. Please check terminal to see the progress.')
        snapshot_download(repo_id=mlx_config[model_name], local_dir=local_model_dir)

    global model, tokenizer
    with open(f"{local_model_dir}/tokenizer_config.json", "r") as f:
        tokenizer_config_ = json.load(f)

    # command = ["python3", "-m", "mlx_lm.server", "--model", local_model_dir]
    if model_load_status is True:
        del model, tokenizer
        print(f"Model {prev_model} Killed")
        time.sleep(2)
    
    # print(f"EOS: {tokenizer_config_['eos_token']}")
    if 'phi-3' in model_name.lower():
        tokenizer_config = {"trust_remote_code": True, "eos_token": "<|end|>"}
    else:
        tokenizer_config = {"trust_remote_code": True, "eos_token": tokenizer_config_['eos_token']}
    try:
        model, tokenizer = load(local_model_dir, tokenizer_config)
        model_load_status = True
        prev_model = model_name
        return f"Model {model_name} Loaded", sys_prompt
    except Exception as e:
        return "Exception occurred: {str(e)}"
    
# def remove_model(model_name, models_list):

#     global yml_path

#     model_name_list = cfg_list[model_name].split("/")
#     directory_path = os.path.dirname(os.path.abspath(__file__))
#     local_model_dir = os.path.join(
#         directory_path, "models", "download", model_name_list[1]
#     )
#     gr.Info(f'Model {model_name} remove!')
#     models_list.remove(model_name)

#     available_list = [[model] for model in sorted(models_list)]
#     config_dir = os.path.join(
#         directory_path, "models", "configs", model_name_list[1]
#     )
#     # print(model_name)
#     if '4bit' in model_name:
#         model_key = model_name_list[0] + '/' + model_name_list[1] + '-4bit'
#     elif '8bit' in model_name:
#         model_key = model_name_list[0] + '/' + model_name_list[1] + '-8bit'
#     else:
#         model_key = model_name_list[0] + '/' + model_name_list[1]
#     yaml_path = yml_path[model_key]
#     # if os.path.exists(yaml_path):
#     #     os.remove(yaml_path)
#     if os.path.exists(local_model_dir):
#         shutil.rmtree(local_model_dir)

#     return gr.Dropdown(choices=models_list, label='Choose model to delete'), available_list
    


def check_file_type(file_path):
    # Check for document file extensions
    if (
        file_path.endswith(".pdf")
        or file_path.endswith(".txt")
        or file_path.endswith(".doc")
        or file_path.endswith(".docx")
    ):
        return True
        # Check for YouTube link formats
    elif (
        file_path.startswith("https://www.youtube.com/")
        or file_path.startswith("https://youtube.com/")
        or file_path.startswith("https://youtu.be/")
    ):
        return True
    else:
        return False


def upload(files):
    supported = check_file_type(files)
    if supported:
        return {url: files, index_status: "Not Done"}
    else:
        return {url: "File type not supported", index_status: "Not Done"}


def indexing(mode, url):
    global vectorstore, emb

    if emb is None:
        gr.Info("Loading embedding model (first time may take longer, check terminal to check download progress)")
        emb = HuggingFaceEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-m",
            model_kwargs={"trust_remote_code": True},
            )

    try:
        if mode == "Files (docx, pdf, txt)":
            if url.endswith(".pdf"):
                loader = PyPDFLoader(url)
            elif url.endswith(".docx"):
                loader = Docx2txtLoader(url)
            elif url.endswith(".txt"):
                loader = TextLoader(url)
            splits = loader.load_and_split(text_splitter)
        elif mode == "YouTube (url)":
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=False, language=["en", "vi"]
            )
            splits = loader.load_and_split(text_splitter)

        vectorstore = Chroma.from_documents(documents=splits, embedding=emb)
        return {index_status: "Indexing Done"}
    except Exception as e:
        # Print the error message or return it as part of the response
        print(f"Error: {e}")  # This will print the error to the console or log
        return {"index_status": "Indexing Error", "error_message": str(e)}


def kill_index():
    global vectorstore
    vectorstore = None
    return {index_status: "Indexing Undone"}


def build_rag_context(docs):
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"

    return context


def chatbot(query, history, temp, max_tokens, repetition_penalty, k_docs, top_p, system_prompt, model_name):
    global chat_history, sys_prompt, local_model_dir, model, tokenizer

    if "vectorstore" in globals() and vectorstore is not None:
        if len(history) == 0:
            chat_history = []
            if system_prompt != '':
                chat_history.append({"role": "system", "content": system_prompt})
            elif system_prompt == '' and sys_prompt is not None:
                chat_history.append({"role": "system", "content": sys_prompt})
            docs = vectorstore.similarity_search(query, k=k_docs)
        else:
            history_str = ""
            for i, message in enumerate(history):
                history_str += f"User: {message[0]}\n"
                history_str += f"AI: {message[1]}\n"

            if system_prompt != '':
                chat_history.append({"role": "system", "content": system_prompt})
            elif system_prompt == '' and sys_prompt is not None:
                chat_history.append({"role": "system", "content": sys_prompt})
            chat_history.append({"role": "user", "content": history_str})
            docs = vectorstore.similarity_search(history_str)

        context = build_rag_context(docs)

        if len(history) == 0:
            prompt = rag_prompt.format(context=context, question=query)
        else:
            prompt = rag_his_prompt.format(
                chat_history=history_str, context=context, question=query
            )
        messages = [{"role": "user", "content": prompt}]
    else:
        if len(history) == 0:
            chat_history = []
            if system_prompt != '':
                chat_history.append({"role": "system", "content": system_prompt})
            elif system_prompt == '' and sys_prompt is not None:
                chat_history.append({"role": "system", "content": sys_prompt})
        else:
            chat_history = []
            if system_prompt != '':
                chat_history.append({"role": "system", "content": system_prompt})
            elif system_prompt == '' and sys_prompt is not None:
                chat_history.append({"role": "system", "content": sys_prompt})
            for i, message in enumerate(history):
                chat_history.append({"role": "user", "content": message[0]})
                chat_history.append({"role": "assistant", "content": message[1]})
        chat_history.append({"role": "user", "content": query})
        messages = chat_history
    if messages[0]['role'] == 'system' and messages[0]['content'] == '':
        messages = messages[1:]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Uncomment for debugging
    # print(messages)

    if 'phi-3-mini' in model_name.lower() or 'phi-3-small' in model_name.lower():
        response = stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)
        stop = ['<|end|>', tokenizer.eos_token]
    else:
        response = stream_generate(model, tokenizer, prompt, temp=temp, max_tokens=max_tokens, repetition_penalty=repetition_penalty, top_p=top_p,)
        stop = [tokenizer.eos_token]
    
    partial_message = ""
    for chunk in response:
        if chunk not in stop:
            partial_message = partial_message + chunk
            yield partial_message



def completions(prompt, temp, max_tokens, repetition_penalty, top_p, stop_sequence, model_name):
    global model, tokenizer
    partial_message = prompt
    prompt_tokens_len = len(tokenizer.encode(prompt))
    stops = [stop.strip() for stop in stop_sequence.split(',')]

    if 'phi-3-mini' in model_name.lower():
        partial_message = partial_message + ' '
        response = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
        response_split = re.findall(r'\S+|\s|[^\w\s]', response)
        output_tokens_cnt = len(tokenizer.encode(response))
        output_tokens = f'Input Tokens: {prompt_tokens_len}, Output Tokens: {output_tokens_cnt}, Total Tokens: {prompt_tokens_len + output_tokens_cnt}'
        new_msg = ''
        for word in response_split:
            new_msg = new_msg + word
            partial_message = partial_message + word
            stop_found = False 
            for stop in stops:
                if stop in new_msg:
                    stop_found = True
                    break
            if stops != [''] and stop_found:
                yield partial_message, output_tokens
                break

            yield partial_message, output_tokens
    else:
        response = stream_generate(model, tokenizer, prompt, temp=temp, max_tokens=max_tokens, repetition_penalty=repetition_penalty, top_p=top_p,)
        output_tokens_cnt = 0
        new_msg = ""

        for chunk in response:
            if chunk == tokenizer.eos_token:
                break
            else:
                output_tokens_cnt += 1
                new_msg = new_msg + chunk
                partial_message = partial_message + chunk
                output_tokens = f'Input Tokens: {prompt_tokens_len}, Output Tokens: {output_tokens_cnt}, Total Tokens: {prompt_tokens_len + output_tokens_cnt}'
                # Check for stop sequences
                stop_found = False
                for stop in stops:
                    if stop in new_msg:
                        stop_found = True
                        break
                
                if stops != [''] and stop_found:
                    yield partial_message, output_tokens
                    break

                
            yield partial_message, output_tokens




def add_model(original_repo, mlx_repo, quantization, lang, system_prompt, available_models):

    global local_model_dir, model_dicts, yml_path, cfg_list, mlx_config

    directory_path = os.path.dirname(os.path.abspath(__file__))

    local_config_dir = os.path.join(
        directory_path, "models", "configs"
    )
    repo_name = original_repo.split("/")[0]
    model_name = original_repo.split("/")[-1]
    if quantization != 'none':
        new_model_dict = {
            'original_repo': original_repo,
            'mlx-repo': mlx_repo,
            'quantize': quantization,
            'default_language': lang_dict[lang],
            'system_prompt': system_prompt
        }
        model_name_str = f"{repo_name}/{model_name}-{flags[lang_dict[lang]],quantization}"
        model_configs_name = os.path.join(local_config_dir, f"{model_name}-{quantization}.yaml")
    else:
        new_model_dict = {
            'original_repo': original_repo,
            'mlx-repo': mlx_repo,
            'default_language': lang_dict[lang],
            'system_prompt': system_prompt
        }
        if new_model_dict['system_prompt'] == '':
            del new_model_dict['system_prompt']
        model_name_str = f"{model_name}-{flags[lang_dict[lang]]}"
        model_configs_name = os.path.join(local_config_dir, f"{model_name}.yaml")
    
    if os.path.exists(model_configs_name):
        gr.Warning("Model is already existed.")
        return f"Model is already existed. Please add an another model.", sorted(available_models)

    
    available_models.append([model_name_str])
    
    with open(model_configs_name, 'w') as file:
        yaml.dump(new_model_dict, file, allow_unicode=True, sort_keys=False)
    model_dicts, yml_path, cfg_list, mlx_config = model_info()
    print(f"YAML file generated successfully for {original_repo}.")
    gr.Info("Please restart the program to use the new model.")

    return f"Status: Model **{original_repo}** added. Please restart the program to see the added model.", sorted(available_models)

def reset_sys_prompt():
    global sys_prompt
    if sys_prompt is not None:
        return sys_prompt
    else:
        return ''

with gr.Blocks(fill_height=True, theme=GusStyle(), css=css) as demo:
    title = gr.HTML("<h1>🍎 Chat with MLX </h1>")
    with gr.Tab("Chat"):
        with gr.Row():
            system_prompt = gr.Textbox(placeholder="System prompt (blank will use model's default system prompt)", label="System Prompt", interactive=True, render=False, scale=9)
            with gr.Column(scale=1):
                with gr.Row():
                    config_text = gr.Markdown('### Configuration', show_label=False)
                
                    
                ## SELECT MODEL
                model_name = gr.Dropdown(
                    label="Select Model",
                    choices=sorted(model_list),
                    interactive=True,
                    render=False,
                )
                model_name.render()
                language = gr.Dropdown(
                    label="Language",
                    choices=sorted(SUPPORTED_LANG),
                    value="default",
                    interactive=True,
                )
                model_status = gr.Textbox("Model Not Loaded", label="Model Status")
                btn1 = gr.Button("Load Model", variant="primary")
                
                # btn3 = gr.Button("Unload Model", variant="stop")
                with gr.Accordion("RAG Setting", open=False):
                    # FILE
                    mode = gr.Dropdown(
                        label="Dataset Type",
                        info="Choose your dataset type",
                        choices=["Files (docx, pdf, txt)", "YouTube (url)"],
                        scale=5,
                    )
                    url = gr.Textbox(
                        label="URL",
                        placeholder="Enter your filepath (URL for Youtube)",
                        interactive=True,
                    )
                    upload_button = gr.UploadButton(
                        label="Upload File", variant="secondary",
                    )
                    # MODEL STATUS
                    # data = gr.Textbox(visible=lambda mode: mode == 'YouTube')
                    
                    index_status = gr.Textbox("Not Index", label="Index Status")
                    btn1.click(
                        load_model,
                        inputs=[model_name, language],
                        outputs=[model_status, system_prompt],
                    )
                    # btn3.click(kill_process, outputs=[model_status])
                    upload_button.upload(
                        upload, inputs=upload_button, outputs=[url, index_status]
                    )

                    index_button = gr.Button("Start Indexing", variant="primary")
                    index_button.click(
                        indexing, inputs=[mode, url], outputs=[index_status]
                    )
                    stop_index_button = gr.Button("Stop Indexing")
                    stop_index_button.click(kill_index, outputs=[index_status])
                    retrieve_docs = gr.Slider(
                                label="No. Retrieval Docs",
                                value=3,
                                minimum=1,
                                maximum=10,
                                step=1,
                                interactive=True,
                            )


                with gr.Accordion("Advanced Setting", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            
                            temp_slider = gr.Slider(
                                label="Temperature",
                                value=0.2,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                interactive=True,
                            )

                            max_gen_token = gr.Slider(
                                label="Max Tokens",
                                value=512,
                                minimum=128,
                                maximum=32768,
                                step=128,
                                interactive=True,
                            )

                            rep_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.05,
                                minimum=1.05,
                                maximum=5,
                                step=0.05,
                                interactive=True,
                            )

                            top_p = gr.Slider(
                                label="top-p",
                                value=0.9,
                                minimum=0.1,
                                maximum=0.99,
                                step=0.01,
                                interactive=True,
                            )
            with gr.Column(scale=4):
                with gr.Row():
                    system_prompt.render()
                    default_prompt_btn = gr.Button("Default Prompt", variant="secondary", scale=1)
                    default_prompt_btn.click(reset_sys_prompt, outputs=[system_prompt])

                gr.ChatInterface(
                    chatbot=gr.Chatbot(height=562, render=False, layout="bubble", bubble_full_width=False),
                    fn=chatbot,  # Function to call on user input
                    title=None,  # Title of the web page
                    submit_btn='↑',
                    retry_btn='Retry',
                    undo_btn='Undo',
                    clear_btn='Clear',
                    additional_inputs=[temp_slider, max_gen_token, rep_penalty, retrieve_docs, top_p, system_prompt, model_name],
                ) 
    with gr.Tab("Completion"):
        with gr.Row():
            temp_slider_completion = gr.State(0.2)
            max_gen_token_completion = gr.State(512)
            rep_penalty_completion = gr.State(1.05)
            top_p_completion = gr.State(0.9)
            language = gr.State("default")
            playground = gr.Textbox(placeholder="Enter your text...",interactive=True, lines=31, scale=4, show_label=False, render=False)
            total_token_completion = gr.Markdown("Input Tokens: 0, Output Tokens: 0, Total Tokens: 0", label="Total Token", show_label=False, render=False)
            with gr.Column(scale=1):
                with gr.Row():
                    config_text = gr.Markdown('### Configuration', show_label=False)
                
                    
                ## SELECT MODEL
                model_name_com = gr.Dropdown(
                    label="Select Model",
                    choices=sorted(model_list),
                    interactive=True,
                    render=False,
                )
                model_name_com.render()
                model_status_com = gr.Textbox("Model Not Loaded", label="Model Status")
                btn1 = gr.Button("Load Model", variant="primary")
                btn1.click(
                    load_model,
                    inputs=[model_name_com, language],
                    outputs=[model_status_com, system_prompt],
                )

                stop_sequence_completion = gr.Textbox(
                    placeholder="Stop sequence...seperate by ,",
                    label='Stop Sequences', 
                    interactive=True, 
                    lines=1, 
                    show_label=False
                )

                temp_slider_completion = gr.Slider(
                    label="Temperature",
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    interactive=True,
                )
                max_gen_token_completion = gr.Slider(
                    label="Max Tokens",
                    value=512,
                    minimum=128,
                    maximum=32768,
                    step=128,
                    interactive=True,
                )
                rep_penalty_completion = gr.Slider(
                    label="Repetition Penalty",
                    value=1.05,
                    minimum=1.05,
                    maximum=5,
                    step=0.05,
                    interactive=True,
                )
                top_p_completion = gr.Slider(
                    label="top-p",
                    value=0.9,
                    minimum=0.1,
                    maximum=0.99,
                    step=0.01,
                    interactive=True,
                )

                submit_completion = gr.Button("Submit", variant="primary")
                submit_completion.click(completions, inputs=[playground, temp_slider_completion, max_gen_token_completion, rep_penalty_completion, top_p_completion, stop_sequence_completion, model_name_com], outputs=[playground, total_token_completion])
                clear_button = gr.Button("Clear Text", variant="secondary")
                clear_button.click(lambda : "", outputs=[playground])
            with gr.Column(scale=5):
                playground.render()
                total_token_completion.render()
    with gr.Tab("Model Manager"):
        with gr.Row():
            
            with gr.Column(scale=3):
                sorted_models = sorted(model_list)
                model_choices = gr.State(sorted_models)
                available_list = [[model] for model in sorted(model_list)]
                available_list_state = gr.State(available_list)
                available_models = gr.DataFrame(headers=['Available Models'], value=available_list, scale=3, show_label=False, height=500)
            with gr.Column(scale=2):
                support_lang_list = sorted(SUPPORTED_LANG)[:-1]
                support_lang_list.append('Multilingual')
                gr.Markdown('## Add new model')
                new_model_repo = gr.Textbox(info='Original Repo', show_label=False, placeholder="Original Repo (i.e teknium/OpenHermes-2.5)", interactive=True)
                new_mlx_model_repo = gr.Textbox(info='MLX Repo', show_label=False, placeholder="MLX Repo (i.e mlx-community/OpenHermes-2.5-Mistral-7B)", interactive=True)
                with gr.Row():
                    quantization_mode = gr.Dropdown(label='Quantization Mode', choices=['none','4bit','8bit'], interactive=True, value='none')
                    language_choices = gr.Dropdown(label="Default Language", choices=support_lang_list, value="English", interactive=True)
                default_sys_prompt = gr.Textbox(info="Default System Prompt", show_label=False, placeholder="ex: You are a helpful assistant.", interactive=True)
                with gr.Row():
                    add_model_button = gr.Button("Add Model", variant="primary", scale=5)
                    quit_button = gr.Button("Quit", variant='stop', scale=2, elem_classes='red-btn')
                    
                add_model_status = gr.Markdown("Status: None")
                add_model_button.click(add_model, inputs=[new_model_repo, new_mlx_model_repo, quantization_mode, language_choices, default_sys_prompt, available_list_state], outputs=[add_model_status, available_models])
                quit_button.click(terminate_process, outputs=[add_model_status])
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown('### Recommended Usage')
                size_md = gr.Markdown(recommended_usage)
            # with gr.Column(scale=2):
            #     sorted_models = sorted(model_list)
            #     sorted_models_rm = gr.State(sorted_models)
            #     del_md = gr.Markdown('### Delete Model')
            #     avail_model = gr.Dropdown(choices=sorted_models, label='Choose model to delete')
            #     delete_model_btn = gr.Button("Delete Model", variant="stop", elem_classes='red-btn')
            # delete_model_btn.click(remove_model, inputs=[avail_model, sorted_models_rm], outputs=[avail_model, available_models])

def app(port, share):
    print(f"Starting MLX Chat on port {port}")
    print(f"Sharing: {share}")
    demo.launch(inbrowser=True, share=share, server_port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with MLX \n"
        "Native RAG on MacOS and Apple Silicon with MLX 🧑‍💻"
    )
    parser.add_argument(
        "--version", action="version", version=f"Chat with MLX {__version__}"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the app",
    )
    parser.add_argument(
        "--share",
        default=False,
        help="Enable sharing the app",
    )
    args = parser.parse_args()
    app(port=args.port, share=args.share)