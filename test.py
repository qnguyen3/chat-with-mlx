import gradio as gr
from mlx_lm import load, generate
from openai import OpenAI
import subprocess
from huggingface_hub import snapshot_download
from chat_with_mlx.models.utils import model_info
import os
import time
import tqdm

openai_api_base = "http://127.0.0.1:8080/v1"
model_dicts = model_info()
model_list = list(model_dicts.keys())

def load_model(model_name):
    model_name_list = model_name.split('/')
    local_model_dir = os.path.join(os.getcwd(), 'chat_with_mlx', 'models', 'download', f'{model_name_list[1]}')
    if os.path.exists(local_model_dir) == False:
        snapshot_download(repo_id=model_dicts[f"{model_name}"], local_dir=local_model_dir)
    command = [
    "python", "-m", "mlx_lm.server",
    "--model", local_model_dir
    ]
    # Execute the commands
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
        process.stdin.write('y\n')
        process.stdin.flush()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

client = OpenAI(api_key='EMPTY',base_url=openai_api_base)

def chatbot(query, history):
    global chat_history

    if len(history) == 0:
        chat_history = []
    else:
        for i, message in enumerate(history[0]):
            if i % 2 == 0:
                chat_content = {'role': 'user', 'content': message}
                chat_history.append(chat_content)
            else:
                chat_content = {'role': 'assistant', 'content': message}
                chat_history.append(chat_content)
    chat_history.append({'role': 'user', 'content': query})
    response = client.chat.completions.create(
        model='gpt',
        messages=chat_history,
        temperature=0.2,
        frequency_penalty=1.05,
        max_tokens=512,
        stream=True,
    )
    stop = '<|endoftext|>'
    partial_message = ""
    for chunk in response:
        if len(chunk.choices) != 0:
            if chunk.choices[0].delta.content != stop:
                partial_message = partial_message + chunk.choices[0].delta.content
            else:
                partial_message = partial_message + ''
            yield partial_message

css = """
#component-5 {
  background: lightblue;
  color: #ffffff;
}
"""

with gr.Blocks(fill_height=True, css=css) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            model_name = gr.Dropdown(label='Model',info= 'Select your model', choices=model_list)
            btn1 = gr.Button("Load Model")
            btn1.click(load_model, inputs=[model_name])
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=5):
                    mode = gr.Dropdown(label='Dataset',info= 'Choose your dataset type', choices=['Files (docx, pdf, txt)', 'YouTube (url)'], scale=5)
                    url = gr.Textbox(label='URL', info='Enter your filepath (URL for Youtube)')
                # data = gr.Textbox(visible=lambda mode: mode == 'YouTube')
                btn2 = gr.Button("Start indexing", scale=1)

    # Gradio UI inference function
    gr.ChatInterface(
        chatbot=gr.Chatbot(render=False),
        fn=chatbot,                        # Function to call on user input
        title="Chat with MLX",    # Title of the web page
        description="Chat with your data using Apple MLX Backend",    # Description
    )

demo.launch()    