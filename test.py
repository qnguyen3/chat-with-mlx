import gradio as gr
from mlx_lm import load, generate
from openai import OpenAI

openai_api_base = "http://127.0.0.1:8080/v1"

model, tokenizer = load("mlx-community/stablelm-2-zephyr-1_6b", tokenizer_config={'trust_remote_code':True})
chat_history = []
client = OpenAI(api_key='EMPTY',base_url=openai_api_base)

def chatbot(query, history):
    if len(history) > 0:
        for i, message in enumerate(history[0]):
            if i % 2 == 0:
                chat_content = {'role': 'user', 'content': message}
                chat_history.append(chat_content)
            else:
                chat_content = {'role': 'assistant', 'content': message}
                chat_history.append(chat_content)
    chat_history.append({'role': 'user', 'content': query})
    response = client.chat.completions.create(
        model='mlx-community/stablelm-2-zephyr-1_6b',
        messages=chat_history,
        temperature=0.2,
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
    slider = gr.Slider(10, 100, render=False)

    # Gradio UI inference function
    gr.ChatInterface(
        fn=chatbot,                        # Function to call on user input
        title="Chat with MLX",    # Title of the web page
        description="Chat with your data using Apple MLX Backend",    # Description
    )

demo.launch()    