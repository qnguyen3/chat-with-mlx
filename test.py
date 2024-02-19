import gradio as gr
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/stablelm-2-zephyr-1_6b", tokenizer_config={'trust_remote_code':True})
chat_history = []

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
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=4096)
    return response

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