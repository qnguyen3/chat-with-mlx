import gradio as gr
from mlx_lm import load, generate
from openai import OpenAI
import subprocess
from huggingface_hub import snapshot_download
from chat_with_mlx.models.utils import model_info
import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import YoutubeLoader

openai_api_base = "http://127.0.0.1:8080/v1"
model_dicts = model_info()
model_list = list(model_dicts.keys())
client = OpenAI(api_key='EMPTY',base_url=openai_api_base)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
emb = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1', model_kwargs={'trust_remote_code':True})

rag_prompt = """You are given a context from a document and your job is to answer a question from a user about that given context
---CONTEXT---
{doc_1}
{doc_2}
{doc_3}
---END---
Based on the given context and information. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
"""

rag_prompt_history = """
You are given a context from a document and a chat history between the user and you. Your job is to answer a question from a user about that given context and the chat history:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{doc_1}
{doc_2}
{doc_3}
---END---
Based on the given context, information and chat history. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
"""

# def load_model(model_name):
#     global process
#     model_name_list = model_name.split('/')
#     local_model_dir = os.path.join(os.getcwd(), 'chat_with_mlx', 'models', 'download', f'{model_name_list[1]}')
#     if os.path.exists(local_model_dir) == False:
#         snapshot_download(repo_id=model_dicts[f"{model_name}"], local_dir=local_model_dir)
#     command = [
#     "python", "-m", "mlx_lm.server",
#     "--model", local_model_dir
#     ]
#     # Execute the commands
#     try:
#         process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
#         process.stdin.write('y\n')
#         process.stdin.flush()
#         return {model_status: 'Model Loaded'}
#     except subprocess.CalledProcessError as e:
#         print(f"Error executing command: {e}")
#         return {model_status: e}

def load_model(model_name):
    global process
    model_name_list = model_name.split('/')
    local_model_dir = os.path.join(os.getcwd(), 'chat_with_mlx', 'models', 'download', model_name_list[1])
    
    if not os.path.exists(local_model_dir):
        snapshot_download(repo_id=model_dicts[model_name], local_dir=local_model_dir)
    
    command = [
        "python", "-m", "mlx_lm.server",
        "--model", local_model_dir
    ]
    
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        process.stdin.write('y\n')
        process.stdin.flush()
        
        # Wait for the process to complete, or use process.poll() for non-blocking check
        # returncode = process.poll()
        return {model_status: f"Model Loaded"}
    except Exception as e:
        return {model_status: f"Exception occurred: {str(e)}"}


def kill_process():
    process.terminate()
    time.sleep(2)
    if process.poll() is None:  # Check if the process has indeed terminated
        process.kill()  # Force kill if still running

    print("Model Killed")
    return {model_status: 'Model Unloaded'}

def check_file_type(file_path):
    # Check for document file extensions
    if file_path.endswith('.pdf') or file_path.endswith('.txt') or file_path.endswith('.doc') or file_path.endswith('.docx'):
        return True
    # Check for YouTube link formats
    elif file_path.startswith('https://www.youtube.com/') or file_path.startswith('https://youtube.com/') or file_path.startswith('https://youtu.be/'):
        return True
    else:
        return False


def upload(files):
    supported = check_file_type(files)
    if supported:
        return {url: files, index_status: 'Not Done'}
    else:
        return {url: 'File type not supported', index_status: 'Not Done'}
    
def indexing(mode, url):
    global vectorstore

    try:

        if mode == 'Files (docx, pdf, txt)':
            loader = PyPDFLoader(url)
            splits = loader.load_and_split(text_splitter)
        elif mode == 'YouTube (url)':
            loader = YoutubeLoader.from_youtube_url(url, 
                                                    add_video_info=False, language=['en', 'vi'])
            splits = loader.load_and_split(text_splitter)

        vectorstore = Chroma.from_documents(documents=splits, embedding=emb)
        return {index_status: 'Indexing Done'}
    except Exception as e:
        # Print the error message or return it as part of the response
        print(f"Error: {e}")  # This will print the error to the console or log
        return {'index_status': 'Indexing Error', 'error_message': str(e)}



def chatbot(query, history):
    global chat_history

    if len(history) == 0:
        chat_history = []
        chat_history.append({'role': 'user', 'content': query})
        docs = vectorstore.similarity_search(query)
    else:
        history_str = ''
        for i, message in enumerate(history[0]):
            if i % 2 == 0:
                chat_content = {'role': 'user', 'content': message}
                # chat_history.append(chat_content)
                history_str += f"User: {message}\n"
            else:
                chat_content = {'role': 'assistant', 'content': message}
                # chat_history.append(chat_content)
                history_str += f"AI: {message}\n"
        chat_history.append({'role': 'user', 'content': history_str})
        docs = vectorstore.similarity_search(history_str)
    
    doc_1 = docs[0].page_content
    doc_2 = docs[1].page_content
    doc_3 = docs[2].page_content
    if len(history) == 0:
        prompt = rag_prompt.format(doc_1=doc_1, doc_2=doc_2, doc_3=doc_3, question=query)
    else:
        prompt = rag_prompt_history.format(chat_history=history_str,doc_1=doc_1, doc_2=doc_2, doc_3=doc_3, question=query)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model='gpt',
        messages=messages,
        temperature=0.2,
        frequency_penalty=1.05,
        max_tokens=512,
        stream=True,
    )
    stop = ['<|im_end|>', '<|endoftext|>']
    partial_message = ""
    for chunk in response:
        if len(chunk.choices) != 0:
            if chunk.choices[0].delta.content not in stop:
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

    gr.ChatInterface(
        chatbot=gr.Chatbot(height=600,render=False),
        fn=chatbot,                        # Function to call on user input
        title="Chat with MLX",    # Title of the web page
        description="Chat with your data using Apple MLX Backend",    # Description
    )

    with gr.Row():
        with gr.Column(scale=2):
            model_name = gr.Dropdown(label='Model',info= 'Select your model', choices=model_list)
            btn1 = gr.Button("Load Model")
            btn3 = gr.Button("Unload Model")
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=9):
                    mode = gr.Dropdown(label='Dataset',info= 'Choose your dataset type', choices=['Files (docx, pdf, txt)', 'YouTube (url)'], scale=5)
                    url = gr.Textbox(label='URL', info='Enter your filepath (URL for Youtube)', interactive=True)
                    upload_button = gr.UploadButton(label='Upload File')
                    
                # data = gr.Textbox(visible=lambda mode: mode == 'YouTube')
                with gr.Column(scale=1):
                    model_status = gr.Textbox('Model Not Loaded', label='Model Status')
                    index_status = gr.Textbox("Not Index", label='Index Status')
                    btn1.click(load_model, inputs=[model_name], outputs=[model_status])
                    btn3.click(kill_process, outputs=[model_status])
                    upload_button.upload(upload, inputs=upload_button, outputs=[url, index_status])

                    index_button = gr.Button('Start Indexing')
                    index_button.click(indexing, inputs=[mode, url], outputs=[index_status])
    

demo.launch(inbrowser=True)