from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
loader = PyPDFLoader("test_doc.pdf")

emb = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs={'trust_remote_code':True}, multi_process=False)
lol = emb.embed_query('hello')
print(lol)