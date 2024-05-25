from llama_index.core import (
    VectorStoreIndex, 
    Settings, 
    Document
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from constants import (
    DOCUMENTS_FOLDER,
    PERSIST_DIRECTORY,
    LLM_MODEL,
    EMBEDDINGS_MODEL,
)
from documents_info import processes_info

from utils import pdf_to_text

# configure LLM and embeddings
llm = Ollama(model=LLM_MODEL, request_timeout=360.0, temperature=0.4)
embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL)

# configure settings for the vector store
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.embed_model = embed_model
Settings.llm = llm

def process_documents():
    # convert pdf to text 
    docs = []
    for process_info in processes_info:
        file_name = process_info['file_name']
        print('Processing:', file_name)
        pdf_url = f"{DOCUMENTS_FOLDER}/{file_name}" 
        doc = pdf_to_text(pdf_url)
        docs.append(doc)



    index = VectorStoreIndex.from_documents([])
    for doc in docs:
        for page in doc:
            index.insert(Document(text=page))

    index.storage_context.persist(persist_dir=PERSIST_DIRECTORY)
    print('Documents processed successfully!')

if __name__ == "__main__":
    process_documents()
