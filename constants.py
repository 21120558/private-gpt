import os

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db_processes')
DOCUMENTS_FOLDER = os.environ.get('DOCUMENTS_FOLDER', 'source_documents')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL', 'nomic-embed-text')
LLM_MODEL = os.environ.get('MODEL', 'llama3')
