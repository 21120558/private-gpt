import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import (
    Settings, 
    get_response_synthesizer, 
    StorageContext, 
    load_index_from_storage,
    VectorStoreIndex
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.objects import ObjectIndex, SimpleObjectNodeMapping
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
import chromadb
import ollama


from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

from prompt_template import (
    LANGUAGE_PROMPT,
    LENGTH_PROMPT,
    PURPOSE_PROMPT,
    RESTRICT_PROMPT,
)

from constants import (
    EMBEDDINGS_MODEL, 
    LLM_MODEL, 
    PERSIST_DIRECTORY,
)
from documents_info import docs_info
from azure_file_storage import AzureFileStorage

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')

    parser.add_argument(
        "--lang", 
        '-L', 
        type=str, 
        default='vi',
        help="The language of the documents and the queries. Choose from 'en', 'vi'."
    )
    parser.add_argument(
        "--length-response", 
        '-LR', 
        type=str, 
        default='medium',
        help="The length of the response to the query. Choose from 'short', 'medium', 'long'.")
    parser.add_argument(
        '--save',
        '-S',
        type=bool,
        default=True,
        help='Save the conversation to a file and upload it to Azure Blob Storage.'
    )

    return parser.parse_args()

Settings.llm = Ollama(model=LLM_MODEL, request_timeout=500)
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL)
client_db = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

file_storage = AzureFileStorage()

def is_collection_exists(client, collection_name):
    collections = client.list_collections()
    return any(collection.name == collection_name for collection in collections)

tools = []
for doc in docs_info:
    collection_name = doc['collection']
    title = doc['title']

    if not is_collection_exists(client_db, collection_name):
        exit(1)

    chroma_collection = client_db.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

    tool = QueryEngineTool(
        query_engine=index.as_query_engine(),
        metadata=ToolMetadata(
            name=title,
            description=(
                "Useful for questions related to specific aspects of"
                f" {title}."
            ),
        ),
    )
    tools.append(tool)


my_objects = {
    str(hash(str(obj))): obj for i, obj in enumerate(tools)
}

def from_node_fn(node):
    return my_objects[node.id_]

def to_node_fn(obj):
    return TextNode(id_=str(hash(str(obj))), text=obj.metadata.name)

object_index = ObjectIndex.from_objects(
    tools,
    index_cls=VectorStoreIndex,
    from_node_fn=from_node_fn,
    to_node_fn=to_node_fn,
)
object_retriever = object_index.as_retriever(similarity_top_k=1)
args = parse_arguments()
prior_prompt = "" 
prior_prompt += LANGUAGE_PROMPT[args.lang] + '\n'
prior_prompt += LENGTH_PROMPT[args.length_response] + '\n'
prior_prompt += PURPOSE_PROMPT + '\n'
prior_prompt += RESTRICT_PROMPT + '\n'

       
def main():


    while True:
        print('\n\n')
        print('=' * 50)
        query = input("\nEnter a query: ")

        if query == "/exit":
            break
        if query.strip() == "":
            continue

        full_query = f'Question: {query}\n{prior_prompt}'
        objs = object_retriever.retrieve(query)

        tool_objs_retrieve = ObjectIndex.from_objects(
            objs,
            index_cls=VectorStoreIndex,
        )
        query_engine = ToolRetrieverRouterQueryEngine(tool_objs_retrieve.as_retriever())
        print(query_engine.query(full_query))

        if (args.save):
            title = objs[0].metadata.name
            file_name = next((doc['file_name'] for doc in docs_info if title == doc['title']), None)

            url = file_storage.upload(file_name)
            print(f'\n\nReferences: {url}')

@app.route('/generate', methods=['POST'])
def handle_query():
    data = request.json
    
    if data.get('type') == 'title':
        prompt = data.get('messages', '')
        message = prompt[len(prompt) - 1].get('content', '')
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': message + '\nResponse in Vietnamese',
            }
        ])

        return jsonify({
            'response': response['message']['content']
        })
    else:
        prompt = data.get('messages', '')
        message = prompt[len(prompt) - 1].get('content')

        objs = object_retriever.retrieve(message)
        tool_objs_retrieve = ObjectIndex.from_objects(
            objs,
            index_cls=VectorStoreIndex,
        )
        query_engine = ToolRetrieverRouterQueryEngine(tool_objs_retrieve.as_retriever())

        full_query = f'Question: {message}\n{prior_prompt}'
        response = query_engine.query(full_query)

        title = objs[0].metadata.name
        file_name = next((doc['file_name'] for doc in docs_info if title == doc['title']), None)
        try:
            url = file_storage.upload(file_name)
        except Exception as e:
            print(e)
            url = ''

        return jsonify({
            'response': response.response + '\n' + url,
        })
        
            


if __name__ == "__main__":
    # main()
    app.run(debug=True, host='0.0.0.0', port=5000)
