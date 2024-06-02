import argparse
from datetime import datetime

from llama_index.core import (
    Settings, 
    get_response_synthesizer, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

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

from azure_file_storage import AzureFileStorage

def main():
    # Parse the command line arguments
    args = parse_arguments()
    file_storage = AzureFileStorage()

    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL)

    print('Loading index from storage...')
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIRECTORY)
    index = load_index_from_storage(storage_context)
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=4,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact", # refine, compact, tree_summarize
    )


    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    prior_prompt = "" 
    prior_prompt += LANGUAGE_PROMPT[args.lang] + '\n'
    prior_prompt += LENGTH_PROMPT[args.length_response] + '\n'
    prior_prompt += PURPOSE_PROMPT + '\n'
    prior_prompt += RESTRICT_PROMPT + '\n'


    conversation_content = []
    while True:
        query = input("\nEnter a query: ")

        if query == "/exit":
            break
        if query.strip() == "":
            continue

        full_query = query + prior_prompt
        response = query_engine.query(full_query)
        print(response.get_response())

        conversation_content.append({
            'query': query,
            'response': response.get_response()
        })

    file_content = ''
    for i, content in enumerate(conversation_content):
        file_content += str(i + 1) + '.' + content['query'] + '\n' + content['response'] + '\n\n'

    if (args.save):
        url = file_storage.upload(file_content, f'conversation-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")}.txt')
        print('\n\nThe conversation has been saved to the following URL:')
        print(url)
    

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
        default='long',
        help="The length of the response to the query. Choose from 'short', 'medium', 'long'.")
    parser.add_argument(
        '--save'
        '-S'
        type=bool,
        default=True,
        help='Save the conversation to a file and upload it to Azure Blob Storage.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
