import argparse

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

def main():
    # Parse the command line arguments
    args = parse_arguments()

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
        streaming=True # print response as it is generated
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


    while True:
        try:
            query = input("\nEnter a query: ").encode('utf-8').decode('utf-8')
        except UnicodeDecodeError as e:
            print(f"Lỗi giải mã Unicode: {e}")
        except UnicodeEncodeError as e:
            print(f"Lỗi mã hóa Unicode: {e}")

        if query == "exit":
            break
        if query.strip() == "":
            continue

        full_query = query + prior_prompt
        response = query_engine.query(full_query)
        response.print_response_stream()        

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

    return parser.parse_args()


if __name__ == "__main__":
    main()
