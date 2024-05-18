#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "llama3")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 16))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(
        model=model,
        callbacks=callbacks,
        temperature=0.2,       
    )
    

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    prompt_prefix = """
    Lưu ý quan trọng: Từ bây giờ, bạn phải trả lời tất cả câu hỏi bằng tiếng Việt. Không được sử dụng bất kỳ ngôn ngữ nào khác ngoại trừ tiếng Việt. Dù câu hỏi có thể được đưa ra bằng ngôn ngữ khác, bạn vẫn phải trả lời bằng tiếng Việt. 
    Tất cả câu trả lời phải rõ ràng, chính xác và bằng tiếng Việt. Không được sử dụng từ ngữ nào không phải tiếng Việt.

    Nhắc lại: Bạn phải trả lời bằng tiếng Việt, bất kể câu hỏi bằng ngôn ngữ nào. Không được sử dụng ngôn ngữ khác.

    Đây là câu hỏi của tôi:
    """


    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        full_query = prompt_prefix + query

        start = time.time()
        res = qa(full_query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
