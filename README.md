### Requirements

Ensure you have Ollama installed with `llama3` and `nomic-embed-text`. If not, you can install them by running the following commands:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

* Install Llama-Index
```bash
pip install llama-index-embeddings-huggingface install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-ollama llmsherpa
```

### Instructions
* Place your document files in the sources_document directory.
* Add the filenames to documents_info.py.
* Run the following command to process the documents:
```bash
python ingest.py
```
* To start the prompt, run:
```bash
python private-gpt
```

### Parameters
* `--lang`: Specify the language (`vi`: default , `en` for English).
* `--length-response`: Define the response length (`short`, `medium`, `long`: default).