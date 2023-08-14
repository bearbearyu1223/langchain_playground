
from build_knowledge_base import build_vector_db
from config import MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, GPU_LAYERS, CONTEXT_LENGTH,  TEMPERATURE, TOP_K, RETURN_SOURCE_DOCUMENTS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
import os
import argparse
import timeit


os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """Use the retrieved content to answer the question. If you cannot find answers, just say that you don't know.

Context: {context}
Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def _load_on_device_llm():
    model_dir = os.path.join(os.getcwd(),MODEL_BIN_PATH)
    llm = CTransformers(model=model_dir, 
                        model_type=MODEL_TYPE, 
                        config={
                            'max_new_tokens': MAX_NEW_TOKENS, 
                            'context_length': CONTEXT_LENGTH, 
                            'temperature': TEMPERATURE,
                            'gpu_layers': GPU_LAYERS,
                            'stream': True,
                        }
    )
    return llm


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("config",
                        type=str,
                        help="config to set running model inference on device or on server ",
                        choices=["local", "server"],
                        nargs="?", 
                        default="server")
    parser.add_argument('-c', '--config', action='store_true', dest="config")
    args = parser.parse_args()
    start = timeit.default_timer()
    vector_db = build_vector_db()
    end = timeit.default_timer()
    print('='*60)
    print("Time to build the VectorDB: {}".format(end - start))

    if args.config.lower() == 'local': 
        print('='*60)
        print('Set model inference to run on local device ...')
        llm = _load_on_device_llm()
    else:
        print('Set model inference to run on server with OpenAI ...')
        llm=ChatOpenAI(temperature=TEMPERATURE)
        
    while True:
        try:
            query = input('Enter a query related to food preparation and cooking: ')
            start = timeit.default_timer()
        
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={'k': TOP_K}), 
            return_source_documents=RETURN_SOURCE_DOCUMENTS, chain_type_kwargs={'prompt': PROMPT})
            response = qa({"query": query})
            end = timeit.default_timer()
            print('='*60)
            print("Time to produce response: {}".format(end - start))
            print("Answer:{}".format(response["result"]))
            print('='*60)
            # source_docs = response['source_documents']
            # for i, doc in enumerate(source_docs):
            #     print("Source Document: {}".format(i+1))
            #     print("Source Text: {}".format(doc.page_content))
            #     print('='* 60)
        except KeyboardInterrupt:  # Ctrl + C - will exit program immediately if not caught
            break
    print()
    print("Program Exit")