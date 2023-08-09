import argparse
from build_knowledge_base import build_vector_db
from config import RETURN_SOURCE_DOCUMENTS
from food_qna_on_device.utils import _load_on_device_llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import timeit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """Use the following pieces of context to answer the question in a fewer sentences. If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



if __name__ == "__main__": 

    vector_db = build_vector_db()
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='How to make creamy squash soup?',
                        help='Enter a query related to food preparation and cooking.')
    args = parser.parse_args()

    query = args.input

    start = timeit.default_timer()
    llm = _load_on_device_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={'k': 1}), 
    return_source_documents=RETURN_SOURCE_DOCUMENTS, chain_type_kwargs={'prompt': PROMPT})
    response = qa({"query": query})
    end = timeit.default_timer()
    print('='*60)
    print("Time to produce response: {}".format(end - start))
    print("Answer:{}".format(response["result"]))
    print('='*60)
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print("Source Document: {}".format(i+1))
        print("Source Text: {}".format(doc.page_content))
        print("Document Name: {}".format(doc.metadata["source"]))
        print("Page Number: {}".format(doc.metadata["page"]))
        print('='* 60)