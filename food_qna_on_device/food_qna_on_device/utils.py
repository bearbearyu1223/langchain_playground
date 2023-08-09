from config import MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, RETURN_SOURCE_DOCUMENTS, EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """Use the following pieces of context to answer the question in shorter sentences. If you don't know the answer, just say that you don't know.

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
                            'temperature': TEMPERATURE
                        }
    )
    return llm


def _set_up_retrieval_qa(llm, prompt, vector_db):
    qa_client = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vector_db.as_retriever(search_kwargs={'k': TOP_K}),
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_client

def set_up_on_device_qa(): 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': 'cpu'})
    db_dir = os.path.join(os.getcwd(),VECTOR_DB_PATH)
    vector_db = FAISS.load_local(db_dir, embeddings)
    llm = _load_on_device_llm()
    qa_client = _set_up_retrieval_qa(llm=llm, prompt=PROMPT, vector_db=vector_db)
    return qa_client



