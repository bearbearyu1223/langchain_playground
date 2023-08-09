from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def build_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': 'cpu'})
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print("Total number of Indexed Document: {}".format(vector_db.index.ntotal))
    return vector_db

if __name__ == '__main__':
    vector_db = build_vector_db()
    query = "How to make creamy squash soup?"
    search_results = vector_db.similarity_search_with_relevance_scores(query)
    print("Semantic Search Results: {}".format(search_results[0]))