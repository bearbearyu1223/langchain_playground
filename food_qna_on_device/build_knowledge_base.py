from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

def build_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    chuncked_documents = text_splitter.split_documents(raw_documents)
    for document in chuncked_documents:
        texts.append(document.page_content)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': 'cpu'})
    vector_db = FAISS.from_texts(texts, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print("Total number of Indexed Document: {}".format(vector_db.index.ntotal))
    return vector_db

if __name__ == '__main__':
    vector_db = build_vector_db()
    query = "How to make GROUND BEEF STROGANOFF?"
    search_results = vector_db.similarity_search_with_relevance_scores(query)
    for result in search_results:
        print('='* 60)
        print("Semantic Search Results with score {}\n: {}".format(result[1], result[0].page_content))