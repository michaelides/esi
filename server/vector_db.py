from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from pypdf import PdfReader
import io
from config import settings

class VectorDB:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = None

    def add_pdf(self, pdf_file):
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = self.text_splitter.split_text(text)
        if self.vector_store is None:
            self.vector_store = DocArrayInMemorySearch.from_texts(documents, self.embeddings)
        else:
            self.vector_store.add_texts(documents)

    def search(self, query):
        if self.vector_store is None:
            return "Vector store is not initialized. Please upload a PDF first."

        docs = self.vector_store.similarity_search(query)
        return "\n".join([doc.page_content for doc in docs])

# Make a single instance available
_vector_db_instance = None

def get_vector_db():
    global _vector_db_instance
    if _vector_db_instance is None:
        if settings.GOOGLE_API_KEY:
            try:
                _vector_db_instance = VectorDB()
            except Exception as e:
                print(f"Failed to initialize VectorDB: {e}")
                # Return a dummy object if initialization fails
                class DummyVectorDB:
                    def search(self, query):
                        return f"Vector store initialization failed: {e}"
                    def add_pdf(self, pdf_file):
                        return f"Vector store initialization failed: {e}"
                return DummyVectorDB()
        else:
            # Return a dummy object if no API key
            class DummyVectorDB:
                def search(self, query):
                    return "Vector store is not configured. Please provide a GOOGLE_API_KEY."
                def add_pdf(self, pdf_file):
                    return "Vector store is not configured. Please provide a GOOGLE_API_KEY."
            return DummyVectorDB()
    return _vector_db_instance
