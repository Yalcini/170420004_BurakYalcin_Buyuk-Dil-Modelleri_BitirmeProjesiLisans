# import basics
import os
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import OllamaEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

#supabase db yi başlatır
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# gömme modelini başlatır
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

#documents klasöründeki pdf dosyalarını yükler
loader = PyPDFDirectoryLoader("documents")

# belgeleri chunk'lara böler.
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# chunkları vektör store'da saklar.
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)