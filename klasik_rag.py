import os
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chat_models import init_chat_model
from supabase.client import Client, create_client

# ENV yükle
load_dotenv()  

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

print("RAG Chatbot'a hoş geldiniz! Çıkmak için 'q' yazın.\n")
while True:
    user_question = input("Kullanıcı: ")
    if user_question.strip().lower() == "q":
        print("Görüşmek üzere!")
        break

    # RAG pipeline
    relevant_docs = vector_store.similarity_search(user_question, k=3)
    sources = []
    docs_content = []
    for doc in relevant_docs:
        src = doc.metadata.get("source", "Bilinmeyen kaynak")
        sources.append(f"{src}")
        docs_content.append(doc.page_content)

    context = "\n\n".join([f"Kaynak: {src}\n{content}" for src, content in zip(sources, docs_content)])
    full_prompt = f"""Aşağıda kullanıcının bir sorusu ve ilgili kaynaklardan alınan bilgiler var. 
                    Kullanıcının sorusunu sadece aşağıdaki kaynaklara dayanarak, kısa ve açık bir şekilde Türkçe cevapla ve kullandığın kaynak(lar)ı belirt.

                    Kullanıcı sorusu: 
                    {user_question}

                    İlgili belgeler:
                    {context}

                    Yanıt:
"""
    answer = llm.invoke(full_prompt)
    print(f"Chatbot: {answer}\n")
