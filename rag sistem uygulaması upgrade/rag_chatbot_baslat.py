
import streamlit as st
import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("Chatbot")

#pinecone db'yi başlat.
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)

# gömme modeli ve vektör store'u başlat
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("Sen bir soru-cevap görevi için seçilmiş asistansın"))

# tekrar başlatıldığında geçmiş sohbet mesajlarını göster.
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# sorgu girilebilen sohbet barını oluştur.
prompt = st.chat_input("Selam, yönetmelikle alakalı sorularını sorabilirsin.")

# kullanıcı bir sorgu gönderdiğinde çalışır.
if prompt:

    # kullanıcı sorgusunu sohbette yazdır.
    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))

    # llm'i başlat
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1
    )

    # retriever'ı oluştur ve tetikle.
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # system prompt
    system_prompt = """Soru cevaplama görevleri için bir asistansın. 
    Soruyu yanıtlamak için aşağıdaki alınan context parçalarını kullan. 
    Eğer cevabı bilmiyorsan, sadece bulamadığını söyle. 
    En fazla üç cümle kullan ve cevabı kısa tut.
    Context: {context}:"""

    # retrieved context bilgisiyle system prompt'u birleştir.
    system_prompt_fmt = system_prompt.format(context=docs_text)


    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # system prompt'u mesaj geçmişine ekle.
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # llm'i tetikle
    result = llm.invoke(st.session_state.messages).content

    # llm'den gelen cevabı chat'e ekle.
    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))