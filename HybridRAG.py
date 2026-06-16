import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever




def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def store_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        cache_folder=".cache/huggingface/hub"
    )
    vectorstore = Chroma.from_texts(
        texts = chunks,
        embedding=embeddings,
        persist_directory=None
        )
    return vectorstore


def ask(query,vectorstore,chunks):
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 5
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5])
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=Groq(api_key=st.secrets["GROQ_API_KEY"])
        )
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer questions using only the context provided.Decorate the output with table lines and bullet points."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])
    chain = prompt | llm | StrOutputParser()
    docs = ensemble_retriever.invoke(query)
    if not docs:         
        return "Sorry, I couldn't find any relevant information in the PDF."
    context = "\n\n".join([doc.page_content for doc in docs])
    return chain.invoke({"context": context, "question": query})

st.title("Chat with your Hybrid PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        collection = store_chunks(chunks)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

    query = st.text_input("Ask a question about your PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = ask(query, collection, chunks)
        st.write(answer)