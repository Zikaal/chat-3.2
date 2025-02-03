import streamlit as st
import logging
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)

mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["rag_db"]
collection = mongo_db["documents"]

class EmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        if len(input) == 0 or all([text.strip() == "" for text in input]):
            raise ValueError("Input query cannot be empty.")
        vectors = self.model.encode(input)
        if len(vectors) == 0:
            raise ValueError("Empty embedding generated.")
        return vectors


embedding = EmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")


def add_document_to_mongodb(documents, ids):
    try:
        for doc, doc_id in zip(documents, ids):
            if not doc.strip():
                raise ValueError("Cannot add an empty or whitespace-only document.")

            embedding_vector = embedding(doc)
            logging.info(f"Generated embedding for document '{doc}': {embedding_vector}")

            collection.insert_one({
                "_id": doc_id,
                "document": doc,
                "embedding": embedding_vector[0].tolist()
            })
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents. 
By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the similarity-based search. 
Provide these alternative questions separated by newlines. Original question: {question}"""


prompt_perspectives = ChatPromptTemplate.from_template(template)


openai_api_key = "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"


def generate_multi_queries(query_text):
    if not isinstance(query_text, str):
        raise ValueError("Expected a string input for query_text.")

    return (
            prompt_perspectives
            | ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if isinstance(q, str)])
    )


def reciprocal_rank_fusion(results, k=60):
    fused_scores = defaultdict(float)
    for rank, docs in enumerate(results):
        for idx, doc in enumerate(docs):
            fused_scores[doc[1]["document"]] += 1 / (k + idx + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def query_documents_from_mongodb_rag_fusion(query_text, n_results=3):
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Query text must be a non-empty string.")

    queries = generate_multi_queries(query_text)

    all_results = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            query_embedding = embedding(q)[0]
            docs = collection.find()
            similarities = []
            for doc in docs:
                doc_embedding = np.array(doc["embedding"])
                similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, doc))
            sorted_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:n_results]
            all_results.append(sorted_results)
    fused_results = reciprocal_rank_fusion(all_results)
    return [doc[0] for doc in fused_results[:n_results]]


def query_with_ollama(prompt, model_name):
    try:
        logging.info(f"Sending prompt to Ollama with model {model_name}: {prompt}")
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        logging.info(f"Ollama response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error with Ollama query: {e}")
        return f"Error with Ollama API: {e}"

def retrieve_and_answer(query_text, model_name):
    retrieved_docs = query_documents_from_mongodb_rag_fusion(query_text)
    context = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    return query_with_ollama(augmented_prompt, model_name)


def get_constitution_text():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        constitution_text = ""
        for paragraph in soup.find_all("p"):
            constitution_text += paragraph.get_text() + "\n"
        logging.info(f"Extracted Constitution text: {constitution_text[:500]}...")
        return constitution_text

    else:
        logging.error("Error fetching the Constitution text from the website.")
        return "Error fetching the Constitution text from the website."

st.title("Chat with Ollama - Enhanced with Multi-Query & RAG Fusion")

model = "llama3.2:1b"
menu = st.sidebar.selectbox("Choose an action", [
    "Show Documents in MongoDB", "Add New Document to MongoDB as Vector",
    "Upload File and Ask Question", "Ask Ollama a Question",
    "Ask Question About Constitution"])

if menu == "Ask Ollama a Question":
    query = st.text_input("Ask a question")
    if query:
        response = retrieve_and_answer(query, model)
        st.write("Response:", response)
if menu == "Ask Question About Constitution":
    question = st.text_input("Ask a question about the Constitution of Kazakhstan")
    if question:
        try:

            constitution_text = get_constitution_text()
            if constitution_text:

                context = constitution_text[:2000]  #
                logging.info(f"Constitution text: {context[:500]}...")


                augmented_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                response = query_with_ollama(augmented_prompt, model)


                st.write("Constitution Text (Extract):")
                st.text_area("Constitution Text", context, height=200)
                st.write("Response from Ollama:", response)


                summary_prompt = f"Summarize the following content: {context}"
                summary = query_with_ollama(summary_prompt, model)
                st.write("Summary of the Constitution Text:", summary)
            else:
                st.write("Failed to fetch Constitution text.")
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            st.write("An error occurred while processing the request.")

