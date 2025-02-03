import streamlit as st
import logging
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
import requests
import chardet
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

def generate_multi_queries(query_text, num_queries=5):
        """
        Generate alternative queries from the original question using Ollama.
        """
        prompt = (
            f"You are an AI language model assistant. Your task is to generate {num_queries} different "
            f"versions of the given user question to retrieve relevant documents from a vector database. "
            f"By generating multiple perspectives on the user question, your goal is to help the user overcome "
            f"some of the limitations of the distance-based similarity search. Provide these alternative questions "
            f"separated by newlines.\nOriginal question: {query_text}"
        )
        response = query_with_ollama(prompt, model)
        alternative_queries = [q.strip() for q in response.split("\n") if q.strip()]

        st.write("Generated queries:")
        for q in alternative_queries:
            st.write(f"- {q}")

        return alternative_queries


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

if menu == "Show Documents in MongoDB":
    st.subheader("Stored Documents in MongoDB")
    documents = collection.find()
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc['document']}")
    else:
        st.write("No data available!")

elif menu == "Add New Document to MongoDB as Vector":
    st.subheader("Add a New Document to MongoDB")
    new_doc = st.text_area("Enter the new document:")
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

    if st.button("Add Document"):
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                detected_encoding = chardet.detect(file_bytes)['encoding']
                if not detected_encoding:
                    raise ValueError("Failed to detect file encoding.")
                file_content = file_bytes.decode(detected_encoding)

                doc_id = f"doc{collection.count_documents({}) + 1}"
                st.write(f"Adding document from file: {uploaded_file.name}")
                add_document_to_mongodb([file_content], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        elif new_doc.strip(): 
            try:
                doc_id = f"doc{collection.count_documents({}) + 1}"
                st.write(f"Adding document: {new_doc}")
                add_document_to_mongodb([new_doc], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        else:
            st.warning("Please enter a non-empty document or upload a file before adding.")

elif menu == "Upload File and Ask Question":
    st.subheader("Upload a file and ask a question about its content")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            detected_encoding = chardet.detect(file_bytes)['encoding']
            if not detected_encoding:
                raise ValueError("Failed to detect file encoding.")
            file_content = file_bytes.decode(detected_encoding)

            st.write("File content successfully loaded:")
            st.text_area("File Content", file_content, height=200)

            question = st.text_input("Ask a question about this file's content:")
            if question:
                response = query_with_ollama(f"Context: {file_content}\n\nQuestion: {question}\nAnswer:", model)
                st.write("Response:", response)

        except Exception as e:
            st.error(f"Failed to process the file: {e}")


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

