# Chat with Ollama - Enhanced with Multi-Query & RAG Fusion

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system with Multi-Query enhancement, utilizing MongoDB for document storage and retrieval, Langchain for AI integration, and Streamlit for the front-end interface.

## Features

- Multi-Query Expansion to improve document retrieval.
- RAG Fusion to enhance response quality.
- Integration with MongoDB for storing document embeddings.
- Utilizes Langchain and OpenAI APIs for natural language processing.
- Allows users to interact with the system through a Streamlit-based UI.
- Fetches and processes the Constitution of Kazakhstan for legal Q&A.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/chat-3.2.git
   cd chat-3.2
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Ensure MongoDB is running locally on `mongodb://localhost:27017/`.
  
5. Run the application:
   ```bash
   streamlit run temple.py
   ```

## Usage

To start the application, run:

```sh
streamlit run temple.py
```

## Example Queries

- "What are the fundamental rights in the Constitution of Kazakhstan?"
- "Summarize the key sections of the Constitution."
- "How does Ollama AI generate responses?"

## Project Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── temple.py
├── test/
│   ├── temple.py
```

## Contributors

- **Alexandr Chshudro**
- **Zinetov Alikhan**
- **Alisher Samat**


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

