# Advanced RAG Chatbot with Multi-Document Comparison

This Streamlit application demonstrates a dual-mode Retrieval-Augmented Generation (RAG) system. It allows users to chat with their documents in a standard conversational mode or activate a powerful **Comparison Mode** to perform a deep, thematic analysis across multiple documents simultaneously.

 

---

## Key Features

- **Multi-Source Ingestion**: Upload multiple documents (`.pdf`, `.txt`, `.docx`, `.csv`) or scrape web content using a built-in BFS web crawler.
- **Standard Chat Mode**: A conversational interface with chat history awareness for asking questions about the entire knowledge base.
- **N-Way Comparison Mode**: An advanced mode to compare two or more documents. It solves common RAG failures for broad, comparative queries.
- **Agentic-Style Analysis**: Utilizes an "Iterative Thematic Search" workflow to ensure robust and comprehensive comparisons without exceeding context windows.
- **Scalable Architecture**: The comparison logic is designed to work efficiently even with a large number of documents in the knowledge base.
- **Metadata-Aware**: The system is aware of document sources and uses this metadata to perform fair, per-document retrieval during comparison.

---

## Architecture Deep Dive: The "Iterative Thematic Search"

A simple RAG system fails when asked a broad, comparative query like "list all the changes between these documents." A basic vector search for "list all changes" will retrieve irrelevant introductory chunks.

This application solves that problem with a more intelligent, multi-step approach for its Comparison Mode:

1.  **Theme Prediction**: Instead of a direct, naive search, the system first uses an LLM to predict the likely themes of comparison based on the user's query and the filenames of the selected documents. For example, when comparing two legal acts, it might predict themes like "Penalties and Fines," "Licensing Requirements," and "Definitions."

2.  **Per-Document Thematic Retrieval**: The system then loops through each predicted theme. For **each theme**, it performs a new loop through **each selected document**. In this inner loop, it executes a targeted vector search for the theme, using a **metadata filter** to ensure it only retrieves chunks from that single document. This guarantees that every document gets a fair chance to contribute relevant information for every topic, solving the "document dominance" problem where a larger document might crowd out a smaller one.

3.  **Final Synthesis**: All the relevant chunks gathered from the thematic retrieval are compiled into a single, curated context. This context, which is much smaller than the full documents but highly relevant to the query, is then sent in a **single call** to the LLM with a powerful prompt instructing it to generate a final, structured, comparative analysis.

This architecture is both **scalable** (it never loads full documents into context) and **robust** (it transforms vague queries into a series of specific, targeted searches).

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- A Google Gemini API Key

### 1. Clone the Repository
```bash
git clone <https://github.com/ritviksharma54/web-crawler-and-scraper-rag.git>
cd <https://github.com/ritviksharma54/web-crawler-and-scraper-rag.git>
```

### 2. Create and Activate a Virtual Environment
- **On macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows:**
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
This project uses a `.env` file to manage the API key securely.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your Google Gemini API key to the file as follows:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
3.  **Important**: Add `.env` to your `.gitignore` file to prevent accidentally committing your secret key.

---

## How to Run the Application
With your virtual environment activated and your `.env` file set up, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```
The application will open in your web browser.

---

## How to Use

1.  **Add Sources**: Upload one or more documents, or enter URLs to be scraped in the sidebar.
2.  **Process Sources**: Click the "Process All Sources" button. This will load, chunk, and embed the content into the vector store.
3.  **Standard Chat**: Ask questions in the main chat input to query all processed documents.
4.  **Comparison Mode**:
    -   Toggle on "Enable Comparison Mode" in the sidebar.
    -   A multi-select box will appear. You can:
        -   **Select 2 or more documents** to run a focused comparison on just those.
        -   **Leave it blank** to automatically run the comparison on **all** processed documents.
    -   Ask a broad, comparative question like "What are the key differences?" or "list all the changes."

---

## Project Structure
```
.
├── .env                  # Stores your secret API key (you must create this)
├── requirements.txt      # Lists all Python dependencies
├── streamlit_app.py      # Main Streamlit frontend and application logic
├── rag_chatbot.py        # Core RAG and comparison chain logic
├── web_crawler.py        # Web scraping and content extraction logic
└── README.md             # This file
```