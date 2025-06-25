#rag_chatbot.py
import os
import warnings
from typing import List, Dict, Any
import tempfile
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

load_dotenv()
warnings.filterwarnings('ignore')

# Global store for session histories (managed by Streamlit's session_state in the app)
# st.session_state.store will be used in the main app.
# This function will be passed to RunnableWithMessageHistory
def get_session_history(session_id: str, store: Dict[str, BaseChatMessageHistory]) -> BaseChatMessageHistory:
    """Get or create session history for a given session ID from the provided store."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def load_document_from_uploaded_file(uploaded_file) -> List[Document]:
    """Load a document from an uploaded file object based on its file type."""
    documents = []
    
    # Create a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(tmp_file_path)
        elif file_extension == 'csv':
            loader = CSVLoader(tmp_file_path)
        else:
            # This error should be displayed in Streamlit UI
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loaded_docs = loader.load()
        
        # Add metadata about the source file
        for doc in loaded_docs:
            doc.metadata['source'] = uploaded_file.name
            doc.metadata['source_type'] = 'file'
            doc.metadata['file_type'] = file_extension
        documents.extend(loaded_docs)
            
    except Exception as e:
        # Propagate exception to be handled by Streamlit UI
        raise RuntimeError(f"Error loading {uploaded_file.name}: {str(e)}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    
    return documents


def initialize_rag_system(documents: List[Document], google_api_key: str):
    """Initialize the RAG system with processed documents."""
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found or provided.")
    
    os.environ["GOOGLE_API_KEY"] = google_api_key # Set for Langchain components
    
    try:
        # Initialize embeddings and model
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            convert_system_message_to_human=True,
            temperature=0.1
        )
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Increased chunk_size a bit
            chunk_overlap=200, # Increased overlap a bit
            separators=["\n\n", "\n", ". ", "!", "?", ",", " ", ""],
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("No text could be extracted from the documents after splitting. Check document content and text splitter.")

        # Create vector store and retriever
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=gemini_embeddings
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr", # Maximal Marginal Relevance
            search_kwargs={"k": 5, "lambda_mult": 0.7} # Fetch 5, diversity 0.7
        )
        
        # System prompt for QA
        system_prompt_qa = (
            "You are an assistant for question-answering tasks based on provided context. "
            "Use ONLY the following pieces of retrieved context to answer the question. "
            "If the context doesn't contain enough information to answer the question, clearly state that "
            "the information is not available in the provided sources. "
            "When referencing information, try to mention the source (file name or website URL from metadata) if available and relevant. "
            "Keep your answer accurate and concise, aiming for 3-5 sentences unless more detail is crucial."
            "\n\n"
            "Context from sources:\n{context}"
        )
        
        # Prompt for reformulating question based on history
        retriever_prompt_history = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", retriever_prompt_history),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_qa),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            model, retriever, contextualize_q_prompt
        )
        
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # The get_session_history function needs the actual store,
        # which will be st.session_state.store in the Streamlit app.
        # So, we'll wrap it in a lambda when creating RunnableWithMessageHistory in the app.
        
        return rag_chain
        
    except Exception as e:
        # Propagate exception to be handled by Streamlit UI
        raise RuntimeError(f"Error initializing RAG system: {str(e)}")