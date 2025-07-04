#rag_chatbot.py 
import os
import re
import warnings
from typing import List, Dict
import tempfile
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()
warnings.filterwarnings('ignore')

# --- Prompts ---
CONTEXTUAL_QUESTION_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
QA_SYSTEM_PROMPT_TEMPLATE = (
    "You are an assistant for question-answering tasks based on provided context. "
    "Use ONLY the following pieces of retrieved context to answer the question. "
    "If the context doesn't contain enough information, state that clearly. "
    "When referencing information, try to mention the source (file name or URL from metadata). "
    "Keep your answer accurate and concise."
    "\n\n"
    "Context from sources:\n{context}"
)
THEME_GENERATION_PROMPT_TEMPLATE = (
    "You are a research planning AI. Your goal is to predict the key topics for comparing a set of documents based on their filenames and the user's query. "
    "Do not answer the query. Simply list the most important themes to investigate."
    "\n\nUser Query: '{user_query}'"
    "\n\nDocument Filenames: {filenames}"
    "\n\nBased on the above, generate a list of 5-7 distinct themes or topics that are likely to contain the most significant differences between these documents. "
    "For example, if comparing legal acts, themes might be 'Definitions', 'Penalties and Fines', 'Licensing Procedures', etc."
    "Output your response as a numbered list."
)
FINAL_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a meticulous comparative analysis expert. You will be given a user's query and a collection of text chunks retrieved from multiple documents. "
    "These chunks have been gathered by searching for specific themes related to the query."
    "\n\n**Your Task:**"
    "\n1. **Synthesize Findings:** Analyze all the provided context to identify the key similarities and differences between the source documents."
    "\n2. **Structure the Report:** Create a final, well-structured report that directly answers the user's query. Use clear, thematic headings."
    "\n3. **Detail the Comparison:** Under each heading, use bullet points to detail the specifics from each source document (e.g., '- source1.pdf: [detail]', '- source2.pdf: [detail]')."
    "\n4. **Be Comprehensive but Concise:** Cover all relevant points found in the context. If the context for a theme is missing or insufficient, state that."
    "\n\n**User's Query:** '{user_query}'"
    "\n\n**Retrieved Context for Analysis:**\n---\n{context}\n---"
)

# ... (Helper functions like get_session_history, load_document_from_uploaded_file, etc. are unchanged) ...

def get_session_history(session_id: str, store: Dict[str, BaseChatMessageHistory]) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def load_document_from_uploaded_file(uploaded_file) -> List[Document]:
    documents = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file_path = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        loader_map = {
            '.pdf': PyPDFLoader, '.txt': TextLoader, '.docx': Docx2txtLoader, 
            '.doc': Docx2txtLoader, '.csv': CSVLoader
        }
        if file_extension not in loader_map:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loader = loader_map[file_extension](tmp_file_path)
        loaded_docs = loader.load()
        
        for doc in loaded_docs:
            doc.metadata['source'] = uploaded_file.name
        documents.extend(loaded_docs)
            
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    return documents

def _format_docs_for_prompt(docs: List[Document]) -> str:
    final_doc_strings = []
    for doc in docs:
        doc_source = doc.metadata.get('source', 'Unknown Source')
        page_content = doc.page_content
        description = f"Source: {doc_source}\nContent: {page_content}"
        final_doc_strings.append(description)
    return "\n\n---\n\n".join(final_doc_strings)

def _parse_list_from_llm(response_content: str) -> List[str]:
    lines = response_content.strip().split('\n')
    items = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if line.strip()]
    return items


# ######################################################################
# ##### START OF MODIFIED THEMATIC SEARCH WORKFLOW FUNCTION #####
# ######################################################################

def thematic_search_workflow(input_dict: dict, model, vectorstore) -> dict:
    user_query = input_dict["input"]
    selected_sources = input_dict["selected_sources"]

    # 1. Generate search themes based on query and filenames
    theme_generation_prompt = ChatPromptTemplate.from_template(THEME_GENERATION_PROMPT_TEMPLATE)
    theme_generation_chain = theme_generation_prompt | model | StrOutputParser()
    
    filenames_str = ", ".join([os.path.basename(s) for s in selected_sources])
    themes_response = theme_generation_chain.invoke({
        "user_query": user_query,
        "filenames": filenames_str,
    })
    search_themes = _parse_list_from_llm(themes_response)

    if not search_themes:
        return {"answer": "I could not determine the key topics for comparison. Please try a more specific query."}

    # 2. For each theme AND each document, retrieve relevant chunks
    all_retrieved_docs = []
    seen_docs = set() 
    
    for theme in search_themes:
        for source in selected_sources:
            # Create a retriever specifically for this document and theme
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                # Retrieve a smaller number (k=2 or 3) since we are searching per-document
                search_kwargs={"k": 2, "filter": {"source": {"$in": [source]}}} 
            )
            
            # Invoke the retriever for the current theme within the current document
            retrieved_docs = retriever.invoke(theme)
            
            for doc in retrieved_docs:
                if doc.page_content not in seen_docs:
                    all_retrieved_docs.append(doc)
                    seen_docs.add(doc.page_content)

    if not all_retrieved_docs:
        return {"answer": "I could not find any relevant information for the predicted themes in the selected documents."}

    # 3. Synthesize the final answer from the collected, balanced set of chunks
    formatted_context = _format_docs_for_prompt(all_retrieved_docs)
    
    final_analysis_prompt = ChatPromptTemplate.from_template(FINAL_ANALYSIS_PROMPT_TEMPLATE)
    final_analysis_chain = final_analysis_prompt | model | StrOutputParser()
    
    final_response = final_analysis_chain.invoke({
        "user_query": user_query,
        "context": formatted_context,
    })

    return {"answer": final_response, "context": []}

# ####################################################################
# ##### END OF MODIFIED WORKFLOW FUNCTION #####
# ####################################################################


def initialize_rag_system(documents: List[Document], google_api_key: str):
    # ... (This entire function is now correct and does not need to be changed) ...
    os.environ["GOOGLE_API_KEY"] = google_api_key
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        raise ValueError("No text could be extracted or split from the documents.")

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
    
    # --- Standard RAG Chain (Path 1) ---
    standard_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 7})
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUAL_QUESTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(model, standard_retriever, contextualize_q_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT_TEMPLATE),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    question_answer_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | model
    )
    standard_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- Thematic Comparison Chain (Path 2) ---
    thematic_comparison_chain = (
        {
            "input": lambda x: x["input"],
            "selected_sources": lambda x: x["selected_sources"],
        }
        | RunnableLambda(lambda x: thematic_search_workflow(x, model, vectorstore))
    )

    # --- Branching Logic ---
    branch = RunnableBranch(
        (lambda x: x.get("is_comparison_mode", False), thematic_comparison_chain),
        standard_rag_chain,
    )
    
    return branch