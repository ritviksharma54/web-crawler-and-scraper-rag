#streamlit_app.py
import streamlit as st
import os
import time # For simulated streaming and crawler delay
from dotenv import load_dotenv
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

# Import functions from our modules
from web_crawler import bfs_web_crawler # Ensure this matches your web_crawler.py
from rag_chatbot import initialize_rag_system, get_session_history, load_document_from_uploaded_file

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# --- Styling (Optional) ---
st.markdown("""
    <style>
    .stSpinner > div > div {
        border-top-color: #007bff; /* Primary blue for spinner */
    }
    .stButton>button {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- API Key Check ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()


# --- Session State Initialization ---
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None
if 'chat_history_store' not in st.session_state:
    st.session_state.chat_history_store = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sources_processed' not in st.session_state:
    st.session_state.sources_processed = False
if 'processed_documents_summary' not in st.session_state:
    st.session_state.processed_documents_summary = []
if 'last_scraped_links' not in st.session_state: # To store links for display
    st.session_state.last_scraped_links = []


# --- Helper function for history with RunnableWithMessageHistory ---
def get_streamlit_session_history(session_id: str) -> BaseChatMessageHistory:
    return get_session_history(session_id, st.session_state.chat_history_store)


# --- UI: Sidebar ---
with st.sidebar:
    st.header("📚 Data Sources")
    st.markdown("Upload documents or provide web URLs to build your knowledge base.")

    # File upload
    st.subheader("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, DOCX, CSV)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'doc', 'csv'],
        help="Upload one or more documents."
    )
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            st.caption(f"• {file.name}")

    st.markdown("---")

    # URL input for Custom BFS Crawler
    st.subheader("🌐 Web Content (Custom Crawler)")
    st.markdown("Enter Base URLs to crawl (one per line).")
    
    url_input = st.text_area(
        "Base URLs:",
        placeholder="https://example.com/page1\nhttps://another-site.com/article",
        height=100
    )
    urls_to_crawl = [url.strip() for url in url_input.split('\n') if url.strip()]

    # Depth is fixed at 1 as per requirement, but showing it disabled can be informative
    st.info("ℹ️ Crawler Depth is fixed at 1 (main page + its direct internal links).")
    # crawl_depth_fixed = 1 # For clarity

    # NEW: Max links to crawl input
    max_links_input = st.number_input(
        "Max URLs to Crawl (0-500):",
        min_value=0,
        max_value=500,
        value=20, # Default to a smaller number to prevent accidental large crawls
        step=5,
        help="Maximum unique URLs the crawler will attempt to visit (includes base URLs). "
             "If 0, no web crawling. To crawl only base URLs, set this to the number of base URLs provided."
    )

    if urls_to_crawl:
        st.write(f"**{len(urls_to_crawl)} Base URL(s) to crawl (up to {max_links_input} total):**")
        for url in urls_to_crawl:
            st.caption(f"• {url}")
    
    st.markdown("---")

    # Process button
    if (uploaded_files or urls_to_crawl):
        if st.button("🔄 Process All Sources", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing sources... Web crawling may take some time."):
                all_documents: List[Document] = []
                processed_summary = []
                st.session_state.last_scraped_links = [] # Reset before processing

                # 1. Process uploaded files
                if uploaded_files:
                    st.write("Processing uploaded files...")
                    for uploaded_file in uploaded_files:
                        try:
                            docs = load_document_from_uploaded_file(uploaded_file)
                            all_documents.extend(docs)
                            processed_summary.append(f"📄 File: {uploaded_file.name} ({len(docs)} section(s))")
                            st.success(f"Loaded {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                
                # 2. Process URLs with Custom BFS Crawler
                if urls_to_crawl:
                    if max_links_input > 0: # Only crawl if max_links is positive
                        st.write(f"Starting web crawl (Depth 1, Max URLs: {max_links_input})...")
                        try:
                            # Depth is fixed at 1 for this implementation in app.py
                            # Pass max_links_input to the crawler
                            web_docs, successfully_scraped_links = bfs_web_crawler(
                                urls_to_crawl, 
                                depth_limit=1, 
                                max_links_to_crawl=max_links_input # Pass the new limit
                            ) 
                            all_documents.extend(web_docs)
                            st.session_state.last_scraped_links = successfully_scraped_links

                            for doc in web_docs: 
                                 processed_summary.append(f"🌐 Web: {doc.metadata.get('source', 'Unknown URL')} (depth {doc.metadata.get('depth', 'N/A')})")
                            # Success message is now handled inside bfs_web_crawler
                            # st.success(f"Crawled and processed {len(web_docs)} web pages.")
                            
                        except Exception as e:
                            st.error(f"Error during web crawling: {e}")
                    else:
                        st.info("Web crawling skipped as 'Max URLs to Crawl' is set to 0.")


                if all_documents:
                    try:
                        base_rag_chain = initialize_rag_system(all_documents, GOOGLE_API_KEY)
                        
                        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                            base_rag_chain,
                            get_streamlit_session_history,
                            input_messages_key="input",
                            history_messages_key="chat_history",
                            output_messages_key="answer",
                        )
                        st.session_state.sources_processed = True
                        st.session_state.processed_documents_summary = processed_summary
                        st.session_state.messages = [] 
                        st.session_state.chat_history_store = {}  
                        st.success(f"✅ Successfully processed {len(all_documents)} total document sections!")
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Failed to initialize RAG system: {e}")
                else:
                    st.warning("No content could be processed from the provided sources. Please check crawler logs or upload files.")
    
    st.markdown("---")
    
    # Display processed sources summary
    if st.session_state.sources_processed and st.session_state.processed_documents_summary:
        with st.expander("📊 Processed Sources Summary", expanded=False): 
            for item in st.session_state.processed_documents_summary:
                st.caption(item)
    
    # Display Scraped Links from last run
    if st.session_state.last_scraped_links:
        with st.expander("🔗 View Successfully Scraped URLs (Last Run)", expanded=False):
            if st.session_state.last_scraped_links:
                for link in st.session_state.last_scraped_links:
                    st.markdown(f"- `{link}`") 
            else:
                st.caption("No URLs were scraped in the last processing run or list is empty.")


    st.markdown("---")
    # Reset/Clear buttons
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_store = {} 
        st.success("Chat history cleared.")
        st.rerun()
        
    if st.button("⚠️ Reset All & Clear Sources", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_store = {}
        st.session_state.conversational_rag_chain = None
        st.session_state.sources_processed = False
        st.session_state.processed_documents_summary = []
        st.session_state.last_scraped_links = []
        st.success("All settings, sources, and chat history have been reset.")
        st.rerun()

# --- Main Page UI ---
st.title("🤖 URL Crawler RAG Chatbot")
st.markdown("Chat with your documents and web content (crawled with Depth=1 and link limits) powered by Google Gemini.")

if not st.session_state.sources_processed:
    st.info("👋 Welcome! Please upload documents or add web URLs in the sidebar and click 'Process All Sources' to begin.")
    st.markdown("""
    #### How to use:
    1.  **Add Sources**: Use the sidebar to upload files or provide base web URLs.
        *   Configure the **Max URLs to Crawl** limit.
        *   Web content will be crawled with a depth of 1 (the base URL and its direct internal links), up to the specified URL limit.
    2.  **Process**: Click "Process All Sources". Your data will be processed and indexed.
    3.  **Chat**: Once processing is complete, ask questions in the chat window below!
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your processed content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.conversational_rag_chain:
                        response = st.session_state.conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": "streamlit_user_session"}} 
                        )
                        answer = response.get("answer", "No answer found.")
                        
                        # Simple simulated stream for nicer UX
                        if isinstance(answer, str):
                            for chunk in answer.split(): 
                                full_response_content += chunk + " "
                                time.sleep(0.02) 
                                message_placeholder.markdown(full_response_content + "▌")
                            message_placeholder.markdown(full_response_content)
                        else: # Handle non-string answers if they occur
                            full_response_content = str(answer)
                            message_placeholder.markdown(full_response_content)


                    else:
                        answer = "Error: RAG chain is not initialized. Please process sources first."
                        message_placeholder.error(answer)
                        full_response_content = answer


                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    answer = f"Sorry, I encountered an error: {str(e)}"
                    message_placeholder.error(answer)
                    full_response_content = answer


            st.session_state.messages.append({"role": "assistant", "content": full_response_content}) # Store full response

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Powered by Langchain, Google Gemini & Streamlit (Custom Crawler with Link Limiter)</div>", 
    unsafe_allow_html=True
)