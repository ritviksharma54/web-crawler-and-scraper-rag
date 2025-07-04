# streamlit_app.py
import streamlit as st
import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from web_crawler import bfs_web_crawler
from rag_chatbot import initialize_rag_system, get_session_history, load_document_from_uploaded_file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chatbot with N-Way Comparison", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stSpinner > div > div { border-top-color: #007bff; }
    .stButton>button { border-radius: 0.5rem; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #007bff !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
default_session_state = {
    'conversational_rag_chain': None,
    'chat_history_store': {},
    'messages': [],
    'sources_processed': False,
    'processed_documents_summary': [],
    'last_scraped_links': [],
    'comparison_mode_active': False,
    'comparison_docs_selected': [],
    'available_sources_for_comparison': [] 
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper function for history ---
def get_streamlit_session_history(session_id: str) -> BaseChatMessageHistory:
    return get_session_history(session_id, st.session_state.chat_history_store)

def generate_display_name(source_name: Optional[str]) -> str:
    if not source_name:
        return ""
    # Truncate long names for display
    name = os.path.basename(source_name) # Get filename from path/URL
    return name[:40] + '...' if len(name) > 40 else name


# --- UI: Sidebar ---
with st.sidebar:
    st.header("📚 Data Sources")

    # File upload
    st.subheader("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, DOCX, CSV)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'doc', 'csv']
    )

    st.markdown("---")

    # URL input
    st.subheader("🌐 Web Content (Custom Crawler)")
    url_input = st.text_area(
        "Base URLs (one per line):",
        placeholder="https://example.com/page1\nhttps://another-site.com/article",
        height=100
    )
    urls_to_crawl = [url.strip() for url in url_input.split('\n') if url.strip()]
    max_links_input = st.number_input(
        "Max URLs to Crawl (0-500):", min_value=0, max_value=500, value=10, step=5
    )
    
    st.markdown("---")

    # Process button
    if (uploaded_files or urls_to_crawl):
        if st.button("🔄 Process All Sources", type="primary", use_container_width=True):
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            if not GOOGLE_API_KEY:
                st.error("🚨 GOOGLE_API_KEY not found. Please set it as an environment variable.")
                st.stop()

            with st.spinner("🧠 Processing sources... This may take some time."):
                current_docs_for_rag: List[Document] = []
                
                # 1. Process uploaded files
                if uploaded_files:
                    st.write("Processing uploaded files...")
                    for uploaded_file in uploaded_files:
                        try:
                            docs = load_document_from_uploaded_file(uploaded_file)
                            current_docs_for_rag.extend(docs)
                            summary_item = {
                                'name': uploaded_file.name, 
                                'type': 'file', 
                                'sections': len(docs),
                                'display_name': generate_display_name(uploaded_file.name)
                            }
                            if not any(s['name'] == summary_item['name'] for s in st.session_state.processed_documents_summary):
                                st.session_state.processed_documents_summary.append(summary_item)
                            st.success(f"Loaded {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                
                # 2. Process URLs
                if urls_to_crawl and max_links_input > 0:
                    st.write(f"Starting web crawl (Depth 1, Max URLs: {max_links_input})...")
                    try:
                        web_docs, successfully_scraped_links = bfs_web_crawler(
                            urls_to_crawl, depth_limit=1, max_links_to_crawl=max_links_input
                        )
                        current_docs_for_rag.extend(web_docs)
                        st.session_state.last_scraped_links.extend(successfully_scraped_links)
                        
                        for doc_url in successfully_scraped_links:
                            doc_sections = sum(1 for d in web_docs if d.metadata.get('source') == doc_url)
                            summary_item = {
                                'name': doc_url, 
                                'type': 'web', 
                                'sections': doc_sections,
                                'display_name': generate_display_name(doc_url)
                            }
                            if not any(s['name'] == summary_item['name'] for s in st.session_state.processed_documents_summary):
                                st.session_state.processed_documents_summary.append(summary_item)
                    except Exception as e:
                        st.error(f"Error during web crawling: {e}")

                if current_docs_for_rag:
                    try:
                        st.session_state.available_sources_for_comparison = sorted(list(set(
                            item['name'] for item in st.session_state.processed_documents_summary
                        )))

                        base_rag_chain = initialize_rag_system(current_docs_for_rag, GOOGLE_API_KEY)
                        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                            base_rag_chain,
                            get_streamlit_session_history,
                            input_messages_key="input",
                            history_messages_key="chat_history",
                            output_messages_key="answer",
                        )
                        st.session_state.sources_processed = True
                        st.session_state.messages = [] 
                        st.session_state.chat_history_store = {}
                        st.success(f"✅ RAG system (re)initialized with {len(current_docs_for_rag)} document sections from this run.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize RAG system: {e}")
                else:
                    st.warning("No content could be processed from the provided sources in this run.")
    
    st.markdown("---")

    # --- Comparison Mode UI ---
    st.subheader("⚖️ Multi-Document Comparison")
    st.session_state.comparison_mode_active = st.toggle(
        "Enable Comparison Mode", 
        value=st.session_state.comparison_mode_active,
        help="Compare content between two or more specific documents using an agentic workflow."
    )

    if st.session_state.comparison_mode_active:
        if not st.session_state.processed_documents_summary:
            st.warning("No documents have been processed yet.")
            st.session_state.comparison_mode_active = False # Disable if no sources
        else:
            st.markdown("Optionally, select specific documents to compare. **Leave blank to compare all.**")
            
            display_name_to_source_map = {s['display_name']: s['name'] for s in st.session_state.processed_documents_summary}
            all_source_names = list(display_name_to_source_map.values())
            
            selected_display_names = st.multiselect(
                "Documents to Compare:",
                options=display_name_to_source_map.keys(),
                help="Select specific documents to analyze. If empty, all documents will be compared."
            )
            
            
            # Convert selected display names back to their actual source names
            selected_source_names = [display_name_to_source_map[name] for name in selected_display_names]

            # If the user has selected any documents, use that list.
            # Otherwise, use the full list of all available documents.
            if selected_source_names:
                st.session_state.comparison_docs_selected = selected_source_names
                if len(st.session_state.comparison_docs_selected) == 1:
                     st.warning("Please select at least two documents for a comparison, or clear the selection to compare all.")
                else:
                    st.success(f"Comparing {len(st.session_state.comparison_docs_selected)} selected documents.")
            else:
                st.session_state.comparison_docs_selected = all_source_names
                st.info(f"Comparing all {len(st.session_state.comparison_docs_selected)} available documents.")


    st.markdown("---")

    if st.session_state.processed_documents_summary:
        with st.expander("📊 Processed Sources Summary", expanded=False): 
            for item in st.session_state.processed_documents_summary:
                st.caption(f"{'📄' if item['type'] == 'file' else '🌐'} {item['display_name']} ({item['sections']} sections)")
    
    if st.session_state.last_scraped_links:
        with st.expander("🔗 View Scraped URLs", expanded=False):
            for link in st.session_state.last_scraped_links:
                st.markdown(f"- `{link}`") 
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_store = {} 
        st.success("Chat history cleared.")
        st.rerun()
        
    if st.button("⚠️ Reset All & Clear Sources", use_container_width=True, type="secondary"):
        for key in default_session_state.keys():
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Main Page UI ---
st.title("🤖 RAG Chatbot with Agentic Comparison")

if st.session_state.comparison_mode_active:

    # Check if we have enough documents to compare
    if len(st.session_state.comparison_docs_selected) >= 2:
        st.info(f"⚖️ **Agentic Comparison Mode Active:** Analyzing **{len(st.session_state.comparison_docs_selected)} documents**. Your query will be broken down into sub-questions to synthesize a comprehensive answer.")
    else:
        st.warning("Comparison mode requires at least two processed documents. Please process more sources or clear your selection to compare all available ones.")

else:
    st.markdown("Chat with your documents and web content. Enable Comparison Mode in the sidebar to analyze multiple documents.")


if not st.session_state.conversational_rag_chain:
    st.info("👋 Welcome! Please upload documents or add web URLs and click 'Process All Sources' to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            
            
            # The condition for being "ready" is now simpler and more robust
            is_comparison_ready = (
                st.session_state.comparison_mode_active and 
                len(st.session_state.comparison_docs_selected) >= 2
            )
            
            with st.spinner("Thinking... (Agentic mode may take longer)"):
                try:
                    chain_input = {"input": prompt}
                    if is_comparison_ready:
                        chain_input["is_comparison_mode"] = True
                        chain_input["selected_sources"] = st.session_state.comparison_docs_selected
                    else:
                        chain_input["is_comparison_mode"] = False
                        chain_input["selected_sources"] = []

                    response = st.session_state.conversational_rag_chain.invoke(
                        chain_input,
                        config={"configurable": {"session_id": "streamlit_user_session"}} 
                    )
                    
                    answer_object = response.get("answer")
                    
                    if hasattr(answer_object, 'content'):
                        full_response_content = answer_object.content
                    elif isinstance(answer_object, str):
                        full_response_content = answer_object
                    else:
                        full_response_content = str(answer_object)

                    # Fake streaming for better UX
                    streamed_text = ""
                    for chunk in full_response_content.split(" "):
                        streamed_text += chunk + " "
                        time.sleep(0.02) 
                        message_placeholder.markdown(streamed_text + "▌")
                    message_placeholder.markdown(streamed_text.strip())
                    full_response_content = streamed_text.strip()

                except Exception as e:
                    import traceback
                    st.error(f"Error generating response: {str(e)}")
                    st.code(traceback.format_exc())
                    full_response_content = f"Sorry, I encountered an error: {str(e)}"
                    message_placeholder.error(full_response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Powered by Langchain, Google Gemini & Streamlit</div>", 
    unsafe_allow_html=True
)