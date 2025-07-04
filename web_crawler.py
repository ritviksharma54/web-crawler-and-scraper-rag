# web_crawler.py

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set, Tuple
from langchain_core.documents import Document
import streamlit as st 


def scrape_webpage_content(url: str, timeout: int = 10, max_content_length: int = 50000) -> Tuple[str, str]:
    """
    Scrapes the textual content and title from a single webpage.

    Args:
        url: The URL of the webpage to scrape.
        timeout: Request timeout in seconds.
        max_content_length: Maximum characters to extract to prevent overly large documents.

    Returns:
        A tuple (page_text, page_title). Returns (None, None) on failure.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            # st.warning(f"Skipping non-HTML content at {url} (type: {content_type})") # Can be verbose
            return None, None

        soup = BeautifulSoup(response.content, 'html.parser')
        page_title = soup.title.string.strip() if soup.title else url.split('/')[-1] or url

        for S in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe", "noscript"]):
            S.decompose()
        
        body = soup.find('body')
        if not body:
            return None, page_title

        text_parts = []
        for element in body.find_all(string=True):
            if element.parent.name in ["script", "style", "noscript", "header", "nav", "footer", "aside", "form"]:
                continue
            stripped_text = element.strip()
            if stripped_text:
                text_parts.append(stripped_text)
        
        text = "\n".join(text_parts)
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
        if len(text) > max_content_length:
            text = text[:max_content_length] + "..."
            # st.info(f"Content from {url} was truncated to {max_content_length} characters.") # Can be verbose
        
        return text, page_title
        
    except requests.exceptions.RequestException: # More specific exceptions can be caught if needed
        # st.warning(f"Failed to fetch {url}: {str(e)}") # Often noisy, general failure logged by caller
        return None, None
    except Exception:
        # st.warning(f"Error scraping content from {url}: {str(e)}")
        return None, None

def get_internal_links(base_url: str, soup: BeautifulSoup) -> Set[str]:
    """
    Extracts internal (same-domain) links from a BeautifulSoup object.
    """
    internal_links = set()
    parsed_base_url = urlparse(base_url)
    
    for link_tag in soup.find_all('a', href=True):
        href = link_tag['href']
        full_url = urljoin(base_url, href)
        parsed_full_url = urlparse(full_url)

        if (parsed_full_url.scheme in ['http', 'https'] and
            parsed_full_url.netloc == parsed_base_url.netloc and
            parsed_full_url.path and
            '#' not in parsed_full_url.path.split('/')[-1] and
            not parsed_full_url.fragment and
            not full_url.startswith('mailto:') and
            not full_url.startswith('tel:')):
            if not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx', '.gif', '.mp4', '.mp3', '.svg', '.webp']): # Expanded exclusions
                internal_links.add(full_url.split('#')[0])
    return internal_links

def bfs_web_crawler(start_urls: List[str], depth_limit: int = 1, max_links_to_crawl: int = 500) -> Tuple[List[Document], List[str]]:
    """
    Crawls web pages starting from start_urls using BFS up to a specified depth,
    respecting a maximum number of links to crawl.

    Args:
        start_urls: A list of URLs to begin crawling from.
        depth_limit: Max depth to crawl. 0 means only start_urls, 1 means start_urls + their direct links.
        max_links_to_crawl: Maximum number of unique URLs to add to the crawl queue.
                            Includes the initial start_urls. If 0, no URLs are crawled.

    Returns:
        A tuple: (list of Langchain Document objects, list of successfully scraped URLs).
    """
    documents: List[Document] = []
    scraped_urls_success: List[str] = []
    
    queue: List[Tuple[str, int]] = [] # (url, current_depth)
    visited_urls: Set[str] = set()
    limit_reached_message_displayed = False

    if max_links_to_crawl == 0:
        st.info("Max links to crawl is set to 0. No URLs will be processed.")
        return [], []

    for url_input in start_urls:
        url = url_input.strip()
        if url: # Ensure URL is not empty
            if len(visited_urls) < max_links_to_crawl:
                if url not in visited_urls: # Only add if truly new
                    queue.append((url, 0))
                    visited_urls.add(url)
            else:
                if not limit_reached_message_displayed:
                    st.info(f"Maximum link limit ({max_links_to_crawl}) reached during initial URL processing. Some base URLs may not be added to the queue.")
                    limit_reached_message_displayed = True
                break # Stop adding more start_urls if limit is hit

    if not queue:
        st.info("No valid start URLs provided or limit too low for crawling.")
        return [], []

    total_to_process_estimate = len(queue) # Initial estimate based on queued start URLs
    processed_count = 0
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    while queue:
        current_url, current_depth = queue.pop(0)
        processed_count += 1
        
        progress = min(1.0, processed_count / total_to_process_estimate) if total_to_process_estimate > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Processing ({processed_count}/{len(visited_urls)} visited, {total_to_process_estimate} queued): {current_url} (Depth: {current_depth})")

        # st.write(f"Attempting to scrape: {current_url}") # Verbose logging

        content, title = scrape_webpage_content(current_url)

        if content and content.strip():
            doc = Document(
                page_content=content,
                metadata={
                    'source': current_url,
                    'source_type': 'web_bfs_crawl',
                    'title': title or current_url.split('/')[-1],
                    'depth': current_depth
                }
            )
            documents.append(doc)
            scraped_urls_success.append(current_url)

            if current_depth < depth_limit:
                if len(visited_urls) >= max_links_to_crawl:
                    if not limit_reached_message_displayed:
                        st.info(f"Maximum link limit ({max_links_to_crawl}) reached. No new links will be discovered from {current_url}.")
                        limit_reached_message_displayed = True
                else:
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response_for_links = requests.get(current_url, headers=headers, timeout=5)
                        response_for_links.raise_for_status()
                        if 'html' in response_for_links.headers.get('content-type', '').lower():
                            soup_for_links = BeautifulSoup(response_for_links.content, 'html.parser')
                            links = get_internal_links(current_url, soup_for_links)
                            for link in links:
                                if link not in visited_urls:
                                    if len(visited_urls) < max_links_to_crawl:
                                        visited_urls.add(link)
                                        queue.append((link, current_depth + 1))
                                        total_to_process_estimate += 1 # Update estimate
                                    else:
                                        if not limit_reached_message_displayed:
                                            st.info(f"Maximum link limit ({max_links_to_crawl}) reached. No more new links will be queued.")
                                            limit_reached_message_displayed = True
                                        break # Stop adding more links from THIS page
                        # else: # Can be noisy
                            # st.caption(f"Skipping link extraction (not HTML): {current_url}")
                    except Exception as e:
                        st.warning(f"Could not extract links from {current_url}: {e}")
        # else: # Can be noisy
            # st.warning(f"No substantial content or failed to scrape: {current_url}")

        time.sleep(0.2) # Politeness delay

    progress_bar.empty()
    status_text.empty()
    
    if documents:
        st.success(f"BFS Crawling finished. Scraped content from {len(scraped_urls_success)} out of {len(visited_urls)} visited/considered URLs.")
    elif len(visited_urls) > 0 : # Visited some but scraped none
        st.warning(f"BFS Crawling finished. Visited {len(visited_urls)} URLs but no content was successfully scraped.")
    else: # Did not visit any URL (e.g. max_links_to_crawl=0 or no valid start_urls)
        st.info("BFS Crawling finished. No URLs were processed.")
        
    return documents, scraped_urls_success

if __name__ == '__main__':
    print("Custom BFS Web Crawler Module (run from Streamlit app for UI elements")