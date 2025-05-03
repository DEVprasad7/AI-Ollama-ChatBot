# web_search.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys - Get from environment variables or set directly
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # Google Search API alternative
SERP_API_KEY = os.getenv("SERP_API_KEY")      # Alternative search API

class WebSearch:
    def __init__(self):
        self.search_results_cache = {}
        self.content_cache = {}
    
    def search_using_serper(self, query, num_results=5):
        """Search using Serper.dev API (Google Search API alternative)"""
        if not SERPER_API_KEY:
            return {"error": "Serper API key not configured"}
        
        try:
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": num_results
            }
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            search_results = []
            if 'organic' in data:
                for result in data['organic'][:num_results]:
                    search_results.append({
                        "title": result.get('title', 'No title'),
                        "url": result.get('link', ''),
                        "snippet": result.get('snippet', 'No description')
                    })
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error using Serper API: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}
    
    def search_using_serpapi(self, query, num_results=5):
        """Search using SerpAPI (Alternative)"""
        if not SERP_API_KEY:
            return {"error": "SerpAPI key not configured"}
        
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://serpapi.com/search?q={encoded_query}&api_key={SERP_API_KEY}&num={num_results}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            search_results = []
            if 'organic_results' in data:
                for result in data['organic_results'][:num_results]:
                    search_results.append({
                        "title": result.get('title', 'No title'),
                        "url": result.get('link', ''),
                        "snippet": result.get('snippet', 'No description')
                    })
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error using SerpAPI: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}
    
    def search_fallback(self, query, num_results=5):
        """Fallback search method using direct HTTP requests (less reliable but requires no API key)"""
        try:
            # Encode the search query
            encoded_query = urllib.parse.quote_plus(query)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            }
            
            # Try using DuckDuckGo HTML search (no API key needed)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            response = requests.get(url, headers=headers, timeout=10)
            
            search_results = []
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', class_='result')
                
                for result in results[:num_results]:
                    title_elem = result.find('a', class_='result__a')
                    url_elem = title_elem['href'] if title_elem else ''
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    title = title_elem.text if title_elem else 'No title'
                    snippet = snippet_elem.text if snippet_elem else 'No description'
                    
                    # Clean the URL (DuckDuckGo uses redirects)
                    if url_elem and 'uddg=' in url_elem:
                        parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(url_elem).query)
                        clean_url = parsed_url.get('uddg', [''])[0]
                    else:
                        clean_url = url_elem
                    
                    search_results.append({
                        "title": title,
                        "url": clean_url,
                        "snippet": snippet
                    })
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}
    
    def search_google(self, query, num_results=5):
        """Search using available methods and return search results"""
        # Check if results are already cached
        cache_key = f"search_{query}_{num_results}"
        if cache_key in self.search_results_cache:
            logger.info(f"Returning cached results for query: {query}")
            return self.search_results_cache[cache_key]
        
        # Try each search method in order of preference
        search_results = None
        
        # Method 1: Serper.dev API
        if SERPER_API_KEY:
            search_results = self.search_using_serper(query, num_results)
            if not isinstance(search_results, dict) or "error" not in search_results:
                self.search_results_cache[cache_key] = search_results
                return search_results
        
        # Method 2: SerpAPI
        if SERP_API_KEY:
            search_results = self.search_using_serpapi(query, num_results)
            if not isinstance(search_results, dict) or "error" not in search_results:
                self.search_results_cache[cache_key] = search_results
                return search_results
        
        # Method 3: Fallback method
        search_results = self.search_fallback(query, num_results)
        if not isinstance(search_results, dict) or "error" not in search_results:
            self.search_results_cache[cache_key] = search_results
            return search_results
        
        # If all methods failed
        return {"error": "All search methods failed"}
    
    def fetch_content(self, url):
        """Fetch and extract content from a webpage"""
        # Check if content is already cached
        if url in self.content_cache:
            logger.info(f"Returning cached content for URL: {url}")
            return self.content_cache[url]
        
        try:
            # Use requests for fetch
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                    element.decompose()
                
                # Extract main content (prioritize article content if available)
                main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower()))
                
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    # Fall back to body content
                    text = soup.get_text(separator=' ', strip=True)
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Remove excess whitespace
                text = ' '.join(text.split())
                
                # Cache the content
                self.content_cache[url] = text
                return text
            else:
                return f"Failed to retrieve content: HTTP status {response.status_code}"
        
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return f"Failed to retrieve content: {str(e)}"
    
    def search_and_retrieve(self, query, max_results=3, max_content_per_url=5000):
        """Search for query and retrieve content from top results"""
        search_results = self.search_google(query, num_results=max_results)
        
        if isinstance(search_results, dict) and "error" in search_results:
            return f"Web search error: {search_results['error']}"
        
        # Extract URLs from search results
        urls = [result["url"] for result in search_results if "url" in result and result["url"]]
        
        # Retrieve content from each URL in parallel
        contents = []
        with ThreadPoolExecutor(max_workers=max_results) as executor:
            future_to_url = {executor.submit(self.fetch_content, url): url for url in urls}
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    content = future.result()
                    # Skip empty content
                    if not content or len(content) < 50:
                        continue
                    
                    # Get the title from search results
                    title = next((result["title"] for result in search_results if result["url"] == url), "Unknown Source")
                    
                    # Format content
                    formatted_content = f"Source: {title}\nURL: {url}\n\n"
                    
                    # Truncate content if too long
                    if len(content) > max_content_per_url:
                        content = content[:max_content_per_url] + "... [content truncated]"
                    
                    formatted_content += content
                    contents.append(formatted_content)
                except Exception as e:
                    logger.error(f"Error processing content from {url}: {str(e)}")
        
        # Combine results
        if contents:
            combined_content = f"Web Search Results for '{query}':\n\n" + "\n\n" + "-"*50 + "\n\n".join(contents)
        else:
            combined_content = f"No usable content found for query: '{query}'"
        
        return combined_content
    
    def format_search_results_for_display(self, query):
        """Format search results for display in Streamlit"""
        search_results = self.search_google(query, num_results=5)
        
        if isinstance(search_results, dict) and "error" in search_results:
            return f"Error: {search_results['error']}"
        
        formatted_results = f"### Search Results for: '{query}'\n\n"
        
        for i, result in enumerate(search_results, 1):
            formatted_results += f"**{i}. [{result['title']}]({result['url']})**\n"
            formatted_results += f"{result['snippet']}\n\n"
        
        return formatted_results

# Create a Streamlit component for the web search toggle
def web_search_toggle():
    # Initialize search toggle in session state if not present
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = False
    
    # Initialize WebSearch instance in session state if not present
    if "web_search" not in st.session_state:
        st.session_state.web_search = WebSearch()
    
    # Create toggle with API key status indicator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_enabled = st.toggle(
            "Enable Web Search",
            value=st.session_state.web_search_enabled,
            help="Toggle web search capability for more up-to-date information"
        )
    
    # Show API status in the second column
    with col2:
        if search_enabled:
            api_status = "✅ Ready" if SERPER_API_KEY or SERP_API_KEY else "⚠️ Using fallback"
            st.write(api_status)
    
    # Update session state if toggle value has changed
    if search_enabled != st.session_state.web_search_enabled:
        st.session_state.web_search_enabled = search_enabled
        
        if search_enabled:
            if not (SERPER_API_KEY or SERP_API_KEY):
                st.warning("No API keys configured. Using fallback search method which may be less reliable.")
        
    return search_enabled

def perform_web_search(query):
    """Perform web search if enabled, return results for LLM context"""
    if not st.session_state.get("web_search_enabled", False):
        return "Web search is disabled. Please enable it to search for up-to-date information."
    
    with st.spinner(f"Searching the web for '{query}'..."):
        try:
            # Get search results and content
            results = st.session_state.web_search.search_and_retrieve(query)
            return results
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return f"Web search failed: {str(e)}"

def cleanup_web_search():
    """Clean up web search resources when app is closing"""
    # No active cleanup needed with the new implementation
    pass