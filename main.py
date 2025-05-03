import streamlit as st
import time
import os
import re
import tempfile
import json
import sqlite3
import hashlib
import uuid
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from web_search import web_search_toggle, perform_web_search, cleanup_web_search
import pandas as pd
import PyPDF2
import docx2txt





# Initialize the database
def init_db():
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Hash password with salt
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Verify user credentials
def verify_user(username, password):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        user_id, stored_hash = result
        if stored_hash == hash_password(password):
            return user_id
    return None

# Register new user
def register_user(full_name, username, password):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        c.execute(
            'INSERT INTO users (id, full_name, username, password_hash) VALUES (?, ?, ?, ?)',
            (user_id, full_name, username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        conn.close()
        return False

def needs_web_search(query):
    """Determine if a query likely needs up-to-date information from the web"""
    # Define patterns that suggest a need for current information
    current_info_patterns = [
        query,
        r'latest', r'recent', r'current', r'news', r'today', 
        r'update', r'what is happening', r'how to', r'who is',
        r'weather', r'stock', r'price', r'event', r'information about',r'find out',
        r'find', r'latest news', r'latest update', r'current events',r'time',
        r'current time', r'current date', r'latest trends', r'latest developments',r'latest information',
        r"Many", r"most", r"all", r"some", r"few", r"several",
        r"today", r"yesterday", r"tomorrow", r"this week", r"this month",
        r"New", r"Old", r"original",r"today's", r"yesterday's", r"tomorrow's",
        r"today's", r"this week's", r"this month's", r"this year's", r"How", r"last"
    ]
    
    # Check for date or time references
    date_patterns = [
        r'\b202[0-9]\b',  # Years 2020-2029
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(today|yesterday|tomorrow|last week|next week|this month|last month|this year)\b'
    ]
    
    # Check if query matches any patterns
    query_lower = query.lower()
    for pattern in current_info_patterns + date_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def main():
    # Initialize database
    init_db()
    
    # Initialize session state for authentication
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    # Set up the sidebar
    st.sidebar.title("Conversations")
    
    # Authentication section at the bottom of sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("Authentication", expanded=not st.session_state.user_id):
        if st.session_state.user_id:
            # User is logged in, show logout button
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.conversations = {}
                st.rerun()
        else:
            # User is not logged in, show login and registration forms
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    login_button = st.form_submit_button("Login")
                    
                    if login_button:
                        user_id = verify_user(username, password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
            
            with tab2:
                with st.form("register_form"):
                    full_name = st.text_input("Full Name")
                    new_username = st.text_input("Username", key="reg_username")
                    new_password = st.text_input("Password", type="password", key="reg_password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    register_button = st.form_submit_button("Register")
                    
                    if register_button:
                        if not all([full_name, new_username, new_password]):
                            st.error("All fields are required")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            success = register_user(full_name, new_username, new_password)
                            if success:
                                st.success("Registration successful! You can now login.")
                            else:
                                st.error("Username already exists")
    
    # Check if user is authenticated before showing the main app
    if not st.session_state.user_id:
        st.title("Ollama-LLM AI")
        st.info("Please login or register to use the application")
        return
    
    # Initialize storage for conversations
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    if "current_conversation_id" not in st.session_state:
        # Create a default conversation on first run
        new_conversation_id = f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.conversations[new_conversation_id] = {
            "title": "New Conversation",
            "history": [],
            "file_content": "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user_id": st.session_state.user_id  # Associate conversation with user
        }
        st.session_state.current_conversation_id = new_conversation_id
    
    # Create a new conversation button
    if st.sidebar.button("New Chat 💬"):
        new_conversation_id = f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.conversations[new_conversation_id] = {
            "title": "New Conversation",
            "history": [],
            "file_content": "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user_id": st.session_state.user_id  # Associate conversation with user
        }
        st.session_state.current_conversation_id = new_conversation_id
        st.rerun()
    
    # Display saved conversations in sidebar
    st.sidebar.markdown("### Saved Chats")
    
    # Filter conversations for current user
    user_conversations = {
        conv_id: conv_data for conv_id, conv_data in st.session_state.conversations.items()
        if conv_data.get("user_id") == st.session_state.user_id
    }
    
    # Sort conversations by creation time (newest first)
    sorted_conversations = sorted(
        user_conversations.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True
    )
    
    # Display conversation list in sidebar
    for conv_id, conv_data in sorted_conversations:
        # Get the conversation title or first few words of first message
        if conv_data["history"]:
            first_msg = conv_data["history"][0]["content"]
            display_title = conv_data.get("title", first_msg[:20] + "...")
        else:
            display_title = conv_data.get("title", "Empty Conversation")
        
        # Create a container for each conversation with title and buttons side by side
        col1, col2, col3 = st.sidebar.columns([5, 1, 1])
        
        # Highlight current conversation
        if conv_id == st.session_state.current_conversation_id:
            col1.markdown(f"**→ {display_title}**")
        else:
            # Make conversation title clickable
            if col1.button(display_title, key=f"select_{conv_id}"):
                st.session_state.current_conversation_id = conv_id
                st.rerun()
        
        # Add rename button
        if col2.button("✏️", key=f"rename_{conv_id}", help="Rename conversation"):
            st.session_state.renaming_conversation = conv_id
            st.rerun()
        
        # Add delete button
        if col3.button("🗑️", key=f"delete_{conv_id}", help="Delete conversation"):
            st.session_state.deleting_conversation = conv_id
            st.rerun()
    
    # Handle conversation renaming
    if hasattr(st.session_state, "renaming_conversation"):
        conv_id = st.session_state.renaming_conversation
        current_title = st.session_state.conversations[conv_id].get("title", "New Conversation")
        
        with st.sidebar.form(key="rename_form"):
            new_title = st.text_input("New title:", value=current_title)
            col1, col2 = st.columns(2)
            
            if col1.form_submit_button("Save"):
                st.session_state.conversations[conv_id]["title"] = new_title
                delattr(st.session_state, "renaming_conversation")
                st.rerun()
                
            if col2.form_submit_button("Cancel"):
                delattr(st.session_state, "renaming_conversation")
                st.rerun()
    
    # Handle conversation deletion
    if hasattr(st.session_state, "deleting_conversation"):
        conv_id = st.session_state.deleting_conversation
        
        with st.sidebar.form(key="delete_form"):
            st.write(f"Delete this conversation?")
            col1, col2 = st.columns(2)
            
            if col1.form_submit_button("Confirm"):
                del st.session_state.conversations[conv_id]
                
                # If we're deleting the current conversation, switch to another one
                if conv_id == st.session_state.current_conversation_id:
                    user_conversations = {
                        k: v for k, v in st.session_state.conversations.items()
                        if v.get("user_id") == st.session_state.user_id
                    }
                    if user_conversations:
                        st.session_state.current_conversation_id = next(iter(user_conversations))
                    else:
                        # Create a new conversation if all were deleted
                        new_id = f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.conversations[new_id] = {
                            "title": "New Conversation",
                            "history": [],
                            "file_content": "",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "user_id": st.session_state.user_id
                        }
                        st.session_state.current_conversation_id = new_id
                
                delattr(st.session_state, "deleting_conversation")
                st.rerun()
                
            if col2.form_submit_button("Cancel"):
                delattr(st.session_state, "deleting_conversation")
                st.rerun()
    
    # Add a separator in sidebar
    st.sidebar.markdown("---")
    
    # Export/Import functionality
    with st.sidebar.expander("Export/Import Chats"):
        # Export functionality
        if st.button("Export All Conversations"):
            # Convert datetime objects to strings for JSON serialization
            # Only export user's conversations
            user_conversations = {
                k: v for k, v in st.session_state.conversations.items()
                if v.get("user_id") == st.session_state.user_id
            }
            export_json = json.dumps(user_conversations, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=export_json,
                file_name=f"ollama_chats_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        # Import functionality
        uploaded_file = st.file_uploader("Import Conversations", type=["json"])
        if uploaded_file is not None:
            try:
                imported_data = json.loads(uploaded_file.getvalue().decode())
                
                # Assign current user_id to imported conversations
                for conv_id, conv_data in imported_data.items():
                    conv_data["user_id"] = st.session_state.user_id
                
                st.session_state.conversations.update(imported_data)
                st.success("Conversations imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing conversations: {str(e)}")
    
    # Access the current conversation
    current_conversation = st.session_state.conversations[st.session_state.current_conversation_id]
    current_history = current_conversation["history"]
    current_file_content = current_conversation.get("file_content", "")
    
    # Set up the main title of the app
    st.title("Ollama-LLM AI")

    # Toggle for web search - updated to use the new implementation
    search_enabled = web_search_toggle()
    
    # Initialize the LLM model
    model = OllamaLLM(model="llama3.2")
    
    # MODIFIED: Different prompt templates based on whether web search is enabled
    standard_template = """
        Answer the questions below.
        You are a helpful assistant. You can answer questions based on the conversation history, any file content provided, and web search results when available.
        Please provide a concise and accurate answer.

        Here is the conversation history: {history}

        File content (if any): {file_content}

        Web search results (if any): {web_search_results}

        Question: {question}

        Answer:
        """
    
    web_search_template = """
        You are an AI assistant tasked with answering questions strictly based on the provided web search results.  
        You may use your own reasoning to interpret and organize the information, but you must not rely on prior knowledge beyond the search results.

        Instructions:
        - Use only the web search results to formulate your answer.
        - If the web search results do not provide sufficient information to answer the question, respond with:  
          "I don't have enough information from the web search results to answer this question."
        - Do not invent, assume, or supplement missing information using your own knowledge.
        - Only use the conversation history for context and continuity; do not derive factual answers from it.
        - Refer to any attached file content **only if** it is directly relevant to the question.
        - try to provide a concise answer with minimal elaboration.

        Inputs:
        - Conversation History (for context only): {history}
        - File Content (if provided and relevant): {file_content}
        - Web Search Results (primary information source): {web_search_results}
        - User Question: {question}

        Important:
        - Base your response solely on information that is relevant to the question from the web search results.
        - Maintain accuracy, clarity, and conciseness in your answers.

        Response:

                """

    def process_file(uploaded_file):
    
    
        file_content = ""
    
        # Create a temporary file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        try:
        # Handle different file types using appropriate loaders
            file_extension = uploaded_file.name.split('.')[-1].lower()
        
            if file_extension == 'txt':
            # Process text files using pandas
                with open(temp_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
        
            elif file_extension == 'pdf':
            # Process PDF files using PyPDF2
                pdf_text = []
                with open(temp_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pdf_text.append(page.extract_text())
                file_content = '\n'.join(pdf_text)
        
            elif file_extension in ['docx', 'doc']:
            # Process Word documents using docx2txt
                if file_extension == 'docx':
                    file_content = docx2txt.process(temp_path)
                else:
                    file_content = f"Note: Direct .doc support requires additional libraries. Consider converting to .docx"
        
            elif file_extension == 'csv':
            # Process CSV files - create a summary and preview
                df = pd.read_csv(temp_path)
                file_content = f"CSV Summary:\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n"
                file_content += df.head(5).to_string()
        
            elif file_extension in ['jpg', 'jpeg', 'png']:
            # Just acknowledge image files (no content extraction)
                file_content = f"[Image file uploaded: {uploaded_file.name}]"
            
            elif file_extension in ['xlsx', 'xls']:
            # Process Excel files
                df = pd.read_excel(temp_path)
                file_content = f"Excel Summary:\nSheets: {', '.join(pd.ExcelFile(temp_path).sheet_names)}\n"
                file_content += f"First sheet data:\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n"
                file_content += df.head(5).to_string()
            
            else:
                file_content = f"Unsupported file type: {file_extension}"
            
        except Exception as e:
            file_content = f"Error processing file: {str(e)}"
    
        finally:
        # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
        return file_content

    def chat_stream(user_input, file_content=""):
        """Generate streaming response from the AI"""
        # Format history for context
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in current_history])
    
        # Check if web search is enabled and needed for this query
        web_search_results = ""
        use_web_search_template = False
        
        if st.session_state.get("web_search_enabled", False):
            # Always perform web search when the feature is enabled
            with st.spinner("Searching the web for relevant information..."):
                web_search_results = perform_web_search(user_input)
                # Use the web search template only if we got actual results
                if web_search_results and "Web search is disabled" not in web_search_results:
                    use_web_search_template = True
    
        # Choose the appropriate template based on web search results
        if use_web_search_template:
            prompt = ChatPromptTemplate.from_template(web_search_template)
        else:
            prompt = ChatPromptTemplate.from_template(standard_template)
            
        # Create the chain with the selected template
        chain = prompt | model
    
        # Get response from the LLM chain
        response = chain.invoke({
            "history": history_text, 
            "question": user_input,
            "file_content": file_content,
            "web_search_results": web_search_results
        })
    
        # Stream the response character by character for a better UX
        full_response = ""
        for char in response:
            full_response += char
            yield char
            time.sleep(0.02)  # Small delay for streaming effect
    
        # Add the response to history after streaming is complete
        current_history.append({"role": "assistant", "content": full_response})
    
        # Update conversation title if this is the first exchange
        if len(current_history) == 2:  # After first Q&A
            # Use the first few words of the user's first message as the title
            first_msg = current_history[0]["content"]
            # truncated_msg = first_msg[:30] + ("..." if len(first_msg) > 30 else "")
            current_conversation["title"] = first_msg[:20] + "..."

    def save_feedback(index):
        """Save user feedback (thumbs up/down) to the message history"""
        current_history[index]["feedback"] = st.session_state[f"feedback_{index}"]

    # Display chat history
    for i, message in enumerate(current_history):
        with st.chat_message(message["role"]):
            # Show the message content
            st.write(message["content"])
            
            # Display the file if present in the message
            if "file" in message:
                if any(message["file"].name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    # Display images directly
                    st.image(message["file"])
                else:
                    # Add download button for other file types
                    st.download_button(
                        label=f"Download {message['file'].name}",
                        data=message["file"],
                        file_name=message["file"].name,
                        mime="application/octet-stream"
                    )
            
            # Add feedback option for assistant messages
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                st.session_state[f"feedback_{i}"] = feedback
                st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    disabled=feedback is not None,
                    on_change=save_feedback,
                    args=[i],
                )

    # Chat input with file upload support
    prompt = st.chat_input(
        "Ask a question and/or upload a file",
        accept_file=True,
        file_type=["pdf", "txt", "docx", "csv", "jpg", "jpeg", "png"]
    )
    
    # Process user input when submitted
    if prompt:
        user_message = ""
        file_content = ""
        uploaded_file = None
        
        # Extract text and files from the prompt
        if hasattr(prompt, "text") and prompt.text:
            user_message = prompt.text
            
        if hasattr(prompt, "__getitem__") and "files" in prompt and prompt["files"]:
            uploaded_file = prompt["files"][0]
            file_content = process_file(uploaded_file)
            current_conversation["file_content"] = file_content
            
        # Display user message in the chat
        with st.chat_message("user"):
            if user_message:
                st.write(user_message)
            if uploaded_file:
                if any(uploaded_file.name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    st.image(uploaded_file)
                else:
                    st.write(f"[Uploaded file: {uploaded_file.name}]")
        
        # Add user message to chat history
        history_entry = {"role": "user", "content": user_message or f"[Uploaded file: {uploaded_file.name}]"}
        if uploaded_file:
            # Note: We can't store the actual file object in session state because it's not serializable
            # Instead, we'll just note that a file was uploaded and process its content
            history_entry["file_reference"] = uploaded_file.name
        current_history.append(history_entry)
        
        # Generate and display AI response
    
        if user_message or file_content:
            with st.container(border=True):
                with st.chat_message("assistant"):
                # Show spinner while waiting for response to start
                    with st.spinner("Thinking...", show_time=True):
                    # Get the first token from the model to end the spinner
                    # This makes a small initial request to start the generation
                        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in current_history])
                        response_generator = chat_stream(user_message, current_conversation["file_content"])
                    
                    # Get first token (will trigger model loading and initial thinking)
                        try:
                            first_token = next(response_generator)
                        except StopIteration:
                            first_token = ""
                
                # Now display the response with the first token already retrieved
                    response_placeholder = st.empty()
                    full_response = first_token  # Start with the first token we already got
                    response_placeholder.markdown(full_response)
                
                # Continue streaming the rest of the response
                    try:
                        for chunk in response_generator:
                            full_response += chunk
                            response_placeholder.markdown(full_response)
                    except StopIteration:
                        pass
                
                # Add feedback widget after response is complete
                feedback_key = f"feedback_{len(current_history) - 1}"
                st.feedback(
                    "thumbs",
                    key=feedback_key,
                    on_change=save_feedback,
                    args=[len(current_history) - 1],
                )

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_web_search()