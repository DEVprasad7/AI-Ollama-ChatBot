# AI-Gemini-ChatBot
AI-Gemini ChatBot is a powerful chatbot built using Python and Streamlit, powered by Google's Gemini AI model. It processes user input in real time, performs intelligent text analysis, and delivers contextual responses with file upload support.

<h1 align="center">🤖 AI-Gemini ChatBot</h1>

<p align="center">
  An AI chatbot powered by Google Gemini, built with Python, Streamlit, and SQLite3.
</p>

---

## 📸 Preview

<!-- Add your screenshots below -->
<p align="center">
  <img src="images/screenshot1.png" alt="Authentication Page" width="600"/>
  <br/>
  <em>SQLite3-powered authentication screen</em>
</p>

<p align="center">
  <img src="images/screenshot2.png" alt="ChatBot Home UI" width="600"/>
  <br/>
  <em>Streamlit-based UI for chat interaction</em>
</p>

---

## 🚀 Features

- 🔐 User authentication using **SQLite3**
- 🧠 Powered by **Google Gemini AI**
- 💬 Interactive, real-time responses with streaming
- 💻 Clean, modern interface using **Streamlit**
- 📁 File upload support (PDF, DOCX, TXT, CSV, Excel, Images)
- 💾 Conversation history and export/import
- 👤 Multi-user support with isolated conversations
- ✏️ Rename and delete conversations
- 👍👎 Feedback system for responses

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DEVprasad7/AI-Gemini-ChatBot.git
   cd AI-Gemini-ChatBot
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Google Gemini API Setup:
   - Get your API key from: https://makersuite.google.com/app/apikey
   - Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Run the Application:
   ```bash
   streamlit run main.py
   ```

---

## 🎯 Usage

1. **Register/Login**: Create an account or login with existing credentials
2. **Start Chatting**: Ask questions and get AI-powered responses
3. **Upload Files**: Drag and drop files (PDF, DOCX, TXT, CSV, Excel) for analysis
4. **Manage Conversations**: Create, rename, or delete chat conversations
5. **Export/Import**: Backup your conversations as JSON files

---

## 📋 Requirements

- Python 3.8+
- Google Gemini API Key
- Internet connection for API calls

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.