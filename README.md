# 📄 SmartDocs

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDF documents and ask questions about them — powered by Claude AI and ChromaDB.

## Features

- Upload any PDF and chat with its content
- Retrieval-Augmented Generation (RAG) pipeline
- Vector search with ChromaDB
- Conversation history support
- Clean Streamlit UI

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
                                                           ↓
User Question → Embedding → Vector Search → Relevant Chunks
                                                    ↓
                              Claude AI (claude-haiku) → Answer
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Claude (Anthropic) |
| Vector DB | ChromaDB |
| PDF Parsing | pypdf |
| Language | Python 3.11+ |

## Project Structure

```
SmartDocs/
├── app.py              # Streamlit UI entry point
├── src/
│   ├── pdf_processor.py    # PDF loading and text chunking
│   ├── vectorstore.py      # ChromaDB vector store wrapper
│   └── chat.py             # Claude API integration
├── data/               # Uploaded PDFs (gitignored)
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/KevinFunck/SmartDocs.git
cd SmartDocs
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Get your API key at [console.anthropic.com](https://console.anthropic.com)

**5. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Usage

1. Open the app in your browser
2. Upload a PDF in the sidebar
3. Click **Process PDF**
4. Ask questions in the chat input

## License

MIT
