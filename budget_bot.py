from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import fitz  # pymupdf
import os

load_dotenv()

# --- Step 1: Load and Process PDF ---
def load_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def build_knowledge_base(pdf_path: str):
    """Build FAISS vector store from PDF."""
    print(f"Loading PDF: {pdf_path}")
    raw_text = load_pdf(pdf_path)
    
    if not raw_text.strip():
        raise ValueError("PDF appears to be empty or unreadable")
    
    print(f"Extracted {len(raw_text)} characters from PDF")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.create_documents([raw_text])
    print(f"Created {len(chunks)} chunks")
    
    # Build FAISS index
    print("Building vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Knowledge base ready.\n")
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Step 2: Build RAG Tool ---
def create_pdf_tool(retriever, pdf_name: str):
    @tool(description=f"Search the {pdf_name} document for relevant information. "
                       "Use this tool when the user asks any question about the document content.")
    def search_document(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document."
        result = ""
        for i, doc in enumerate(docs):
            result += f"Section {i+1}:\n{doc.page_content}\n\n"
        return result.strip()
    
    return search_document
# --- Step 3: Build Agent ---
def build_agent(pdf_path: str):
    pdf_name = os.path.basename(pdf_path)
    retriever = build_knowledge_base(pdf_path)
    pdf_tool = create_pdf_tool(retriever, pdf_name)
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    system_prompt = f"""You are a helpful document assistant. You have access to the contents of '{pdf_name}'.
    When the user asks questions about the document, always use the search_document tool to find relevant information.
    Be accurate and cite what you find. If information is not in the document, say so clearly."""
    
    agent = create_react_agent(
        llm,
        tools=[pdf_tool],
        prompt=system_prompt
    )
    
    return agent

# --- Step 4: Chat with Memory ---
conversation_history = []

def chat(agent, user_input: str):
    global conversation_history
    
    conversation_history.append(HumanMessage(content=user_input))
    
    print(f"You: {user_input}")
    print(f"Bot: ", end="", flush=True)
    
    final_content = ""
    
    for chunk in agent.stream(
        {"messages": conversation_history},
        stream_mode="messages"
    ):
        message_chunk, metadata = chunk
        if (metadata.get("langgraph_node") == "agent" and
            hasattr(message_chunk, "content") and
            message_chunk.content and
            message_chunk.content not in final_content):
            print(message_chunk.content, end="", flush=True)
            final_content += message_chunk.content
    
    print("\n")
    conversation_history.append(AIMessage(content=final_content))

# --- Step 5: Main ---
if __name__ == "__main__":
    # Change this to your PDF path
    PDF_PATH = r"S:\AIProjects\day14\Union_Budget_Analysis-2026-27.pdf"
    
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}")
        print("Please add a PDF file and update PDF_PATH")
        exit()
    
    agent = build_agent(PDF_PATH)
    
    print("PDF Bot ready. Type 'quit' to exit.\n")
    print("="*50)
    
    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Jai Shri Jagannath! 🙏")
            break
        chat(agent, user_input)