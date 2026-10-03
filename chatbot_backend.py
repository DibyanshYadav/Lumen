import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b')

# ---------- database ----------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS thread_titles (
    thread_id TEXT PRIMARY KEY, title TEXT, pinned INTEGER DEFAULT 0)""")
try:  # migrate older databases that don't have the pinned column yet
    conn.execute("ALTER TABLE thread_titles ADD COLUMN pinned INTEGER DEFAULT 0")
except sqlite3.OperationalError:
    pass
conn.execute("CREATE TABLE IF NOT EXISTS user_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT UNIQUE)")
conn.execute("""CREATE TABLE IF NOT EXISTS thread_docs (
    thread_id TEXT, filename TEXT, chunks INTEGER,
    PRIMARY KEY (thread_id, filename))""")
conn.commit()

checkpointer = SqliteSaver(conn=conn)


# ---------- titles, pinning, rename ----------
def generate_title(first_message: str) -> str:
    try:
        resp = model.invoke(
            "Write a short 2-5 word title that describes the topic of a chat "
            "starting with the message below. Examples: 'Paneer recipe', "
            "'Fix SqliteSaver error', 'Trip to Goa'. "
            "Reply with only the title, no quotes.\n\n"
            f"Message: {first_message}"
        )
        title = resp.content.strip().strip('"').strip("'")
    except Exception:
        title = ""
    if not title:
        title = " ".join(first_message.split()[:5])
    return title[:50]


def save_title(thread_id: str, title: str):
    """Insert or rename. Keeps the pinned flag untouched."""
    conn.execute(
        "INSERT INTO thread_titles (thread_id, title) VALUES (?, ?) "
        "ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title",
        (thread_id, title),
    )
    conn.commit()


def load_titles() -> dict:
    return dict(conn.execute("SELECT thread_id, title FROM thread_titles").fetchall())


def set_pinned(thread_id: str, pinned: bool):
    conn.execute(
        "UPDATE thread_titles SET pinned = ? WHERE thread_id = ?",
        (1 if pinned else 0, thread_id),
    )
    conn.commit()


def load_pinned() -> set:
    rows = conn.execute("SELECT thread_id FROM thread_titles WHERE pinned = 1").fetchall()
    return {r[0] for r in rows}


# ---------- long-term memory ----------
class MemoryExtract(BaseModel):
    facts: list[str] = Field(
        default_factory=list,
        description="Durable facts the user stated about themselves (name, job, location, "
                    "preferences, projects). Empty list if there are none.",
    )

extractor = model.with_structured_output(MemoryExtract)


def get_memories() -> list[str]:
    return [r[0] for r in conn.execute("SELECT fact FROM user_memory").fetchall()]


def save_memory(fact: str):
    conn.execute("INSERT OR IGNORE INTO user_memory (fact) VALUES (?)", (fact,))
    conn.commit()


# ---------- RAG ----------
VECTOR_DIR = Path("vectorstores")
VECTOR_DIR.mkdir(exist_ok=True)

_embeddings = None
_stores: dict = {}

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


def get_embeddings():
    """Loaded lazily so the app starts fast; downloads the model on first use."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def _store_path(thread_id: str) -> Path:
    return VECTOR_DIR / thread_id


def get_store(thread_id: str):
    if thread_id in _stores:
        return _stores[thread_id]
    path = _store_path(thread_id)
    if path.exists():
        store = FAISS.load_local(
            str(path), get_embeddings(), allow_dangerous_deserialization=True
        )
        _stores[thread_id] = store
        return store
    return None


def list_docs(thread_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT filename FROM thread_docs WHERE thread_id = ?", (thread_id,)
    ).fetchall()
    return [r[0] for r in rows]


def ingest_document(thread_id: str, file_bytes: bytes, filename: str) -> int:
    """Read, chunk, embed and store a document for this chat. Returns chunk count."""
    suffix = Path(filename).suffix.lower()

    if suffix in (".txt", ".md"):
        text = file_bytes.decode("utf-8", errors="ignore")
        docs = [Document(page_content=text, metadata={})]
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(tmp_path).load()
            elif suffix == ".docx":
                docs = Docx2txtLoader(tmp_path).load()
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        finally:
            os.remove(tmp_path)

    for d in docs:
        d.metadata["source"] = filename

    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return 0

    store = get_store(thread_id)
    if store is None:
        store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)
    _stores[thread_id] = store
    store.save_local(str(_store_path(thread_id)))

    conn.execute(
        "INSERT OR REPLACE INTO thread_docs VALUES (?, ?, ?)",
        (thread_id, filename, len(chunks)),
    )
    conn.commit()
    return len(chunks)


def retrieve_context(thread_id: str, query: str, k: int = 5) -> str:
    store = get_store(thread_id)
    if store is None:
        return ""
    results = store.similarity_search(query, k=k)
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if page is not None else source
        parts.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------- prompts ----------
BASE_PROMPT = """You are Lumen, a friendly, sharp AI assistant in a chat app. Always format answers in clean Markdown so they are easy to scan:

- Start with a direct answer or a one-line summary, then add detail.
- Use `##` headings only for longer answers. Keep short answers short.
- **Bold** the key terms, important points, and warnings.
- Put file names, commands, paths, variable names and function names in `inline code`.
- Use bullet points for lists, numbered lists for steps, and tables for comparisons.
- Put all code in fenced code blocks with the language tag (```python, ```bash).
- Be conversational and warm, not stiff. No filler like "Certainly!" or "Great question!".
- If you are unsure, say so instead of guessing."""

RAG_PROMPT = """

The user has uploaded documents to this chat. Relevant excerpts are below.
- Answer from these excerpts when the question relates to the documents.
- Cite where each fact came from using the file name in `code` style and the page, e.g. (`report.pdf`, p. 3).
- If the excerpts do not contain the answer, say clearly: "I couldn't find this in your documents", and then you may answer from general knowledge, marking it as such.
- Never invent content that is not in the excerpts.

Document excerpts:
{context}"""


# ---------- graph ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    system = BASE_PROMPT

    memories = get_memories()
    if memories:
        system += "\n\nKnown facts about the user:\n" + "\n".join(f"- {m}" for m in memories)

    last_user = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    context = retrieve_context(thread_id, last_user.content)
    if context:
        system += RAG_PROMPT.format(context=context)

    response = model.invoke([SystemMessage(content=system)] + state["messages"])
    return {"messages": [response]}


def remember_node(state: ChatState):
    last_user = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    result = extractor.invoke(
        "Extract durable facts the user stated about themselves in this message. "
        f"Return an empty list if there are none.\n\nMessage: {last_user.content}"
    )
    for fact in result.facts:
        save_memory(fact)
    return {}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("remember_node", remember_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", "remember_node")
graph.add_edge("remember_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------- helpers that need the compiled graph ----------
def retrive_all_thread():
    """Thread ids ordered oldest -> newest (the frontend reverses it for display)."""
    newest_first = []
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if tid not in newest_first:
            newest_first.append(tid)
    return list(reversed(newest_first))


def backfill_titles(thread_ids):
    existing = load_titles()
    for tid in thread_ids:
        if tid in existing:
            continue
        state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
        msgs = state.values.get("messages", [])
        first = next((m for m in msgs if isinstance(m, HumanMessage)), None)
        if first:
            save_title(tid, generate_title(first.content))


def delete_thread(thread_id: str):
    """Permanently remove a chat: messages, title, documents and vector index."""
    for sql in (
        "DELETE FROM checkpoints WHERE thread_id = ?",
        "DELETE FROM writes WHERE thread_id = ?",
        "DELETE FROM thread_titles WHERE thread_id = ?",
        "DELETE FROM thread_docs WHERE thread_id = ?",
    ):
        try:
            conn.execute(sql, (thread_id,))
        except sqlite3.OperationalError:
            pass  # table doesn't exist yet
    conn.commit()
    _stores.pop(thread_id, None)
    shutil.rmtree(_store_path(thread_id), ignore_errors=True)