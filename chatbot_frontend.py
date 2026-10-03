import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot_backend import (
    chatbot, retrive_all_thread, generate_title, save_title, load_titles,
    backfill_titles, ingest_document, list_docs,
    set_pinned, load_pinned, delete_thread,
)

st.set_page_config(page_title="AI Chat Assistant", layout="wide")

# ---------- styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    [data-testid="stChatMessage"] code {
        border-radius: 6px; padding: 0.1rem 0.4rem;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%; text-align: left; border-radius: 10px;
    }
    .welcome {text-align: center; padding: 4rem 1rem 2rem 1rem;}
    .welcome h1 {font-size: 2.2rem; margin-bottom: 0.3rem;}
    .welcome p {opacity: 0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["uploader_key"] += 1  # clears the file uploader


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ---------- dialogs (rename / delete) ----------
@st.dialog("Rename chat")
def rename_dialog(thread_id):
    current = st.session_state["thread_titles"].get(thread_id, "")
    new_title = st.text_input("Chat name", value=current, max_chars=60)
    if st.button("Save", type="primary"):
        new_title = new_title.strip()
        if new_title:
            save_title(thread_id, new_title)
            st.session_state["thread_titles"][thread_id] = new_title
            st.rerun()


@st.dialog("Delete chat?")
def delete_dialog(thread_id):
    title = st.session_state["thread_titles"].get(thread_id, "New chat")
    st.write(f"**{title}** and its uploaded documents will be permanently deleted.")
    c1, c2 = st.columns(2)
    if c1.button("Delete", type="primary", use_container_width=True):
        delete_thread(thread_id)
        if thread_id in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].remove(thread_id)
        st.session_state["thread_titles"].pop(thread_id, None)
        st.session_state["pinned"].discard(thread_id)
        if thread_id == st.session_state["thread_id"]:
            reset_chat()
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.rerun()


# ---------- session init ----------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrive_all_thread()
if "thread_titles" not in st.session_state:
    backfill_titles(st.session_state["chat_threads"])
    st.session_state["thread_titles"] = load_titles()
if "pinned" not in st.session_state:
    st.session_state["pinned"] = load_pinned()
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

add_thread(st.session_state["thread_id"])
tid = st.session_state["thread_id"]


def render_chat_row(thread_id, label, is_current, is_pinned):
    col_title, col_menu = st.columns([5, 1])

    if col_title.button(
        label,
        key=f"thread_{thread_id}",
        type="primary" if is_current else "secondary",
    ):
        st.session_state["thread_id"] = thread_id
        temp_messages = []
        for msg in load_conversation(thread_id):
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                temp_messages.append({"role": "assistant", "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.session_state["uploader_key"] += 1
        st.rerun()

    with col_menu.popover("⋮", use_container_width=True):
        if st.button("Unpin" if is_pinned else "Pin", key=f"pin_{thread_id}"):
            set_pinned(thread_id, not is_pinned)
            if is_pinned:
                st.session_state["pinned"].discard(thread_id)
            else:
                st.session_state["pinned"].add(thread_id)
            st.rerun()
        if st.button("Rename", key=f"rename_{thread_id}"):
            st.session_state["pending_dialog"] = ("rename", thread_id)
        if st.button("Delete", key=f"delete_{thread_id}"):
            st.session_state["pending_dialog"] = ("delete", thread_id)


# ---------- sidebar ----------
with st.sidebar:
    st.title("AI Chat")
    if st.button("New Chat", type="primary"):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "Upload to this chat",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
        label_visibility="collapsed",
    )

    already = set(list_docs(tid))
    for f in uploaded or []:
        if f.name in already:
            continue
        with st.spinner(f"Reading {f.name}..."):
            try:
                n = ingest_document(tid, f.getvalue(), f.name)
                if n:
                    st.success(f"{f.name}: {n} chunks indexed")
                else:
                    st.warning(f"{f.name}: no readable text found")
            except Exception as e:
                st.error(f"{f.name}: {e}")

    docs_now = list_docs(tid)
    if docs_now:
        for d in docs_now:
            st.caption(f"`{d}`")
    else:
        st.caption("No documents yet. Upload one and ask questions about it.")

    st.divider()
    st.subheader("Conversations")
    query = st.text_input(
        "Search chats",
        placeholder="Search chats",
        label_visibility="collapsed",
        key="chat_search",
    )

    titles = st.session_state["thread_titles"]
    pinned = st.session_state["pinned"]

    # newest first; hide empty never-used chats except the one currently open
    visible = [
        t for t in st.session_state["chat_threads"][::-1]
        if t in titles or t == tid
    ]
    if query.strip():
        q = query.strip().lower()
        visible = [t for t in visible if q in titles.get(t, "New chat").lower()]

    pinned_list = [t for t in visible if t in pinned]
    recent_list = [t for t in visible if t not in pinned]

    if not visible:
        st.caption("No chats found.")

    if pinned_list:
        st.caption("Pinned")
        for t in pinned_list:
            render_chat_row(t, titles.get(t, "New chat"), t == tid, True)

    if recent_list:
        if pinned_list:
            st.caption("Recent")
        for t in recent_list:
            render_chat_row(t, titles.get(t, "New chat"), t == tid, False)

# open a rename/delete dialog if one was requested from the menu
pending = st.session_state.pop("pending_dialog", None)
if pending:
    kind, target = pending
    if kind == "rename":
        rename_dialog(target)
    else:
        delete_dialog(target)

# ---------- main chat ----------
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="welcome">
            <h1>How can I help you today?</h1>
            <p>Ask me anything, or upload a document in the sidebar and chat with it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    is_new_title = False
    if tid not in st.session_state["thread_titles"]:
        title = generate_title(user_input)
        save_title(tid, title)
        st.session_state["thread_titles"][tid] = title
        is_new_title = True

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": tid},
        "metadata": {"thread_id": tid},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            if metadata["langgraph_node"] == "chat_node"
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    if is_new_title:
        st.rerun()