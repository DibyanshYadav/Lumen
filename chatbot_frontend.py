import hashlib
import re
import time
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

import chatbot_backend as be
from tools import TOOL_LABELS

st.set_page_config(page_title="Lumen", layout="wide")

# ---------- styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {
        border-radius: 14px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    }
    [data-testid="stChatMessage"] code {border-radius: 6px; padding: 0.1rem 0.4rem;}
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] [data-testid="stPopover"] button {
        width: 100%; text-align: left; border-radius: 10px;
    }
    .stFormSubmitButton button {width: 100%;}
    .welcome {text-align: center; padding: 3rem 1rem 1.5rem 1rem;}
    .welcome h1 {font-size: 2.2rem; margin-bottom: 0.3rem;}
    .welcome p {opacity: 0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_LABEL_BY_ID = {v: k.split(" (")[0] for k, v in be.MODEL_OPTIONS.items()}
TOOL_NAMES = {
    "web_search": "web search",
    "read_url": "page reader",
    "calculator": "calculator",
    "current_datetime": "clock",
}


# ---------- small helpers ----------
def new_thread_id() -> str:
    return str(uuid.uuid4())


def msg_key(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def friendly_error(e: Exception) -> str:
    text = str(e)
    low = text.lower()
    if "429" in text or "rate limit" in low:
        return "Groq's rate limit was reached. Wait a moment, then retry."
    if "401" in text or "invalid api key" in low or "api key" in low:
        return "The Groq API key is missing or invalid. Check GROQ_API_KEY."
    if "timeout" in low or "timed out" in low:
        return "The model took too long to respond. Please retry."
    return f"Something went wrong: {text[:200]}"


def reset_chat():
    st.session_state["thread_id"] = new_thread_id()
    st.session_state["message_history"] = []
    st.session_state["last_error"] = None
    st.session_state["uploader_key"] += 1
    st.session_state["image_key"] += 1


def open_chat(thread_id: str):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = be.load_history(st.session_state["user"], thread_id)
    st.session_state["last_error"] = None
    st.session_state["uploader_key"] += 1
    st.session_state["image_key"] += 1


def export_markdown(title: str, history: list) -> str:
    lines = [f"# {title}", ""]
    for m in history:
        lines += [f"**{'You' if m['role'] == 'user' else 'Lumen'}:**", "", m["content"], ""]
    return "\n".join(lines)


# ---------- login ----------
def auth_screen():
    st.markdown(
        '<div class="welcome"><h1>Lumen</h1>'
        "<p>Chat with your documents. Lumen remembers you.</p></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("signin"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary"):
                    ok, msg = be.verify_user(u, p)
                    if ok:
                        st.session_state["user"] = be.normalize_username(u)
                        st.rerun()
                    else:
                        st.error(msg)
        with tab_up:
            with st.form("signup"):
                u2 = st.text_input("Choose a username")
                p2 = st.text_input("Choose a password (8+ characters)", type="password")
                p3 = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", type="primary"):
                    if p2 != p3:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = be.create_user(u2, p2)
                        if ok:
                            st.session_state["user"] = be.normalize_username(u2)
                            st.rerun()
                        else:
                            st.error(msg)
        st.caption("Your chats, documents and memory are private to your account.")


if "user" not in st.session_state:
    auth_screen()
    st.stop()

# ---------- session init ----------
user = st.session_state["user"]
defaults = {
    "thread_id": new_thread_id(),
    "message_history": [],
    "uploader_key": 0,
    "image_key": 0,
    "audio_key": 0,
    "last_error": None,
    "last_audio_hash": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
tid = st.session_state["thread_id"]
remaining = be.remaining_today(user)


# ---------- dialogs ----------
@st.dialog("Rename chat")
def rename_dialog(thread_id):
    current = be.get_title(user, thread_id) or ""
    new_title = st.text_input("Chat name", value=current, max_chars=60)
    if st.button("Save", type="primary"):
        if new_title.strip():
            be.rename_thread(user, thread_id, new_title)
            st.rerun()


@st.dialog("Delete chat?")
def delete_dialog(thread_id):
    title = be.get_title(user, thread_id) or "This chat"
    st.write(f"**{title}** and its uploaded documents will be permanently deleted.")
    c1, c2 = st.columns(2)
    if c1.button("Delete", type="primary"):
        be.delete_thread(user, thread_id)
        if thread_id == st.session_state["thread_id"]:
            reset_chat()
        st.rerun()
    if c2.button("Cancel"):
        st.rerun()


@st.dialog("Edit your message")
def edit_dialog(text):
    new_text = st.text_area("Message", value=text, height=140)
    if st.button("Resend", type="primary") and new_text.strip():
        st.session_state["action"] = ("edit", new_text.strip())
        st.rerun()


@st.dialog("What Lumen remembers about you")
def memory_dialog():
    memories = be.list_memories(user)
    if not memories:
        st.caption("Nothing yet. Lumen saves facts you share about yourself, like your name or projects.")
        return
    for m in memories:
        c1, c2 = st.columns([5, 1])
        c1.write(m["fact"])
        if c2.button("Delete", key=f"mem_{m['id']}"):
            be.delete_memory(user, m["id"])
            st.session_state["pending_dialog"] = ("memory", None)
            st.rerun()
    st.divider()
    if st.button("Clear all memory"):
        be.clear_memories(user)
        st.session_state["pending_dialog"] = ("memory", None)
        st.rerun()


# ---------- regenerate / edit (rewinds the saved chat before anything renders) ----------
resend = None
action = st.session_state.pop("action", None)
if action:
    if remaining <= 0:
        st.session_state["last_error"] = "You have reached today's message limit."
    else:
        info = be.rewind_to_last_user(user, tid)
        history = st.session_state["message_history"]
        user_positions = [i for i, m in enumerate(history) if m["role"] == "user"]
        if info and user_positions:
            st.session_state["message_history"] = history[: user_positions[-1]]
            resend = {
                "text": action[1] if action[0] == "edit" else info["text"],
                "image": info["image"],
            }

# ---------- sidebar ----------
with st.sidebar:
    st.title("Lumen")
    st.caption(f"Signed in as **{user}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

    if st.button("New Chat", type="primary"):
        reset_chat()
        st.rerun()

    model_label = st.selectbox("Model", list(be.MODEL_OPTIONS), key="model_label")
    model_id = be.MODEL_OPTIONS[model_label]

    st.divider()
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "Upload to this chat",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
        label_visibility="collapsed",
    )
    already = set(be.list_docs(user, tid))
    for f in uploaded or []:
        if f.name in already:
            continue
        with st.spinner(f"Reading {f.name}..."):
            try:
                n = be.ingest_document(user, tid, f.getvalue(), f.name)
                if n:
                    st.success(f"{f.name}: {n} chunks indexed")
                else:
                    st.warning(f"{f.name}: no readable text found")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    docs_now = be.list_docs(user, tid)
    if docs_now:
        for d in docs_now:
            st.caption(f"`{d}`")
    else:
        st.caption("No documents yet. Upload one and ask questions about it.")

    st.divider()
    st.subheader("Conversations")
    query = st.text_input(
        "Search chats", placeholder="Search chats", label_visibility="collapsed", key="chat_search"
    )
    threads = be.list_threads(user)
    if query.strip():
        q = query.strip().lower()
        threads = [t for t in threads if q in t["title"].lower()]
    if not threads:
        st.caption("No chats found." if query.strip() else "No chats yet. Say hello to start one.")

    def render_chat_row(t):
        col_title, col_menu = st.columns([5, 1])
        if col_title.button(
            t["title"],
            key=f"thread_{t['id']}",
            type="primary" if t["id"] == tid else "secondary",
        ):
            open_chat(t["id"])
            st.rerun()
        with col_menu.popover("⋮"):
            if st.button("Unpin" if t["pinned"] else "Pin", key=f"pin_{t['id']}"):
                be.set_pinned(user, t["id"], not t["pinned"])
                st.rerun()
            if st.button("Rename", key=f"rename_{t['id']}"):
                st.session_state["pending_dialog"] = ("rename", t["id"])
            if st.button("Delete", key=f"delete_{t['id']}"):
                st.session_state["pending_dialog"] = ("delete", t["id"])

    pinned_threads = [t for t in threads if t["pinned"]]
    recent_threads = [t for t in threads if not t["pinned"]]
    if pinned_threads:
        st.caption("Pinned")
        for t in pinned_threads:
            render_chat_row(t)
    if recent_threads:
        if pinned_threads:
            st.caption("Recent")
        for t in recent_threads:
            render_chat_row(t)

    st.divider()
    if st.button("Manage memory"):
        st.session_state["pending_dialog"] = ("memory", None)
    if st.session_state["message_history"]:
        title_now = be.get_title(user, tid) or "Chat"
        st.download_button(
            "Export chat (Markdown)",
            data=export_markdown(title_now, st.session_state["message_history"]),
            file_name=re.sub(r"[^a-z0-9]+", "-", title_now.lower()).strip("-") + ".md",
            mime="text/markdown",
        )
    st.caption(f"Messages left today: {remaining}")

# ---------- main: welcome, history ----------
welcome_slot = st.empty()
history = st.session_state["message_history"]
if not history:
    with welcome_slot.container():
        st.markdown(
            '<div class="welcome"><h1>Lumen</h1>'
            "<p>Chat with your documents. Lumen remembers you.</p></div>",
            unsafe_allow_html=True,
        )
        if docs_now:
            prompts = [
                "Summarize my document",
                "List the key points of my document",
                "What are the main conclusions?",
                "Explain the hardest part in simple terms",
            ]
        else:
            prompts = [
                "Explain RAG in simple terms",
                "What is in the news today?",
                "Help me plan a weekly study schedule",
                "What is 18% of 2,450?",
            ]
        cols = st.columns(2)
        for i, p in enumerate(prompts):
            if cols[i % 2].button(p, key=f"suggest_{i}"):
                st.session_state["pending_prompt"] = p

feedback = be.load_feedback(user, tid)
user_idx = [i for i, m in enumerate(history) if m["role"] == "user"]
last_user_idx = user_idx[-1] if user_idx else None

for i, m in enumerate(history):
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=260)
        st.markdown(m["content"])

        if m["role"] == "user" and i == last_user_idx:
            if st.button("Edit", key=f"edit_{tid}_{i}"):
                st.session_state["pending_dialog"] = ("edit", m["content"])

        if m["role"] == "assistant":
            lumen = m.get("lumen") or {}
            sources = lumen.get("sources") or []
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        where = f", page {s['page']}" if s.get("page") else ""
                        st.markdown(f"**`{s['file']}`**{where}")
                        st.caption(s["snippet"])
            bits = []
            if lumen.get("model"):
                bits.append(MODEL_LABEL_BY_ID.get(lumen["model"], lumen["model"]))
            if lumen.get("tools"):
                bits.append("Used " + ", ".join(TOOL_NAMES.get(t, t) for t in lumen["tools"]))
            stats = m.get("stats") or {}
            if stats.get("seconds"):
                bits.append(f"{stats['seconds']:.1f}s")
            if stats.get("tokens"):
                bits.append(f"{stats['tokens']} tokens")
            if bits:
                st.caption(" | ".join(bits))

            key = msg_key(m["content"])
            rating = feedback.get(key)
            c_copy, c_up, c_down, c_regen, _ = st.columns([1, 1.2, 1.5, 1.4, 3])
            with c_copy.popover("Copy"):
                st.code(m["content"], language="markdown")
            if c_up.button("Helpful", key=f"up_{tid}_{i}", type="primary" if rating == 1 else "secondary"):
                be.save_feedback(user, tid, key, 1)
                st.rerun()
            if c_down.button("Not helpful", key=f"down_{tid}_{i}", type="primary" if rating == -1 else "secondary"):
                be.save_feedback(user, tid, key, -1)
                st.rerun()
            if i == len(history) - 1 and c_regen.button("Regenerate", key=f"regen_{tid}_{i}"):
                st.session_state["action"] = ("regenerate",)
                st.rerun()

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
    if st.button("Retry", key="retry"):
        st.session_state["action"] = ("regenerate",)
        st.session_state["last_error"] = None
        st.rerun()

# ---------- attachments: image and voice ----------
voice_text = None
with st.expander("Attach an image or record a voice message"):
    attached = st.file_uploader(
        "Image (sent with your next message)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"image_{st.session_state['image_key']}",
    )
    audio = st.audio_input("Voice message", key=f"audio_{st.session_state['audio_key']}")
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if digest != st.session_state["last_audio_hash"]:
            st.session_state["last_audio_hash"] = digest
            with st.spinner("Transcribing..."):
                try:
                    voice_text = be.transcribe_audio(data) or None
                except Exception as e:
                    st.warning(f"Could not transcribe the audio: {friendly_error(e)}")
            if voice_text is None:
                st.warning("No speech was detected.")
            st.session_state["audio_key"] += 1

# ---------- dialogs requested by buttons above ----------
pending = st.session_state.pop("pending_dialog", None)
if pending:
    kind, target = pending
    if kind == "rename":
        rename_dialog(target)
    elif kind == "delete":
        delete_dialog(target)
    elif kind == "edit":
        edit_dialog(target)
    elif kind == "memory":
        memory_dialog()


# ---------- sending a message ----------
def stream_answer(payload, config, status, meta):
    """Yield answer tokens, and record tool use, sources and token counts in `meta`."""
    for mode, data in be.chatbot.stream(
        {"messages": [payload]}, config=config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, md = data
            if md.get("langgraph_node") != "chat_node":
                continue
            for call in getattr(chunk, "tool_call_chunks", None) or []:
                if call.get("name"):
                    status.caption(TOOL_LABELS.get(call["name"], "Working") + "...")
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                meta["tokens"] += usage.get("total_tokens", 0)
            content = chunk.content
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                status.empty()
                yield content
        elif mode == "updates" and isinstance(data, dict):
            node_out = data.get("chat_node")
            if isinstance(node_out, dict) and node_out.get("messages"):
                lumen = node_out["messages"][-1].additional_kwargs.get("lumen")
                if lumen:
                    meta["lumen"] = lumen


def run_turn(text: str, image_url=None):
    welcome_slot.empty()
    st.session_state["last_error"] = None
    if not be.consume_usage(user):
        st.session_state["last_error"] = "You have reached today's message limit."
        st.rerun()
    if not be.ensure_thread(user, tid):
        st.session_state["last_error"] = "That chat could not be opened."
        st.rerun()

    is_new = be.get_title(user, tid) in (None, "New chat")
    if is_new:
        be.rename_thread(user, tid, be.quick_title(text))
    be.touch_thread(user, tid)

    st.session_state["message_history"].append({"role": "user", "content": text, "image": image_url})
    with st.chat_message("user"):
        if image_url:
            st.image(image_url, width=260)
        st.markdown(text)
    if image_url:
        st.session_state["image_key"] += 1

    config = {
        "configurable": {"thread_id": tid, "user_id": user, "model": model_id},
        "metadata": {"thread_id": tid, "user_id": user},
        "run_name": "chat_turn",
        "recursion_limit": 40,
    }
    meta = {"tokens": 0, "lumen": {}}
    started = time.time()
    answer = ""
    with st.chat_message("assistant"):
        status = st.empty()
        status.caption("Thinking...")
        try:
            answer = st.write_stream(stream_answer(HumanMessage(content=be.build_content(text, image_url)), config, status, meta))
        except Exception as e:
            status.empty()
            st.session_state["last_error"] = friendly_error(e)

    if answer and answer.strip():
        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "lumen": meta["lumen"],
                "stats": {"seconds": time.time() - started, "tokens": meta["tokens"]},
            }
        )
        if is_new:
            be.rename_thread(user, tid, be.generate_title(text))
    elif not st.session_state["last_error"]:
        st.session_state["last_error"] = "The model returned an empty response. Please retry."
    st.rerun()


chat_text = st.chat_input("Type your message here...", disabled=remaining <= 0)
suggested = st.session_state.pop("pending_prompt", None)

if resend:
    run_turn(resend["text"], resend["image"])
else:
    submitted = chat_text or suggested or voice_text
    if submitted:
        image_url = None
        if attached is not None:
            try:
                image_url = be.encode_image(attached.getvalue())
            except Exception:
                st.warning("That image could not be read, so it was not attached.")
        run_turn(submitted, image_url)
