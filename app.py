"""Streamlit chat UI with persistent recent conversations."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
CHATS_FILE = BASE_DIR / "chats.json"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Tamil Nadu Agriculture Schemes", page_icon="🌾", layout="wide")

def load_chats():
    try:
        return json.loads(CHATS_FILE.read_text(encoding="utf-8")) if CHATS_FILE.exists() else {}
    except Exception:
        return {}

def save_chats():
    CHATS_FILE.write_text(json.dumps(st.session_state.chats, ensure_ascii=False, indent=2), encoding="utf-8")

def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {"title": "New chat", "messages": [], "updated": datetime.now().isoformat(timespec="seconds")}
    st.session_state.active_chat = cid
    save_chats()

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "active_chat" not in st.session_state or st.session_state.active_chat not in st.session_state.chats:
    if st.session_state.chats:
        st.session_state.active_chat = sorted(st.session_state.chats, key=lambda c: st.session_state.chats[c].get("updated", ""), reverse=True)[0]
    else:
        new_chat()

with st.sidebar:
    st.title("🌾 Agriculture Assistant")
    if st.button("＋ New chat", use_container_width=True):
        new_chat()
        st.rerun()
    st.caption("Recent chats")
    recent = sorted(st.session_state.chats.items(), key=lambda x: x[1].get("updated", ""), reverse=True)[:15]
    for cid, chat in recent:
        c1, c2 = st.columns([5, 1])
        label = chat.get("title", "New chat")[:34]
        if c1.button(label, key=f"open_{cid}", use_container_width=True):
            st.session_state.active_chat = cid
            st.rerun()
        if c2.button("✕", key=f"del_{cid}"):
            del st.session_state.chats[cid]
            if not st.session_state.chats:
                new_chat()
            elif st.session_state.active_chat == cid:
                st.session_state.active_chat = next(iter(st.session_state.chats))
            save_chats()
            st.rerun()
    st.divider()
    st.caption("Answers are based only on the PDFs placed in the Data folder.")

st.title("Tamil Nadu Agriculture Schemes Assistant")
st.write("Ask about schemes, eligibility, benefits, subsidies, and application details found in your documents.")
chat = st.session_state.chats[st.session_state.active_chat]

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(f"**{source['pdf']}**, page {source['page']}")

prompt = st.chat_input("Ask a question in English or Tamil…")
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"][-6:]]
    chat["messages"].append({"role": "user", "content": prompt})
    if chat.get("title") == "New chat":
        chat["title"] = prompt[:42]
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        sources = []
        try:
            with requests.post(f"{API_URL}/chat/stream", json={"question": prompt, "history": history}, stream=True, timeout=(10, 180)) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    if event["event"] == "token":
                        full += event.get("text", "")
                        placeholder.markdown(full + "▌")
                    elif event["event"] == "replace":
                        full = event.get("text", "")
                        placeholder.markdown(full)
                    elif event["event"] == "sources":
                        sources = event.get("sources", [])
                    elif event["event"] == "error":
                        full = event.get("text", "Request failed.")
                        placeholder.markdown(full)
        except Exception as exc:
            full = f"Could not connect to the API at `{API_URL}`. Start the backend first. Details: {exc}"
            placeholder.error(full)
        placeholder.markdown(full)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"**{source['pdf']}**, page {source['page']}")
    chat["messages"].append({"role": "assistant", "content": full, "sources": sources})
    chat["updated"] = datetime.now().isoformat(timespec="seconds")
    save_chats()