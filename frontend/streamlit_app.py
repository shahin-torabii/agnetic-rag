import streamlit as st
from pathlib import Path
import requests
from core.constants import IMAGE_EXT, AUDIO_EXT
from config.manager import get_config

config = get_config()



st.set_page_config(page_title="RAG Chatbot", layout="wide")

if not "token" in st.session_state:
    st.session_state.token = None
if not "username" in st.session_state:
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_post(path, json = None, headers = None, files = None):
    try:
        res = requests.post(url=f"{config.backend_api_url}{path}", json=json, headers=headers, files=files, timeout=600)
        if res.status_code >=400:
            st.error(res.json().get("detail", res.text))
            return None

        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend connection failed. Is backend running on http://127.0.0.1:8000 ? See backend.log for details.")
        return None


def api_get(path, headers=None):
    try:
        r = requests.get(f"{config.backend_api_url}{path}", headers=headers, timeout=50)
        if r.status_code >= 400:
            st.error(r.json().get("detail", r.text))
            return None
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend connection failed. Is backend running on http://127.0.0.1:8000 ? See backend.log for details.")
        return None
    except Exception as e:
        st.error(f"{path} error: {e}")
        return None


def api_delete(path, headers=None):
    try:
        r = requests.delete(f"{config.backend_api_url}{path}", headers=headers, timeout=30)
        if r.status_code >= 400:
            st.error(r.json().get("detail", r.text))
            return None
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend connection failed. Is backend running on http://127.0.0.1:8000 ? See backend.log for details.")
        return None

############################UI######################
def login_screen():
    st.title("RAG Chabot")

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("Login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            resp = api_post("/auth/login", json={"username": username, "password": password})
            if resp:
                st.session_state.token = resp["access_token"]
                st.session_state.username = username
                st.rerun()
    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Choose a username", key="reg_user")
            password = st.text_input("Choose a password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Create account")
        if submitted:
            resp = api_post("/auth/register", json={"username": username, "password": password})
            if resp:
                st.session_state.token = resp["access_token"]
                st.session_state.username = username
                st.success("Account created.")
                st.rerun()


def sidebar():
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.username}**")
        if st.button("Log Out"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if st.button("+New chat", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        st.caption("Your Chats")
        chats = api_get("/chats", headers=auth_headers()) or []
        for chat in chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                label = chat["title"] or "(untitled)"
                if st.button(label, key=chat["session_id"], use_container_width=True):
                    st.session_state.current_session_id = chat["session_id"]
                    st.query_params["session_id"] = chat["session_id"]
                    history = api_get(f"/chats/{chat['session_id']}/messages", headers=auth_headers()) or []
                    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in history]
                    st.rerun()

            with col2:
                if st.button("Delete", key=f"del_{chat['session_id']}"):
                    api_delete(f"/chats/{chat["session_id"]}", headers=auth_headers())
                    if st.session_state.current_session_id ==chat["session_id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                        st.query_params.clear()
                    st.rerun()

        st.divider()
        st.caption("Your files (across all chats)")
        user_docs = api_get("/documents", headers=auth_headers()) or []
        for d in user_docs:
            print(d)
            st.text(f"{d['filename']} ({d['kind']})")

        st.divider()

        st.caption("Attach files to your next message")
        uploaded = st.file_uploader("Documents, images, or audio", accept_multiple_files=True, key="Uploader")

        if uploaded:
            for f in uploaded:
                files = {"file": (f.name, f.getvalue())}
                resp = api_post("/upload", headers=auth_headers(), files=files)
                if resp:
                    st.session_state.pending_uploads.append(resp["path"])
            st.success(f"{len(uploaded)} file(s) ready to send with your next message.")


def classify_uploads(paths):
    documents, images, audio = [], [], []
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext in IMAGE_EXT:
            images.append(p)
        elif ext in AUDIO_EXT:
            audio.append(p)
        else:
            documents.append(p)
    return documents, images, audio


def chat_screen():
    sidebar()
    st.title("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    query = st.chat_input("Ask Something...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        documents, images, audio = classify_uploads(st.session_state.pending_uploads)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = api_post(
                    "/chat",
                    json={
                        "session_id": st.session_state.current_session_id,
                        "query": query,
                        "documents": documents,
                        "images": images,
                        "audio": audio,
                    },
                    headers=auth_headers(),)

            if resp:
                st.session_state.current_session_id = resp["session_id"]
                st.write(resp["answer"])
                st.session_state.messages.append({"role": "assistant", "content": resp["answer"]})
                st.session_state.pending_uploads = []


def main():
    if st.session_state.token is None:
        login_screen()
    else:
        chat_screen()


if __name__ == "__main__":
    main()
