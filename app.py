from typing import Any

import streamlit as st

from src.runner import new_thread_id, resume, start

st.set_page_config(page_title="Crocoblock AI Support Agent", page_icon="🐊")
st.title("🐊 Crocoblock AI Support Agent")
st.caption("JetFormBuilder · Docs Q&A · Bug Investigator · Code Generator")

ANSWER_FIELDS = ("final_answer", "answer", "response")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
    st.session_state.history = []      # [(role, text)]
    st.session_state.awaiting = False  # True, если граф стоит на interrupt()


def answer_text(state: dict[str, Any]) -> str:
    for field in ANSWER_FIELDS:
        value = state.get(field)
        if value:
            return str(value)
    return "_(no answer found in state)_"


def render_badge(state: dict[str, Any]) -> None:
    query_type, handled_by = state.get("query_type"), state.get("handled_by")
    if query_type or handled_by:
        st.caption(f"`{query_type}` → **{handled_by}**")


with st.sidebar:
    st.subheader("Сессия")
    st.code(st.session_state.thread_id, language=None)
    if st.button("Новый диалог", use_container_width=True):
        st.session_state.thread_id = new_thread_id()
        st.session_state.history = []
        st.session_state.awaiting = False
        st.rerun()

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

placeholder = (
    "Ваш ответ агенту…"
    if st.session_state.awaiting
    else "Опишите проблему или задайте вопрос"
)

if prompt := st.chat_input(placeholder):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Агент работает…"):
            if st.session_state.awaiting:
                state, question = resume(prompt, st.session_state.thread_id, "streamlit")
            else:
                state, question = start(prompt, st.session_state.thread_id, "streamlit")

        reply = question if question else answer_text(state)
        st.session_state.awaiting = bool(question)
        render_badge(state)
        st.markdown(reply)

    st.session_state.history.append(("assistant", reply))
    st.session_state.history.append(("assistant", reply))
    st.rerun()
