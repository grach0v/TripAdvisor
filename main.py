import asyncio

import streamlit as st
from team import Team


def main() -> None:
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
    st.title("AI Chat Assistant 🤖")

    # adding agent object to session state to persist across sessions
    # streamlit reruns the script on every user interaction
    if "agent" not in st.session_state:
        st.session_state["agent"] = Team()

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # displaying chat history messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        async def process_chat():
            team = await st.session_state["agent"].get_team()

            async for response in team.run_stream(task=prompt):
                st.session_state["messages"].append({"role": "assistant", "content": response})
                with st.chat_message(response.source):
                    st.markdown(response.content)

        asyncio.run(process_chat())


if __name__ == "__main__":
    main()