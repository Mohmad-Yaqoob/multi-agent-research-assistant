import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv()


SUMMARISE_AFTER = 10

def should_summarise(messages: list) -> bool:
    turns = sum(1 for m in messages if isinstance(m, (HumanMessage, AIMessage)))
    return turns >= SUMMARISE_AFTER

def summarise_history(messages: list) -> tuple:
    keep = 8
    old = messages[:-keep] if len(messages) > keep else []
    recent = messages[-keep:] if len(messages) > keep else messages

    if not old:
        return "", messages

    text = "\n".join(
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in old
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=300,
    )

    result = llm.invoke([
        SystemMessage(content="Summarise this conversation in under 150 words. Keep key facts, names, numbers."),
        HumanMessage(content=text),
    ])

    return result.content.strip(), recent


def build_messages(messages: list, system_prompt: str) -> list:
    base = SystemMessage(content=system_prompt)

    if not should_summarise(messages):
        return [base] + messages

    summary, recent = summarise_history(messages)

    system_with_memory = SystemMessage(content=(
        f"{system_prompt}\n\n"
        f"--- Earlier conversation summary ---\n{summary}\n"
        f"--- End of summary ---"
    ))

    return [system_with_memory] + recent