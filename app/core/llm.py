from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. "
    "Answer the user's question concisely and accurately. "
    "If asked about specific novel content, let the user know you are in general mode "
    "and suggest they ask a more specific question."
)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_fallback_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _FALLBACK_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()
