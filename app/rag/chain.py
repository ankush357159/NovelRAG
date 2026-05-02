from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.core.llm import get_llm
from app.vectorstore.retriever import get_retriever

_NOVEL_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a literary assistant specializing in the novels provided in the context.\n"
            "Answer the question using ONLY the provided context passages.\n"
            "If the answer is not found in the context, say: "
            "'I could not find an answer in the available text.'\n"
            "Always cite the novel title and chapter label when answering.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{query}"),
    ]
)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', 'Unknown')}, {doc.metadata.get('chapter_label', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


def get_rag_chain(novel_filter: str | None = None):
    """
    Returns an LCEL chain that accepts {"query": str} and produces a dict:
      {
        "query": str,
        "context_docs": list[Document],
        "context": str,        # formatted passages passed to the LLM
        "answer": str,
      }
    """
    retriever = get_retriever(novel_filter=novel_filter)

    chain = (
        RunnablePassthrough.assign(
            context_docs=lambda x: retriever.invoke(x["query"])
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_docs(x["context_docs"])
        )
        | RunnablePassthrough.assign(
            answer=_NOVEL_QA_PROMPT | get_llm() | StrOutputParser()
        )
    )

    return chain
