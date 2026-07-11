from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = "week2/System Design Interview by Alex Xu.pdf"
PERSIST_DIR = "week2/chroma_db_compare"
CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAP = 50
QUERY = "How does spring boot work?"
K = 4

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


def build_and_query(chunk_size):
    collection_name = f"chunks_{chunk_size}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    existing = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    if existing._collection.count() == len(chunks):
        vectorstore = existing
    else:
        existing.delete_collection()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=collection_name
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    results = retriever.invoke(QUERY)

    context = "\n\n".join(
        f"[page {doc.metadata.get('page')}]\n{doc.page_content}" for doc in results
    )
    prompt = f"""Answer the question using only the context below. Cite the page number(s) you used in brackets, e.g. [page 9]. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {QUERY}"""

    answer = llm.invoke(prompt).content
    sources = sorted({doc.metadata.get('page') for doc in results})

    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "context_chars": len(context),
        "sources": sources,
        "answer": answer,
    }


runs = [build_and_query(size) for size in CHUNK_SIZES]

print(f"Query: {QUERY!r}\n")
for run in runs:
    print(f"{'=' * 60}")
    print(f"chunk_size={run['chunk_size']}  chunks_created={run['num_chunks']}  "
          f"context_chars={run['context_chars']}  sources={run['sources']}")
    print(f"{'-' * 60}")
    print(run["answer"])
    print()


demo = gr.ChatInterface(
    fn=chat,
    title="AI/ML Interview Prep Coach",
    description="Practice ML interview questions. Pick a topic and start.",
)
demo.launch()