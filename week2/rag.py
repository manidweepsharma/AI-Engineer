import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "week2/chroma_db"
embeddings = OpenAIEmbeddings()

if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print(f"Loaded existing store with {vectorstore._collection.count()} chunks")
else:
    loader = PyPDFLoader("week2/System Design Interview by Alex Xu.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"Stored {vectorstore._collection.count()} chunks in Chroma")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

query = "How does a consistent hashing work?"
results = retriever.invoke(query)

print(f"\nTop {len(results)} chunks for query: {query!r}\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} (page {doc.metadata.get('page')}) ---")
    print(doc.page_content)
    print()

context = "\n\n".join(
    f"[page {doc.metadata.get('page')}]\n{doc.page_content}" for doc in results
)

prompt = f"""Answer the question using only the context below. Cite the page number(s) you used in brackets, e.g. [page 9]. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {query}"""

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke(prompt)

sources = sorted({doc.metadata.get('page') for doc in results})

print("--- Answer ---")
print(response.content)
print(f"\nSources: pages {sources}")