import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "week2/chroma_db_eval"
embeddings = OpenAIEmbeddings()

if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print(f"Loaded existing store with {vectorstore._collection.count()} chunks")
else:
    loader = PyPDFLoader("week2/System Design Interview by Alex Xu.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = ChatOpenAI(model="gpt-4o-mini")


def run_rag(question):
    results = retriever.invoke(question)
    retrieved_pages = [doc.metadata["page"] for doc in results]

    context = "\n\n".join(
        f"[page {doc.metadata.get('page')}]\n{doc.page_content}" for doc in results
    )
    prompt = f"""Answer the question using only the context below. Cite the page number(s) you used in brackets, e.g. [page 9]. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}"""

    answer = llm.invoke(prompt).content
    return retrieved_pages, answer


def hit_rate(retrieved_pages, expected_pages):
    for page in expected_pages:
        if page in retrieved_pages:
            return 1
    return 0


def recall_at_k(retrieved_pages, expected_pages):
    found = 0
    for page in expected_pages:
        if page in retrieved_pages:
            found += 1
    return found / len(expected_pages)


def mrr(retrieved_pages, expected_pages):
    for i, page in enumerate(retrieved_pages):
        if page in expected_pages:
            return 1 / (i + 1)
    return 0


def judge_answer(question, expected, actual):
    response = llm.invoke([
        SystemMessage(content="Rate the answer quality 1-5. 1=wrong, 3=partially correct, 5=fully correct. Return only the number."),
        HumanMessage(content=f"Question: {question}\nExpected: {expected}\nActual: {actual}")
    ])
    match = re.search(r"[1-5]", response.content)
    return int(match.group()) if match else 0


eval_set = [
    {"question": "How does a load balancer work?", "expected_answer": '''A load balancer evenly distributes incoming traffic among web servers. users connect to the public IP of the load balancer directly. With this setup, web servers are unreachable directly by clients anymore. For better security, private
IPs are used for communication between servers. If the website traffic grows rapidly, and two servers are not enough to handle the traffic,
the load balancer can handle this problem gracefully. You only need to add more servers
to the web server pool, and the load balancer automatically starts to send requests to them.''', "expected_pages": [9]},

    {"question": "What is consistent hashing and why is it needed?", "expected_answer": '''Consistent hashing is a special kind of hashing such that when a
hash table is re-sized and consistent hashing is used, only k/n keys need to be remapped on
average, where k is the number of keys, and n is the number of slots. In contrast, in most
traditional hash tables, a change in the number of array slots causes nearly all keys to be
remapped''', "expected_pages": [74]},

    {"question": "How would you design a URL shortener?", "expected_answer": '''Hash + Collision Resolution
We pass the longURL through standard cryptographic hashing algorithms (like MD5 or SHA-1) and extract the first 7 characters.
* The Issue: Collisions are highly likely with truncated hashes.
* Resolution: If a collision is detected in the database, we append a predefined string to the original URL and re-hash recursively until a unique hash is found. To optimize lookup performance and avoid hitting the database for every single collision check, we can use a Bloom filter (a space-efficient probabilistic data structure)
''', "expected_pages": [11, 13]},

    {"question": "How does rate limiting work?", "expected_answer": '''a rate limiter is used to control the rate of traffic sent by a client or a
service. In the HTTP world, a rate limiter limits the number of client requests allowed to be
sent over a specified period. If the API request count exceeds the threshold defined by the
rate limiter, all the excess calls are blocked.''', "expected_pages": [50]},

    {"question": "What is the CAP theorem and why does it matter for distributed systems?", "expected_answer": '''CAP theorem states it is impossible for a distributed system to simultaneously provide more
than two of these three guarantees: consistency, availability, and partition tolerance.
Consistency: consistency means all clients see the same data at the same time no matter
which node they connect to.
Availability: availability means any client which requests data gets a response even if some
of the nodes are down.
Partition Tolerance: a partition indicates a communication break between two nodes.
Partition tolerance means the system continues to operate despite network partitions.''', "expected_pages": [89]},

    {"question": "Why doesn't auto_increment work for a unique ID generator in a distributed system?", "expected_answer": '''Your first thought might be to use a primary key with the auto_increment attribute in a traditional
database. However, auto_increment does not work in a distributed environment because a
single database server is not large enough and generating unique IDs across multiple
databases with minimal delay is challenging.''', "expected_pages": [109]},

    {"question": "What is a web crawler used for?", "expected_answer": '''A web crawler is known as a robot or spider. It is widely used by search engines to discover
new or updated content on the web. Content can be a web page, an image, a video, a PDF
file, etc. A web crawler starts by collecting a few web pages and then follows links on those
pages to collect new content. A crawler is used for search engine indexing and web archiving,
among other purposes.''', "expected_pages": [131]},

    {"question": "What are the three types of notification formats in a notification system?", "expected_answer": '''A notification is more than just mobile push notification. Three types of notification formats
are: mobile push notification, SMS message, and Email.''', "expected_pages": [150]},

    {"question": "What is the first thing to clarify when designing a chat system?", "expected_answer": '''It is extremely important to nail
down the exact requirements. For example, you do not want to design a system that focuses
on group chat when the interviewer has one-on-one chat in mind. It is important to explore
the feature requirements.''', "expected_pages": [177]},

    {"question": "How does a CDN work?", "expected_answer": '''A CDN is a network of geographically dispersed servers used to deliver static content. CDN
servers cache static content like images, videos, CSS, JavaScript files, etc. Here is how CDN
works at the high-level: when a user visits a website, a CDN server closest
to the user will deliver static content. The further users are from CDN servers, the
slower the website loads.''', "expected_pages": [16]},
]

rows = []
for item in eval_set:
    retrieved_pages, answer = run_rag(item["question"])
    hr = hit_rate(retrieved_pages, item["expected_pages"])
    rec = recall_at_k(retrieved_pages, item["expected_pages"])
    rank = mrr(retrieved_pages, item["expected_pages"])
    quality = judge_answer(item["question"], item["expected_answer"], answer)
    rows.append((item["question"], hr, rec, rank, quality))

print(f"\n{'Question':<50} | {'Hit Rate':<8} | {'Recall@k':<8} | {'MRR':<6} | {'Quality':<7}")
print("-" * 90)
for question, hr, rec, rank, quality in rows:
    print(f"{question[:48]:<50} | {hr:<8} | {rec:<8.2f} | {rank:<6.2f} | {quality}/5")

n = len(rows)
avg_hr = sum(r[1] for r in rows) / n
avg_rec = sum(r[2] for r in rows) / n
avg_mrr = sum(r[3] for r in rows) / n
avg_quality = sum(r[4] for r in rows) / n

print("-" * 90)
print(f"{'AVERAGE':<50} | {avg_hr:<8.2f} | {avg_rec:<8.2f} | {avg_mrr:<6.2f} | {avg_quality:.2f}/5")


def answer_question(question):
    retrieved_pages, answer = run_rag(question)
    return f"{answer}\n\nSources: pages {retrieved_pages}"


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a system design question"),
    outputs=gr.Textbox(label="Answer"),
    title="System Design RAG Q&A"
)

if __name__ == "__main__":
    demo.launch()
