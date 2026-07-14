# Commands run this session

A log of the shell and git commands used while building the `week2` RAG pipeline, eval suite, and Hugging Face deployment.

## Environment / packages

```bash
# Check what's installed
pip show langchain-chroma chromadb

# Install the LangChain <-> Chroma integration
source venv/bin/activate && pip install langchain-chroma

# Check Gradio is available
source venv/bin/activate && pip show gradio

# Lock exact versions for requirements.txt
source venv/bin/activate && pip show langchain langchain-community langchain-openai \
  langchain-chroma langchain-core chromadb gradio python-dotenv pypdf openai
```

## Running the RAG scripts

```bash
# Basic pipeline run (ingest or load + retrieve + generate)
source venv/bin/activate && python week2/rag.py

# Chunk-size comparison (200/500/1000)
source venv/bin/activate && python week2/compare_chunk_sizes.py

# Full retrieval + generation eval suite (10 questions, metrics table)
source venv/bin/activate && python week2/eval.py
```

## Running the Gradio UI in the background (for testing without blocking the session)

```bash
# Launch in background, log to file
source venv/bin/activate && python week2/rag.py > /tmp/gradio_rag.log 2>&1 &

# Wait until the server is actually listening, then confirm
until lsof -i :7860 2>/dev/null | grep -q LISTEN; do sleep 2; done; echo "UI is up"

# Check it responds
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860

# Check what's running / listening
ps aux | grep "week2/eval.py" | grep -v grep
lsof -i :7860

# Stop it
kill <pid>
```

## Inspecting the PDF and Chroma store contents

```bash
# Directory sizes (used to decide what should/shouldn't go into git)
du -sh week2/*

# Search the PDF's parsed pages for topic keywords (to find real page numbers
# for the eval set's ground truth, instead of guessing)
python3 -c "
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('week2/System Design Interview by Alex Xu.pdf')
docs = loader.load()
topics = ['CAP theorem', 'unique ID generator', 'web crawler']
for topic in topics:
    hits = [d.metadata['page'] for d in docs if topic.lower() in d.page_content.lower()]
    print(topic, '->', hits[:8])
"

# Dump the raw text of specific pages to quote accurately in the eval set
python3 -c "
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('week2/System Design Interview by Alex Xu.pdf')
docs = loader.load()
for p in [89, 109, 131]:
    print(f'===== PAGE {p} =====')
    print(docs[p].page_content[:900])
"

# Confirm a Chroma collection's chunk count directly (sanity check against duplication)
python -c "
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
e = OpenAIEmbeddings()
vs = Chroma(persist_directory='week2/chroma_db_compare', collection_name='chunks_500', embedding_function=e)
print(vs._collection.count())
"
```

## Git

```bash
# Orientation
git status
git remote -v
cat .gitignore
git diff week2/rag.py
git log --oneline -5

# Stage + commit (first pass: base RAG pipeline)
git add .gitignore week2/rag.py
git commit -m "Add RAG pipeline with retrieval and cited generation over PDF"

# Stage + commit (Gradio UI + chunk comparison script)
git add .gitignore week2/rag.py week2/compare_chunk_sizes.py
git commit -m "Add Gradio chat UI to rag.py and chunk-size comparison script"

# Push
git push origin main
```

## Hugging Face Spaces deployment prep

```bash
# Check for the CLI and login state
pip show huggingface_hub
which hf
hf auth whoami

# Stage the Space directory
mkdir -p week2/hf_space
cp -r week2/chroma_db_eval week2/hf_space/chroma_db_eval
du -sh week2/hf_space/chroma_db_eval
```
