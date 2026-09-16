import json
import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from vllm import LLM, SamplingParams

PROJECT_DIR = "/content/drive/MyDrive/vanilla_rag_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
INDEX_DIR = os.path.join(
    PROJECT_DIR,
    "index",
    "faiss_multilingual_minilm"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

with open(
    os.path.join(DATA_DIR, "articles.json"),
    encoding="utf-8"
) as f:
    articles = json.load(f)

with open(
    os.path.join(DATA_DIR, "questions.json"),
    encoding="utf-8"
) as f:
    questions = json.load(f)

golden_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "normalized",
        "golden_test.csv"
    )
)

reference_map = dict(
    zip(golden_df["id"], golden_df["answer"])
)

documents = [
    Document(
        page_content=article["text"],
        metadata={
            "article_id": article["id"],
            "title": article["title"]
        }
    )
    for article in articles
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(INDEX_DIR)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=MODEL,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    disable_log_stats=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128
)

SYSTEM_PROMPT = """
Ты отвечаешь на вопросы по корпоративной базе знаний.
Используй только информацию из предоставленного контекста.

Правила:
- отвечай кратко и конкретно;
- не добавляй факты, которых нет в контексте;
- если ответа в контексте нет, скажи:
  "В предоставленном контексте нет ответа.";
- отвечай на русском языке.
""".strip()


def build_context(docs):
    parts = []

    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"[Документ {i}]\n"
            f"Название: {doc.metadata.get('title')}\n"
            f"Текст: {doc.page_content}"
        )

    return "\n\n".join(parts)


conversations = []
metadata = []

for item in questions:

    docs = retriever.invoke(item["question"])
    context = build_context(docs)

    conversations.append([
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"КОНТЕКСТ:\n{context}\n\n"
                f"ВОПРОС:\n{item['question']}"
            )
        }
    ])

    metadata.append({
        "id": item["id"],
        "question": item["question"],
        "reference": reference_map[item["id"]],
        "retrieved_article_ids": [
            str(doc.metadata["article_id"])
            for doc in docs
        ]
    })

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params
)

output_path = os.path.join(
    RESULTS_DIR,
    "local_rag.jsonl"
)

with open(output_path, "w", encoding="utf-8") as f:

    for meta, output in zip(metadata, outputs):

        prediction = output.outputs[0].text.strip()

        record = {
            **meta,
            "prediction": prediction,
            "model": MODEL
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("Saved:", output_path)
