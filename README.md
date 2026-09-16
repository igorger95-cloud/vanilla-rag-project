# Vanilla RAG Pipeline for Corporate Knowledge Base

## Описание проекта

Цель проекта — построить базовый vanilla RAG-пайплайн для ответов на вопросы по корпоративной базе знаний и сравнить его качество с zero-shot LLM без доступа к внутренним документам.

В проекте используются:

- 10 корпоративных статей;
- 50 вопросов;
- 50 эталонных ответов.

Сравниваются три системы:

1. Groq openai/gpt-oss-20b в zero-shot режиме.
2. Локальная Qwen/Qwen2.5-0.5B-Instruct в zero-shot режиме.
3. Qwen/Qwen2.5-0.5B-Instruct + vanilla RAG.

---

## Структура проекта

vanilla_rag_project/
├── data/
│   ├── articles.json
│   ├── questions.json
│   ├── ground_truth.json
│   └── normalized/
│       ├── articles_normalized.json
│       ├── golden_test.csv
│       └── golden_test.json
├── scripts/
│   ├── prepare_data.py
│   ├── zero_shot_groq.py
│   ├── zero_shot_local.py
│   ├── build_rag.py
│   └── evaluate.py
├── results/
├── index/
├── requirements.txt
├── VLLM_INSTALL.md
├── .gitignore
└── README.md

---

## Данные

Используются три исходных файла:

- articles.json — 10 статей корпоративной базы знаний;
- questions.json — 50 вопросов;
- ground_truth.json — 50 эталонных ответов.

После подготовки данных формируется golden test set из 50 пар вопрос–ответ.

---

## Используемые модели

### Groq zero-shot

Модель:

openai/gpt-oss-20b

Модель вызывается через Groq API.

Корпоративные статьи модели не передаются.

### Local zero-shot

Модель:

Qwen/Qwen2.5-0.5B-Instruct

Инференс выполняется локально через vLLM на GPU Tesla T4.

Основные параметры:

- dtype = float16
- max_model_len = 2048
- gpu_memory_utilization = 0.70
- enforce_eager = True
- temperature = 0
- max_tokens = 128

---

## Vanilla RAG

RAG-пайплайн построен с использованием LangChain.

Архитектура:

Corporate articles
↓
LangChain Documents
↓
RecursiveCharacterTextSplitter
↓
Chunks
↓
Embedding model
↓
FAISS
↓
Retriever top-3
↓
Context
↓
Qwen2.5-0.5B-Instruct
↓
Answer

---

## Chunking

Используется RecursiveCharacterTextSplitter.

Параметры:

- chunk_size = 500
- chunk_overlap = 75

10 исходных статей были преобразованы в 20 chunks.

---

## Embeddings

Embedding model:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Embeddings рассчитываются на CPU.

Используется нормализация векторов:

normalize_embeddings = True

---

## Vector Store

Используется FAISS.

Параметры retriever:

- search_type = similarity
- top_k = 3

---

## Retrieval quality

Полученные результаты:

- Recall@1 = 0.88
- Recall@3 = 0.96

Правильная статья находилась:

- на первом месте для 44 из 50 вопросов;
- в top-3 для 48 из 50 вопросов.

Top-3 retrieval miss:

- q4
- q30

---

## Метрики оценки

Для оценки генерации использовались:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- LLM-as-a-Judge

Для BLEU использовался SacreBLEU с tokenizer intl.

Для ROUGE использовалась Unicode-friendly токенизация, так как стандартная токенизация rouge-score некорректно работает с кириллицей.

---

## LLM-as-a-Judge

Judge model:

openai/gpt-oss-120b

Шкала:

- 2 — CORRECT
- 1 — PARTIALLY_CORRECT
- 0 — INCORRECT

Judge сравнивает вопрос, эталонный ответ и ответ модели.

---

## Финальные результаты

| Метрика | Groq 20B zero-shot | Qwen 0.5B zero-shot | Qwen 0.5B + RAG |
|---|---:|---:|---:|
| BLEU | 0.1036 | 0.0523 | 0.1970 |
| ROUGE-1 | 0.4052 | 0.2185 | 0.4573 |
| ROUGE-2 | 0.2121 | 0.0857 | 0.3065 |
| ROUGE-L | 0.3724 | 0.1955 | 0.4263 |
| Fully correct accuracy | 0.0800 | 0.0200 | 0.7600 |
| Partial+correct accuracy | 0.1200 | 0.0200 | 0.7600 |
| Normalized Judge score | 0.1000 | 0.0200 | 0.7600 |

---

## Результаты LLM-as-a-Judge

### Groq zero-shot

- CORRECT: 4
- PARTIALLY_CORRECT: 2
- INCORRECT: 44

Fully correct accuracy: 0.08

Partial+correct accuracy: 0.12

Normalized Judge score: 0.10

### Qwen zero-shot

- CORRECT: 1
- INCORRECT: 49

Fully correct accuracy: 0.02

Partial+correct accuracy: 0.02

Normalized Judge score: 0.02

### Qwen + RAG

- CORRECT: 38
- INCORRECT: 12

Fully correct accuracy: 0.76

Partial+correct accuracy: 0.76

Normalized Judge score: 0.76

---

## Анализ ошибок RAG

Всего неправильных ответов: 12.

Из них:

- Retrieval errors: 2
- Generation errors: 10

Доли:

- Retrieval error share: 16.67%
- Generation error share: 83.33%

Это показывает, что основной источник оставшихся ошибок — генератор, а не retrieval.

Retriever возвращал правильную статью в top-3 в 96% случаев, однако небольшая Qwen2.5-0.5B не всегда могла корректно извлечь и использовать нужный факт из контекста.

---

## Основные выводы

Добавление RAG значительно улучшило качество Qwen2.5-0.5B.

Fully correct accuracy:

2% → 76%

BLEU:

0.0523 → 0.1970

ROUGE-1:

0.2185 → 0.4573

ROUGE-2:

0.0857 → 0.3065

ROUGE-L:

0.1955 → 0.4263

Таким образом, даже небольшая локальная LLM значительно выигрывает от доступа к релевантной корпоративной базе знаний.

---

## Ограничения

Текущий pipeline является базовым vanilla RAG.

В нем отсутствуют:

- query rewriting;
- reranking;
- hybrid search;
- metadata filtering;
- semantic chunking;
- multi-query retrieval;
- answer verification.

---

## Возможные улучшения

### Reranking

Можно получать больше кандидатов через retriever, а затем ранжировать их cross-encoder reranker.

Retriever top-10
↓
Reranker
↓
Top-3

### Query rewriting

Запрос пользователя можно предварительно переписывать в более удобную для поиска форму.

### Hybrid search

Можно объединить dense retrieval и BM25.

### Более сильная локальная LLM

Большинство оставшихся ошибок относятся к generation errors.

Можно протестировать:

- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct

### Улучшение chunking

Можно экспериментировать с:

- chunk_size;
- chunk_overlap;
- semantic chunking.

---

## Запуск

Подготовка данных:

python scripts/prepare_data.py

Groq zero-shot:

python scripts/zero_shot_groq.py

Перед запуском необходимо задать переменную окружения GROQ_API_KEY.

Local zero-shot:

python scripts/zero_shot_local.py

RAG:

python scripts/build_rag.py

Evaluation:

python scripts/evaluate.py

---

## Среда эксперимента

Эксперимент выполнялся в Google Colab.

- GPU: Tesla T4
- GPU memory: около 15 GB
- Python 3.13
- PyTorch 2.13.0+cu129
- CUDA 12.9
- vLLM 0.28.0
- LangChain
- FAISS
- Sentence Transformers
- Groq API

---

## Итог

Vanilla RAG существенно повысил качество небольшой локальной LLM на задаче question answering по закрытой корпоративной базе знаний.

Retrieval уже показывает высокое качество:

Recall@3 = 0.96

Поэтому дальнейшее улучшение системы целесообразно направить прежде всего на качество генератора, а затем на reranking и query rewriting.