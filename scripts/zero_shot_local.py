import json
import os
import pandas as pd

from vllm import LLM, SamplingParams

PROJECT_DIR = "/content/drive/MyDrive/vanilla_rag_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "normalized")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "golden_test.csv")
)

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
Ты отвечаешь на вопросы о корпоративной базе знаний.
Отвечай кратко и конкретно на русском языке.
Не объясняй ход рассуждений.
Если точного ответа не знаешь, дай наиболее вероятный краткий ответ.
""".strip()

conversations = []

for _, row in test_df.iterrows():
    conversations.append([
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": row["question"]
        }
    ])

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params
)

output_path = os.path.join(
    RESULTS_DIR,
    "local_zero_shot.jsonl"
)

with open(output_path, "w", encoding="utf-8") as f:

    for (_, row), output in zip(test_df.iterrows(), outputs):

        prediction = output.outputs[0].text.strip()

        record = {
            "id": row["id"],
            "question": row["question"],
            "reference": row["answer"],
            "prediction": prediction,
            "model": MODEL
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("Saved:", output_path)
