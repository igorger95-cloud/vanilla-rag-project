import json
import os
import pandas as pd
from groq import Groq

PROJECT_DIR = "/content/drive/MyDrive/vanilla_rag_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "normalized")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL = "openai/gpt-oss-20b"

client = Groq(api_key=os.environ["GROQ_API_KEY"])

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "golden_test.csv")
)

SYSTEM_PROMPT = """
Ты отвечаешь на вопросы о корпоративной базе знаний.
Отвечай кратко и конкретно на русском языке.
Не объясняй ход рассуждений.
Если точного ответа не знаешь, дай наиболее вероятный краткий ответ.
""".strip()

output_path = os.path.join(
    RESULTS_DIR,
    "groq_zero_shot.jsonl"
)

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in test_df.iterrows():

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": row["question"]
                }
            ],
            temperature=0,
            max_tokens=128,
            reasoning_effort="low",
            reasoning_format="hidden"
        )

        prediction = response.choices[0].message.content.strip()

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
