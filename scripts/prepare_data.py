import json
import os
import pandas as pd

PROJECT_DIR = "/content/drive/MyDrive/vanilla_rag_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
NORMALIZED_DIR = os.path.join(DATA_DIR, "normalized")

os.makedirs(NORMALIZED_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "articles.json"), encoding="utf-8") as f:
    articles = json.load(f)

with open(os.path.join(DATA_DIR, "questions.json"), encoding="utf-8") as f:
    questions = json.load(f)

with open(os.path.join(DATA_DIR, "ground_truth.json"), encoding="utf-8") as f:
    ground_truth = json.load(f)

articles_df = pd.DataFrame(articles)
questions_df = pd.DataFrame(questions)
ground_truth_df = pd.DataFrame(ground_truth)

golden_df = questions_df.merge(
    ground_truth_df,
    on="id",
    how="inner"
)[["id", "question", "answer"]]

articles_df.to_json(
    os.path.join(NORMALIZED_DIR, "articles_normalized.json"),
    orient="records",
    force_ascii=False,
    indent=2
)

golden_df.to_csv(
    os.path.join(NORMALIZED_DIR, "golden_test.csv"),
    index=False,
    encoding="utf-8-sig"
)

golden_df.to_json(
    os.path.join(NORMALIZED_DIR, "golden_test.json"),
    orient="records",
    force_ascii=False,
    indent=2
)

print("Articles:", len(articles_df))
print("Questions:", len(questions_df))
print("Golden test:", len(golden_df))
