import os
import re
import pandas as pd

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

PROJECT_DIR = "/content/drive/MyDrive/vanilla_rag_project"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


class RussianTokenizer:
    def tokenize(self, text):
        text = str(text).lower()

        return re.findall(
            r"\b[\wёЁ]+\b",
            text,
            flags=re.UNICODE
        )


bleu_metric = BLEU(
    effective_order=True,
    tokenize="intl"
)

rouge_metric = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    tokenizer=RussianTokenizer(),
    use_stemmer=False
)


def evaluate_file(path):

    df = pd.read_json(
        path,
        lines=True
    )

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for _, row in df.iterrows():

        reference = str(row["reference"])
        prediction = str(row["prediction"])

        bleu = (
            bleu_metric
            .sentence_score(
                prediction,
                [reference]
            )
            .score
            / 100
        )

        rouge = rouge_metric.score(
            reference,
            prediction
        )

        bleu_scores.append(bleu)
        rouge1_scores.append(
            rouge["rouge1"].fmeasure
        )
        rouge2_scores.append(
            rouge["rouge2"].fmeasure
        )
        rougeL_scores.append(
            rouge["rougeL"].fmeasure
        )

    print("\n", os.path.basename(path))
    print(f"BLEU: {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores):.4f}")
    print(f"ROUGE-2: {sum(rouge2_scores) / len(rouge2_scores):.4f}")
    print(f"ROUGE-L: {sum(rougeL_scores) / len(rougeL_scores):.4f}")


for filename in [
    "groq_zero_shot.jsonl",
    "local_zero_shot.jsonl",
    "local_rag.jsonl"
]:
    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if os.path.exists(path):
        evaluate_file(path)
