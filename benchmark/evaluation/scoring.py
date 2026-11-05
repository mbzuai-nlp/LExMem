"""Whole-answer English token F1, following SQuAD v1.1 and LongBench.

Official reference implementations:
https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py
"""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import re
import string


METRIC = "squad_v1_longbench_en_token_f1"


def scoring_config():
    return {
        "metric": METRIC,
        "normalization": "lowercase; delete ASCII punctuation; remove a/an/the; collapse whitespace",
        "tokenization": "whitespace split; multiset overlap",
        "prediction_adapter": "join all answer_items with a single space",
        "reference_adapter": "join one canonical/alias per required gold slot; enumerate all combinations",
        "reference_aggregation": "maximum whole-answer token F1",
        "question_aggregation": "macro mean",
        "zero_overlap_f1": 0.0,
    }


def normalize(text):
    """Match SQuAD v1.1 / LongBench English normalization exactly."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_scores(prediction, ground_truth):
    predicted = normalize(prediction).split()
    gold = normalize(ground_truth).split()
    same = sum((Counter(predicted) & Counter(gold)).values())
    precision = same / len(predicted) if predicted else 0.0
    recall = same / len(gold) if gold else 0.0
    # SQuAD v1.1 / LongBench return zero for zero overlap, even if both are empty.
    f1 = 2 * precision * recall / (precision + recall) if same else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "overlap_tokens": same,
        "prediction_tokens": len(predicted), "reference_tokens": len(gold),
    }


def token_f1_text(prediction, ground_truth):
    return _token_scores(prediction, ground_truth)["f1"]


def full_references(keywords):
    """Adapt structured gold into complete references; never omit a required fact."""
    choices = [list(dict.fromkeys([value["canonical"], *value.get("aliases", [])]))
               for value in keywords.values()]
    for items in product(*choices):
        yield " ".join(items)


def answer_f1(predicted, keywords):
    """Score the entire prediction against each complete reference, then take max."""
    prediction = " ".join(predicted)
    best = None
    count = 0
    for reference in full_references(keywords):
        count += 1
        scores = _token_scores(prediction, reference)
        if best is None or scores["f1"] > best["f1"]:
            best = {**scores, "reference_text": reference}
    return {
        **best, "metric": METRIC, "prediction_text": prediction,
        "reference_count": count,
    }


def main():
    p = argparse.ArgumentParser(description='Macro token F1 of a predictions file '
                                            '(rows {"question_id", "prediction": {"answer_items": [...]}}).')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--qa', type=Path, default=Path(__file__).resolve().parent.parent / 'data' / 'qa.json')
    args = p.parse_args()
    rows = {r['question_id']: r for r in json.loads(args.predictions.read_text())}
    scores = []
    for qa in json.loads(args.qa.read_text()):
        row = rows.get(qa['question_id'], {})
        items = (row.get('prediction') or {}).get('answer_items') if row.get('status') in (None, 'ok') else None
        scores.append(answer_f1(items, qa['answer_keywords'])['f1'] if items else 0.0)
    print(json.dumps({'questions': len(scores), 'macro_f1': sum(scores) / len(scores)}))


if __name__ == "__main__":
    main()
