import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

BASE_PATH = os.getenv("DELTAKV_OUTPUT_DIR", "/root/autodl-fs/deltakv_outputs")

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

TASK_HIERARCHY = {
    "SDQA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "MDQA": ["hotpotqa", "2wikimqa", "musique"],
    "SUM": ["gov_report", "qmsum", "multi_news"],
    "FewShot": ["trec", "triviaqa", "samsum"],
    "Syn": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--compressor_path", type=str, default=None)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--path", type=str, default=None, help="The path to the prediction results")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def _round_float(value):
    return round(float(value), 2)


def aggregate_category_scores(task_scores):
    category_scores = {}
    for category, tasks in TASK_HIERARCHY.items():
        present_tasks = [task for task in tasks if task in task_scores]
        if not present_tasks:
            continue

        first_score = task_scores[present_tasks[0]]
        if isinstance(first_score, dict):
            bucket_scores = {}
            for bucket in first_score:
                values = [
                    task_scores[task][bucket]
                    for task in present_tasks
                    if isinstance(task_scores[task], dict) and bucket in task_scores[task]
                ]
                if values:
                    bucket_scores[bucket] = _round_float(np.mean(values))
            category_scores[category] = bucket_scores
        else:
            values = [
                task_scores[task]
                for task in present_tasks
                if not isinstance(task_scores[task], dict)
            ]
            if values:
                category_scores[category] = _round_float(np.mean(values))

    if not category_scores:
        return category_scores, None

    first_category_score = next(iter(category_scores.values()))
    if isinstance(first_category_score, dict):
        overall_score = {}
        for bucket in first_category_score:
            values = [
                score[bucket]
                for score in category_scores.values()
                if isinstance(score, dict) and bucket in score
            ]
            if values:
                overall_score[bucket] = _round_float(np.mean(values))
    else:
        values = [score for score in category_scores.values() if not isinstance(score, dict)]
        overall_score = _round_float(np.mean(values)) if values else None

    return category_scores, overall_score


if __name__ == '__main__':
    args = parse_args()
    
    if args.path:
        path = args.path
    else:
        compressor_path = args.compressor_path if args.compressor_path is not None else args.cfg
        if compressor_path is not None:
            compressor_name = os.path.basename(compressor_path.rstrip('/'))
        else:
            compressor_name = "None"
            
        if args.e:
            path = os.path.join(BASE_PATH, f"benchmark/long_bench/pred_e/{args.model}/{compressor_name}")
        else:
            path = os.path.join(BASE_PATH, f"benchmark/long_bench/pred/{args.model}/{compressor_name}")
    
    task_scores = dict()
    if not os.path.exists(path):
        
        print(f"Path {path} does not exist.")
        exit(1)
        
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        try:
            if args.e:
                score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            else:
                score = scorer(dataset, predictions, answers, all_classes)
            task_scores[dataset] = score
        except:
            print(f"error in {dataset}")
            pass

    category_scores, overall_category_avg = aggregate_category_scores(task_scores)
    scores = {
        **task_scores,
        "category_scores": category_scores,
        "overall_category_avg": overall_category_avg,
    }
    
    out_path = os.path.join(path, "result.json")
    print(scores)
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
