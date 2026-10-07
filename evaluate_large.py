"""
Quantitative evaluation of the screening engine on the generated dataset.

Runs the real screening_engine.py -- nothing is reimplemented here -- against
the 10,000-resume dataset and its ground-truth labels, and reports four things
the small demo batch could not measure:

  1. Skill detection quality. The checklist matcher is scored against the
     skills the generator actually wrote into each resume: precision, recall
     and F1, micro-averaged over every (resume, skill) decision.

  2. Ranking quality, baseline vs hybrid. Each of the 8 job descriptions is
     run against all 10,000 resumes and scored with Precision@k, Recall@k,
     MAP and NDCG@k against relevance judgements derived from ground-truth
     skills -- judgements that never look at the resume text, so they cannot
     favour either scoring method.

  3. The same comparison split by how each resume phrases its skills
     (spelled-out / abbreviated / mixed). This isolates the synonym step on
     exactly the resumes it is designed to help.

  4. Scalability: wall-clock screening time as the pool grows.

Usage:  python evaluate_large.py [--n 10000] [--out results]
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from screening_engine import (extract_matched_skills, extract_skills_from_jd,
                              screen_resumes)

DATA = "dataset"


# ---------------------------------------------------------------- metrics
def precision_at_k(ranked_rel, k):
    top = ranked_rel[:k]
    return sum(top) / k if k else 0.0


def recall_at_k(ranked_rel, k, total_rel):
    if not total_rel:
        return 0.0
    return sum(ranked_rel[:k]) / total_rel


def average_precision(ranked_rel, total_rel):
    if not total_rel:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(ranked_rel, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / total_rel


def ndcg_at_k(ranked_rel, k, total_rel):
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(ranked_rel[:k], start=1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, total_rel) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------- loading
def load(n):
    resumes = pd.read_csv(os.path.join(DATA, "resumes_10k.csv"))
    truth = pd.read_csv(os.path.join(DATA, "ground_truth.csv"))
    rel = pd.read_csv(os.path.join(DATA, "relevance.csv"))
    if n and n < len(resumes):
        resumes = resumes.head(n)
        keep = set(resumes["candidate_name"])
        truth = truth[truth["candidate_name"].isin(keep)]
        rel = rel[rel["candidate_name"].isin(keep)]
    return resumes, truth, rel


# ------------------------------------------------- 1. skill detection
def eval_skill_detection(resumes, truth):
    """Micro-averaged precision/recall/F1 over every (resume, skill) decision.

    The universe of skills is the union of everything the generator can emit,
    so a false positive means the matcher claimed a skill the resume does not
    actually have.
    """
    truth_map = {r.candidate_name: set(str(r.true_skills).split("|"))
                 for r in truth.itertuples()}
    universe = sorted({s for v in truth_map.values() for s in v})

    tp = fp = fn = 0
    per_skill = defaultdict(lambda: [0, 0, 0])      # skill -> [tp, fp, fn]

    for row in resumes.itertuples():
        actual = truth_map.get(row.candidate_name, set())
        matched, _ = extract_matched_skills(row.resume_text, universe)
        predicted = set(matched)
        for s in universe:
            p, a = s in predicted, s in actual
            if p and a:
                tp += 1; per_skill[s][0] += 1
            elif p and not a:
                fp += 1; per_skill[s][1] += 1
            elif a and not p:
                fn += 1; per_skill[s][2] += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    worst = sorted(
        ((s, v) for s, v in per_skill.items() if v[0] + v[2] >= 30),
        key=lambda kv: kv[1][0] / (kv[1][0] + kv[1][2]))[:5]

    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "decisions": len(resumes) * len(universe),
            "worst_recall": [
                {"skill": s, "recall": v[0] / (v[0] + v[2]), "support": v[0] + v[2]}
                for s, v in worst]}


# ------------------------------------------------- 2/3. ranking quality
def eval_ranking(resumes, truth, rel, ks=(10, 25, 50, 100)):
    phrasing = {r.candidate_name: r.phrasing for r in truth.itertuples()}
    jd_dir = os.path.join(DATA, "job_descriptions")
    job_ids = sorted(f[:-4] for f in os.listdir(jd_dir) if f.endswith(".txt"))

    agg = {m: defaultdict(list) for m in ("baseline", "hybrid")}
    by_phrasing = {m: defaultdict(lambda: defaultdict(list))
                   for m in ("baseline", "hybrid")}
    per_job = []

    for job_id in job_ids:
        with open(os.path.join(jd_dir, f"{job_id}.txt")) as f:
            jd = f.read()

        required = extract_skills_from_jd(jd)
        t0 = time.perf_counter()
        results = screen_resumes(jd, resumes, required)
        elapsed = time.perf_counter() - t0

        judged = rel[rel["job_id"] == job_id]
        rel_map = dict(zip(judged["candidate_name"], judged["relevant"]))
        total_rel = int(judged["relevant"].sum())

        row = {"job_id": job_id, "relevant": total_rel, "seconds": elapsed}

        for mode, col in (("baseline", "Baseline_TFIDF_%"),
                          ("hybrid", "Hybrid_Match_%")):
            order = results.sort_values(col, ascending=False)
            names = list(order["Candidate"])
            ranked_rel = [rel_map.get(n, 0) for n in names]

            for k in ks:
                agg[mode][f"P@{k}"].append(precision_at_k(ranked_rel, k))
                agg[mode][f"R@{k}"].append(recall_at_k(ranked_rel, k, total_rel))
                agg[mode][f"NDCG@{k}"].append(ndcg_at_k(ranked_rel, k, total_rel))
            agg[mode]["MAP"].append(average_precision(ranked_rel, total_rel))
            row[f"{mode}_P@10"] = precision_at_k(ranked_rel, 10)
            row[f"{mode}_NDCG@10"] = ndcg_at_k(ranked_rel, 10, total_rel)

            # --- split by how the resume phrases its skills ---------------
            for style in ("full", "abbr", "mixed"):
                sub = [(n, rel_map.get(n, 0)) for n in names
                       if phrasing.get(n) == style]
                sub_rel = [r for _, r in sub]
                sub_total = sum(rel_map.get(n, 0) for n in names
                                if phrasing.get(n) == style)
                if sub_total:
                    by_phrasing[mode][style]["P@10"].append(
                        precision_at_k(sub_rel, 10))
                    by_phrasing[mode][style]["NDCG@10"].append(
                        ndcg_at_k(sub_rel, 10, sub_total))

        per_job.append(row)

    summary = {m: {k: float(np.mean(v)) for k, v in agg[m].items()}
               for m in agg}
    phr = {m: {s: {k: float(np.mean(v)) for k, v in d.items()}
               for s, d in by_phrasing[m].items()} for m in by_phrasing}
    return summary, phr, per_job


# ------------------------------------------------- 4. scalability
def eval_scalability(resumes, sizes=(100, 500, 1000, 2500, 5000, 10000)):
    jd_path = os.path.join(DATA, "job_descriptions", "ml_engineer.txt")
    with open(jd_path) as f:
        jd = f.read()
    required = extract_skills_from_jd(jd)
    out = []
    for n in sizes:
        if n > len(resumes):
            continue
        sub = resumes.head(n)
        t0 = time.perf_counter()
        screen_resumes(jd, sub, required)
        dt = time.perf_counter() - t0
        out.append({"n": n, "seconds": round(dt, 3),
                    "ms_per_resume": round(dt / n * 1000, 3)})
    return out


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    resumes, truth, rel = load(args.n)
    print(f"Loaded {len(resumes)} resumes, {len(rel)} relevance judgements\n")

    print("[1/4] Skill detection against ground truth ...")
    skill = eval_skill_detection(resumes, truth)

    print("[2/4] Ranking quality, baseline vs hybrid, 8 job descriptions ...")
    summary, phr, per_job = eval_ranking(resumes, truth, rel)

    print("[3/4] Split by resume phrasing style ...")

    print("[4/4] Scalability ...")
    scale = eval_scalability(resumes)

    out = {"n_resumes": len(resumes), "skill_detection": skill,
           "ranking": summary, "by_phrasing": phr, "per_job": per_job,
           "scalability": scale}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(args.out, "per_job.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_job[0]))
        w.writeheader(); w.writerows(per_job)

    # ------------------------------------------------ printed report
    L = []
    L.append("$ python evaluate_large.py")
    L.append(f"Dataset : dataset/resumes_10k.csv  ({len(resumes)} resumes, generated)")
    L.append(f"Labels  : dataset/ground_truth.csv + relevance.csv")
    L.append("")
    L.append("--- 1. Skill detection vs ground truth ---")
    L.append(f"Decisions scored : {skill['decisions']:,} (resume x skill)")
    L.append(f"Precision {skill['precision']:.3f}   Recall {skill['recall']:.3f}"
             f"   F1 {skill['f1']:.3f}")
    L.append(f"TP {skill['tp']:,}  FP {skill['fp']:,}  FN {skill['fn']:,}")
    if skill["worst_recall"]:
        L.append("Lowest-recall skills:")
        for w in skill["worst_recall"]:
            L.append(f"  {w['skill']:<28}{w['recall']:.3f}  (n={w['support']})")
    L.append("")
    L.append("--- 2. Ranking quality, averaged over 8 job descriptions ---")
    L.append(f"{'Metric':<12}{'Baseline':>10}{'Hybrid':>10}{'Change':>10}")
    for m in ("P@10", "P@25", "P@50", "P@100", "R@10", "R@100",
              "NDCG@10", "NDCG@100", "MAP"):
        b, h = summary["baseline"][m], summary["hybrid"][m]
        d = h - b
        L.append(f"{m:<12}{b:>10.3f}{h:>10.3f}{d:>+10.3f}")
    L.append("")
    L.append("--- 3. By how the resume phrases its skills (NDCG@10) ---")
    L.append(f"{'Phrasing':<12}{'Baseline':>10}{'Hybrid':>10}{'Change':>10}")
    for style, label in (("full", "spelled out"), ("abbr", "abbreviated"),
                         ("mixed", "mixed")):
        b = phr["baseline"].get(style, {}).get("NDCG@10")
        h = phr["hybrid"].get(style, {}).get("NDCG@10")
        if b is not None:
            L.append(f"{label:<12}{b:>10.3f}{h:>10.3f}{h - b:>+10.3f}")
    L.append("")
    L.append("--- 4. Scalability (single CPU, no GPU) ---")
    L.append(f"{'Resumes':>9}{'Seconds':>10}{'ms/resume':>12}")
    for s in scale:
        L.append(f"{s['n']:>9,}{s['seconds']:>10.2f}{s['ms_per_resume']:>12.3f}")

    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
