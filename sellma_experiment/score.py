"""
Step 5: score the LLM's answers against the key.

Fill in llm_eval/llm_answers_template.csv (save it as results/llm_answers.csv):
  corpus          : python | swedish | norwegian | danish   (options condition)
                    free text, e.g. "Swedish Wikipedia"      (open condition)
  p_*             : the four probabilities (options condition)
  open_confidence : the stated confidence (open condition)

Usage:  python score.py            -> prints tables, writes results/llm_scores.json
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import binomtest

from common import CORPORA, load_json, save_json

KEYWORDS = {"python": ["python"], "swedish": ["swed", "svensk"],
            "norwegian": ["norw", "bokm", "norsk"], "danish": ["danish", "dansk", "denmark"]}


def parse_open(text):
    t = (text or "").lower()
    hits = [c for c, kws in KEYWORDS.items() if any(k in t for k in kws)]
    return hits[0] if len(hits) == 1 else ("ambiguous" if hits else "other")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--answers", default="results/llm_answers.csv")
    a = ap.parse_args()
    key = load_json(os.path.join(a.dir, "blinded", "KEY_DO_NOT_SHARE.json"))
    with open(os.path.join(a.dir, a.answers), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("corpus")]

    report = {}
    for cond in ("options", "open"):
        rs = [r for r in rows if r["condition"] == cond]
        real = [r for r in rs if key[r["model"]]["corpus"] in CORPORA]
        ctrl = [r for r in rs if key[r["model"]]["corpus"] is None]
        if not real:
            continue
        pred = {r["model"]: (r["corpus"].strip().lower() if cond == "options" else parse_open(r["corpus"]))
                for r in real}
        truth = {r["model"]: key[r["model"]]["corpus"] for r in real}
        correct = [pred[m] == truth[m] for m in pred]
        k, n = sum(correct), len(correct)
        chance = 1 / len(CORPORA)
        res = {"n": n, "correct": k, "accuracy": k / n,
               "chance": chance if cond == "options" else None,
               "p_value_vs_chance": binomtest(k, n, chance, alternative="greater").pvalue
               if cond == "options" else None}
        scandi = [m for m in pred if truth[m] != "python"]
        if scandi:
            ks = sum(pred[m] == truth[m] for m in scandi)
            res["scandinavian"] = {"n": len(scandi), "correct": ks, "accuracy": ks / len(scandi),
                                   "chance": 1 / 3 if cond == "options" else None,
                                   "p_value_vs_chance": binomtest(ks, len(scandi), 1 / 3,
                                                                  alternative="greater").pvalue
                                   if cond == "options" else None}
        # Confusion matrix
        labels = CORPORA + ([] if cond == "options" else ["ambiguous", "other"])
        conf = {t: {p: sum(1 for m in pred if truth[m] == t and pred[m] == p) for p in labels}
                for t in CORPORA}
        res["confusion"] = conf

        # Confidence / calibration
        def conf_of(r):
            if cond == "options":
                ps = [fnum(r[f"p_{c}"]) for c in CORPORA]
                return max(p for p in ps if p is not None) if any(p is not None for p in ps) else None
            return fnum(r["open_confidence"])
        cr = [conf_of(r) for r in real if pred[r["model"]] == truth[r["model"]]]
        wr = [conf_of(r) for r in real if pred[r["model"]] != truth[r["model"]]]
        cc = [conf_of(r) for r in ctrl]
        mean = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
        res["mean_confidence"] = {"when_correct": mean(cr), "when_wrong": mean(wr),
                                  "random_control": mean(cc)}
        if cond == "options":
            briers = []
            for r in real:
                ps = [fnum(r[f"p_{c}"]) or 0.0 for c in CORPORA]
                briers.append(sum((p - (c == truth[r["model"]])) ** 2 for p, c in zip(ps, CORPORA)))
            res["brier"] = float(np.mean(briers))
            res["random_control_answers"] = {r["model"]: r["corpus"] for r in ctrl}

        res["per_model"] = {m: {"truth": truth[m], "predicted": pred[m]} for m in sorted(pred)}
        report[cond] = res

        print(f"\n=== Condition: {cond} ===")
        print(f"Accuracy: {k}/{n} = {k / n:.2f}" +
              (f"  (chance {chance:.2f}, one-sided binomial p = {res['p_value_vs_chance']:.3f})"
               if cond == "options" else ""))
        if scandi:
            s = res["scandinavian"]
            print(f"Scandinavian only: {s['correct']}/{s['n']} = {s['accuracy']:.2f}" +
                  (f"  (chance 0.33, p = {s['p_value_vs_chance']:.3f})" if cond == "options" else ""))
        print("Confusion (rows = truth, cols = predicted):")
        print(f"{'':11s}" + "".join(f"{l[:9]:>10s}" for l in labels))
        for t in CORPORA:
            print(f"{t:11s}" + "".join(f"{conf[t][p]:10d}" for p in labels))
        mc = res["mean_confidence"]
        print(f"Mean confidence: correct {mc['when_correct']}, wrong {mc['when_wrong']}, "
              f"random control {mc['random_control']}")

    save_json(report, os.path.join(a.dir, "results", "llm_scores.json"))
    print("\nWrote results/llm_scores.json")


if __name__ == "__main__":
    main()
