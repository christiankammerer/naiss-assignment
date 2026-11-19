"""
Step 6a: turn the LLM's answers into results/llm_answers.csv and check the transcripts for rule
violations (reading other files, forward passes are checked by hand).

Usage:  python parse_answers.py ~/llm_runs          then   python score.py
Rows that could not be parsed are left blank: open answer_<cond>.md and fill them in by hand.
"""
import csv
import os
import re
import sys

from common import CORPORA

SUSPICIOUS = ["KEY_DO_NOT_SHARE", "blinded/", "runs/", "train_summary", "manifest.json",
              "data/", "../", "ckpt.pt", "llm_answers"]


def parse(text, cond):
    row = {}
    m = re.search(r"^\s*\**corpus\**\s*:\s*(.+)$", text, re.I | re.M)
    row["corpus"] = m.group(1).strip().strip("*").strip() if m else ""
    if cond == "options":
        for c in CORPORA:
            m = re.search(rf"{c}\s*=\s*([0-9.]+)", text, re.I)
            row[f"p_{c}"] = m.group(1) if m else ""
        row["corpus"] = row["corpus"].split()[0].lower() if row["corpus"] else ""
    else:
        m = re.search(r"^\s*\**confidence\**\s*:\s*([0-9.]+)", text, re.I | re.M)
        row["open_confidence"] = m.group(1) if m else ""
    return row


def main():
    runs = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/llm_runs")
    rows, warnings = [], []
    for name in sorted(os.listdir(runs)):
        d = os.path.join(runs, name)
        if not os.path.isdir(d):
            continue
        for cond in ("options", "open"):
            ans = os.path.join(d, f"answer_{cond}.md")
            if not os.path.exists(ans):
                continue
            text = open(ans, encoding="utf-8").read()
            row = {"model": name, "condition": cond, "llm": os.environ.get("MODEL", "claude-opus-5-5"),
                   "corpus": "", "open_confidence": "", "notes": "",
                   **{f"p_{c}": "" for c in CORPORA}}
            row.update(parse(text, cond))
            tr = os.path.join(d, f"transcript_{cond}.jsonl")
            if os.path.exists(tr):
                t = open(tr, encoding="utf-8").read()
                hits = [s for s in SUSPICIOUS if s in t]
                if hits:
                    row["notes"] = "CHECK transcript: " + ", ".join(hits)
                    warnings.append(f"{name} ({cond}): {', '.join(hits)}")
            if not text.strip():
                row["notes"] = (row["notes"] + " EMPTY ANSWER").strip()
            rows.append(row)

    os.makedirs("results", exist_ok=True)
    cols = ["model", "condition", "llm", "corpus"] + [f"p_{c}" for c in CORPORA] + \
           ["open_confidence", "notes"]
    with open("results/llm_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to results/llm_answers.csv")
    blanks = [f"{r['model']} ({r['condition']})" for r in rows if not r["corpus"]]
    if blanks:
        print("Could not parse (fill in by hand):", ", ".join(blanks))
    if warnings:
        print("Possible rule violations, read these transcripts:\n  " + "\n  ".join(warnings))


if __name__ == "__main__":
    main()
