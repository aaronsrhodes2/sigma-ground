"""One-off: pull the QUESTION prompts out of the What-If PDF.

We deliberately keep ONLY the short reader questions (the scenario prompts) —
not Munroe's prose or comics — to use as simulator test cases. Text layer only.
"""
import re
import sys

import pdfplumber

PDF = r"C:\Users\aaron\Downloads\what_if_randall_munroe_pdf.pdf"


def main():
    pdf = pdfplumber.open(PDF)
    pages = pdf.pages
    print("PAGES:", len(pages))
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"

    if mode == "sample":
        for i in list(range(0, 12)):
            t = pages[i].extract_text() or ""
            head = t.strip().replace("\n", " | ")[:280]
            print(f"--- p{i}: {head}")

    elif mode == "dump":
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        for i in range(lo, hi):
            t = pages[i].extract_text() or ""
            print(f"\n===== PAGE {i} =====")
            print(t)

    elif mode == "questions":
        # Heuristic: reader questions often end with '?' and are short, and
        # are frequently attributed ("—Name"). Collect candidate prompts.
        hits = []
        for i, pg in enumerate(pages):
            t = pg.extract_text() or ""
            for line in t.split("\n"):
                s = line.strip()
                if s.endswith("?") and 8 <= len(s) <= 200:
                    hits.append((i, s))
        for i, s in hits:
            print(f"p{i}\t{s}")
        print(f"\nTOTAL QUESTION-LIKE LINES: {len(hits)}")


if __name__ == "__main__":
    main()
