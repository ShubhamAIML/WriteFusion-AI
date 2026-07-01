"""
Corpus Stats Generator
========================
Run this once on your training corpus to produce corpus_stats.json,
which the Flask app reads to display "words" / "sentences" in the UI.

Usage:
    python make_corpus_stats.py path/to/your_corpus.txt

This writes corpus_stats.json into the SAME folder as this script
(i.e. your lm_app/ folder) — keep it next to app.py.
"""

import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_corpus_stats.py path/to/corpus.txt")
        return

    corpus_path = Path(sys.argv[1])
    if not corpus_path.exists():
        print(f"[!] File not found: {corpus_path}")
        return

    lines = [l.strip() for l in corpus_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    total_words = sum(len(l.split()) for l in lines)

    stats = {
        "sentences": len(lines),
        "words": total_words,
    }

    out_path = Path(__file__).parent / "corpus_stats.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding='utf-8')

    print(f"Sentences : {stats['sentences']:,}")
    print(f"Words     : {stats['words']:,}")
    print(f"Saved to  : {out_path}")


if __name__ == '__main__':
    main()
