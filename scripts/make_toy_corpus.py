"""Write a tiny synthetic diachronic corpus for smoke-testing the pipeline.

Creates one preprocessed text file per year (one sentence per line, the same
format prepare_data.py produces) in which the target words drift between two
senses, mimicking the thesis findings for *swipe* (card payment -> touch-screen
gesture) and *fam* (family -> close friend). Not real Reddit data.

    python scripts/make_toy_corpus.py --out data/toy --years 2006 2009 2012 2015 --n 400
"""
import argparse
import os
import random

SENSES = {
    "swipe": (["card", "debit", "credit", "atm", "pay", "cashier"],
              ["screen", "scroll", "tap", "phone", "app", "gesture"]),
    "fam": (["family", "mum", "sis", "hubby", "stepdad", "cousin"],
            ["bro", "homies", "dawg", "bruh", "squad", "friend"]),
}
STABLE = {  # words whose contexts do not change, as controls
    "card": ["debit", "credit", "atm", "pay", "cashier", "bank"],
    "screen": ["scroll", "tap", "phone", "app", "gesture", "display"],
    "family": ["mum", "sis", "hubby", "stepdad", "cousin", "dinner"],
    "friend": ["bro", "homies", "dawg", "bruh", "squad", "party"],
}
FILLER = ("the a i you it is was that this and to of in on for with my your just "
          "really so like lol get got go went have had do did not yeah ok").split()


def sentence(rng, word, context):
    words = [word] + rng.sample(context, 3) + rng.sample(FILLER, 5)
    rng.shuffle(words)
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--years", nargs="+", type=int, default=[2006, 2009, 2012, 2015])
    ap.add_argument("--n", type=int, default=400, help="sentences per word per slice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i, year in enumerate(args.years):
        p_new = i / max(1, len(args.years) - 1)  # share of the new sense: 0 -> 1
        lines = []
        for word, (old, new) in SENSES.items():
            for _ in range(args.n):
                lines.append(sentence(rng, word, new if rng.random() < p_new else old))
        for word, ctx in STABLE.items():
            lines += [sentence(rng, word, ctx) for _ in range(args.n)]
        rng.shuffle(lines)
        path = os.path.join(args.out, "comments-%d.txt" % year)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", path, len(lines), "sentences")


if __name__ == "__main__":
    main()
