"""Neighbour, sense-similarity and PCA analysis of aligned TWEC slices.

Reconstruction of the analysis described in thesis section 3.5 (the original
2020 analysis scripts were not found on GitHub or in the local backups):

* for a target word, the 10 most similar words in every slice;
* a 2-D PCA projection of those neighbours' vectors in the most recent slice
  together with the target word's vector in every slice (cf. Hamilton et al.
  2016; thesis figures 5, 7, ...);
* for each "sense" given as three neighbour words, the mean cosine
  similarity between the target and those words per slice (thesis figures
  4, 6, ...).

    python analysis/semantic_change.py swipe --models models \
        --sense card=debit,credit,card --sense ui=tapping,scroll,rotate
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA


def load_slices(model_dir):
    slices = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.kv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        label = re.sub(r"^comments-", "", name)
        slices[label] = KeyedVectors.load(path)
    if not slices:
        raise SystemExit("no *.kv slice models in %s (run train_model.py first)" % model_dir)
    return slices


def neighbours(slices, word, topn):
    out = {}
    for label, kv in slices.items():
        out[label] = kv.most_similar(word, topn=topn) if word in kv else []
    return out


def sense_scores(slices, word, senses):
    rows = []
    for label, kv in slices.items():
        row = {"slice": label}
        for sense, words in senses.items():
            sims = [kv.similarity(word, w) for w in words if word in kv and w in kv]
            row[sense] = float(np.mean(sims)) if sims else float("nan")
        rows.append(row)
    return rows


def plot_pca(slices, word, nbrs, path):
    labels = list(slices)
    latest = slices[labels[-1]]
    context = sorted({w for lst in nbrs.values() for w, _ in lst if w in latest and w != word})
    points = [latest[w] for w in context] + [slices[l][word] for l in labels if word in slices[l]]
    names = context + ["%s_%s" % (word, l) for l in labels if word in slices[l]]
    xy = PCA(n_components=2).fit_transform(np.array(points))
    plt.figure(figsize=(10, 8))
    n_ctx = len(context)
    plt.scatter(xy[:n_ctx, 0], xy[:n_ctx, 1], c="grey", s=10)
    plt.plot(xy[n_ctx:, 0], xy[n_ctx:, 1], "-o", c="tab:red", ms=4)
    for (x, y), name in zip(xy, names):
        plt.annotate(name, (x, y), fontsize=8, color="tab:red" if name.startswith(word + "_") else "black")
    plt.title("PCA of '%s' across slices and its neighbours (latest slice)" % word)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_senses(rows, word, senses, path):
    plt.figure(figsize=(10, 5))
    xs = [r["slice"] for r in rows]
    for sense, words in senses.items():
        plt.plot(xs, [r[sense] for r in rows], "-o", label="%s (%s)" % (sense, ", ".join(words)))
    plt.ylabel("mean cosine similarity")
    plt.title("Sense similarity for '%s'" % word)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("word")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--sense", action="append", default=[], help="NAME=w1,w2,w3 (repeatable)")
    args = ap.parse_args()

    slices = load_slices(args.models)
    os.makedirs(args.out, exist_ok=True)
    nbrs = neighbours(slices, args.word, args.topn)
    for label, lst in nbrs.items():
        print(label, ", ".join("%s (%.2f)" % (w, s) for w, s in lst))
    plot_pca(slices, args.word, nbrs, os.path.join(args.out, "%s_pca.png" % args.word))

    if args.sense:
        senses = dict((s.split("=")[0], s.split("=")[1].split(",")) for s in args.sense)
        rows = sense_scores(slices, args.word, senses)
        with open(os.path.join(args.out, "%s_senses.csv" % args.word), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slice"] + list(senses))
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            print(r)
        plot_senses(rows, args.word, senses, os.path.join(args.out, "%s_senses.png" % args.word))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
