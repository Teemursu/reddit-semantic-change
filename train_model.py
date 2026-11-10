"""Train the TWEC compass and one aligned word2vec model per time slice.

Defaults follow the thesis (section 3.5): CBOW, 300 dimensions, 10 iterations,
window 5, 5 negative samples, min_count 2. (The Dec 2019 version of this
script used window=10, ns=10.)

    python train_model.py --compass data/compass/compass.txt --slices data/slices --out models
"""
import argparse
import glob
import os

from compass.twec import TWEC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compass", default="data/compass/compass.txt", help="compass text file (or folder)")
    ap.add_argument("--slices", default="data/slices", help="folder of slice .txt files")
    ap.add_argument("--out", default="models")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--siter", type=int, default=10, help="compass (static) epochs")
    ap.add_argument("--diter", type=int, default=10, help="slice (dynamic) epochs")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ns", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="retrain the compass even if it exists")
    args = ap.parse_args()

    print("Declaring the TWEC object...")
    # siter is the number of iterations of the compass, diter of each slice
    aligner = TWEC(size=args.size, siter=args.siter, diter=args.diter, workers=args.workers,
                   min_count=args.min_count, window=args.window, ns=args.ns, sg=0, opath=args.out)
    # train the compass: the text should be the concatenation of the text from the slices
    print("Training the compass...")
    aligner.train_compass(args.compass, overwrite=args.overwrite)

    # now train the slices; they will be already aligned
    read_files = sorted(glob.glob(os.path.join(args.slices, "*.txt")))
    for file in read_files:
        print("Training the slice for " + str(file))
        aligner.train_slice(file, save=True)
    print("Done: %d slices in %s" % (len(read_files), args.out))


if __name__ == "__main__":
    main()
