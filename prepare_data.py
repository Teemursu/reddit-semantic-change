"""Build the sampled, preprocessed Reddit corpus and the TWEC compass file.

Step 1 (optional, --raw-dir): read monthly Pushshift comment dumps
(RC_YYYY-MM.{xz,bz2,zst}, one JSON object per line), keep a random sample of
non-empty, non-deleted comment bodies, preprocess each with
gensim.utils.simple_preprocess (lowercase + tokenise; stopwords kept) and write
one comment per line to <month-dir>/comments-YYYY-MM.txt.

Step 2 (--by-year): concatenate the monthly files into yearly slices
<slice-dir>/comments-YYYY.txt (the thesis uses yearly slices 2006-2019).

Step 3: concatenate all slices into the compass file (the atemporal corpus TWEC
trains its compass on).

Original 2019 script: interactive prompts, Windows paths, 1M comments/month
sampled with p=0.5 from 2006-2017. The thesis (May 2020) describes 2M
comments/month and yearly slices 2006-2019; use --per-month 2000000 --by-year.
"""
import argparse
import bz2
import glob
import json
import lzma
import os
import random

from gensim.utils import simple_preprocess


def open_dump(path):
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    if path.endswith(".zst"):
        import io
        import zstandard  # pip install zstandard (newer Pushshift dumps)
        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, encoding="utf-8")


def get_sentences(raw_dir, out_dir, year, month, per_month, keep_prob, rng):
    """Sample up to `per_month` preprocessed comment bodies from RC_<year>-<month>."""
    candidates = glob.glob(os.path.join(raw_dir, "RC_%s-%s.*" % (year, month)))
    if not candidates:
        print("...no dump for %s-%s in %s, skipping" % (year, month, raw_dir))
        return
    filename = candidates[0]
    out_path = os.path.join(out_dir, "comments-%s-%s.txt" % (year, month))
    print("...Reading the file '%s'..." % filename)
    count = 0
    with open_dump(filename) as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if rng.random() >= keep_prob:  # semi-random sample, as in the original script
                continue
            body = json.loads(line).get("body", "")
            if body in ("", "[deleted]", "[removed]"):
                continue
            tokens = simple_preprocess(body)
            if not tokens:
                continue
            out.write(" ".join(tokens) + "\n")
            count += 1
            if count >= per_month:
                break
    print("...wrote %d comments to %s" % (count, out_path))


def merge_by_year(month_dir, slice_dir):
    os.makedirs(slice_dir, exist_ok=True)
    years = sorted({os.path.basename(p)[9:13] for p in glob.glob(os.path.join(month_dir, "comments-*-*.txt"))})
    for year in years:
        out_path = os.path.join(slice_dir, "comments-%s.txt" % year)
        with open(out_path, "wb") as out:
            for p in sorted(glob.glob(os.path.join(month_dir, "comments-%s-*.txt" % year))):
                with open(p, "rb") as f:
                    out.write(f.read())
        print("yearly slice", out_path)


def get_compass(slice_dir, compass_path):
    """TWEC needs a compass file containing all of the processed text."""
    read_files = sorted(glob.glob(os.path.join(slice_dir, "*.txt")))
    os.makedirs(os.path.dirname(compass_path) or ".", exist_ok=True)
    print("Creating the compass from %d files..." % len(read_files))
    n = 0
    with open(compass_path, "wb") as outfile:
        for path in read_files:
            with open(path, "rb") as infile:
                data = infile.read()
                outfile.write(data)
                n += data.count(b"\n")
    print("Compass created: %s (%d sentences)" % (compass_path, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", help="folder with Pushshift RC_YYYY-MM.* dumps; omit to skip sampling")
    ap.add_argument("--month-dir", default="data/comments_monthly", help="sampled monthly files")
    ap.add_argument("--slice-dir", default=None,
                    help="folder of slice files used for training (default: --month-dir, or data/slices with --by-year)")
    ap.add_argument("--compass", default="data/compass/compass.txt")
    ap.add_argument("--years", nargs="+", default=[str(y) for y in range(2006, 2020)])
    ap.add_argument("--months", nargs="+", default=["%02d" % m for m in range(1, 13)])
    ap.add_argument("--per-month", type=int, default=1000000)
    ap.add_argument("--keep-prob", type=float, default=0.5)
    ap.add_argument("--by-year", action="store_true", help="merge monthly files into yearly slices")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.raw_dir:
        os.makedirs(args.month_dir, exist_ok=True)
        rng = random.Random(args.seed)
        for year in args.years:
            for month in args.months:
                get_sentences(args.raw_dir, args.month_dir, year, month, args.per_month, args.keep_prob, rng)

    slice_dir = args.slice_dir or ("data/slices" if args.by_year else args.month_dir)
    if args.by_year:
        merge_by_year(args.month_dir, slice_dir)
    # After preparing the sliced corpus, create the atemporal text the compass is trained on.
    get_compass(slice_dir, args.compass)


if __name__ == "__main__":
    main()
