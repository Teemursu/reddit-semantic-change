# Data

Nothing in this folder is committed except this file.

## Source

All Reddit comments 2006-2019 from the **Pushshift** monthly comment dumps
(`RC_YYYY-MM.{bz2,xz,zst}`, one JSON object per comment; formerly at
`https://files.pushshift.io/reddit/comments/`, which is offline since 2023 --
the same dumps are mirrored as academic torrents, e.g. "Reddit comments/submissions
2005-06 to 2023-12"). All subreddits were used; only the `body` field is kept,
so no user names end up in the corpus.

Pushshift data is third-party Reddit content; check Reddit's terms before
redistributing any derived text.

## Thesis corpus (May 2020)

Per thesis section 3.1: a random sample of up to 2 million comments per month,
merged into one file per year (2006-2019; 2006-2009 have fewer comments).
Preprocessing: `gensim.utils.simple_preprocess` (lowercase, tokenise, stopwords
kept). Table 1 of the thesis lists ~24M comments, ~600-800M tokens per year from
2012 on.

## Local copies on R:

| What | Where | Notes |
|---|---|---|
| Sampled monthly corpus, 2005-12 .. 2008-09 (34 files, 1.3 GB) | `R:\Memories\Old laptop (läppäri)\Users\temep\Dropbox\Reddit corpus\comments-YYYY-MM.txt` | the Dec 2019 "1-mil-comm-per-month" corpus (also what was shared via Dropbox); first 25 months were committed to the old GitHub repo (`comment_data/`, removed from this repo's history) |
| Torrent files for the raw dumps | `R:\Memories\Old laptop (läppäri)\Users\temep\Downloads\RC_2015-01.bz2.torrent`, `reddit_data-85a5bd50...torrent` | raw dumps themselves not found on R: |
| Yearly 2006-2019 thesis corpus, trained compass and slice models | **not found on R:** | most likely produced on a remote machine (CSC) in spring 2020 |

## Layout expected by the scripts

```
data/raw/RC_YYYY-MM.xz            # Pushshift dumps (input to prepare_data.py --raw-dir)
data/comments_monthly/comments-YYYY-MM.txt
data/slices/comments-YYYY.txt     # yearly slices (training input)
data/compass/compass.txt          # concatenation of all slices
models/compass.model, models/comments-YYYY.kv
```

For a smoke test without any Reddit data, generate a synthetic corpus with
`python scripts/make_toy_corpus.py --out data/toy/slices`.
