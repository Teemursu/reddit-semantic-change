# "This is lit, fam": Diachronic word embeddings and classifying semantic change

Code for my BA thesis (Undergraduate Thesis, English Philology, Department of
Languages, Faculty of Arts, University of Helsinki, **May 2020**).

- **Author:** Teemu Pöyhönen
- **Supervisor:** Joseph Flanagan
- **Thesis PDF:** [thesis/ba_thesis_teemu_poyhonen.pdf](thesis/ba_thesis_teemu_poyhonen.pdf)
  (also at <https://teemursu.github.io/assets/documents/ba_thesis_teemu_poyhonen.pdf>)

## Summary

The thesis asks whether word embeddings can give evidence of *short-term*
semantic change in contemporary English, and whether Bloomfield's (1933)
traditional categories (narrowing/broadening, elevation/degeneration, metaphor,
substitution) can be operationalised with them. A sample of Reddit comments
2006-2019 (Pushshift; up to 2M comments per month, one slice per year) was used
to train diachronic word2vec embeddings aligned with **TWEC -- Temporal Word
Embeddings with a Compass** (Di Carlo, Bianchi & Palmonari 2019). For the
target words *swipe, fam, noob, lit, sick* and *toxic* the analysis looks at the
10 nearest neighbours per year, a 2-D PCA projection of the target's yearly
vectors against its neighbours in the 2019 space, and the mean cosine
similarity between the target and three hand-picked words per sense (e.g.
*swipe* ~ debit/credit/card vs. tapping/scroll/rotate). Results suggest
metaphorical change for *swipe* and *toxic*, elevation for *noob, lit, sick*
and broadening for *fam*; the categories fit short time frames unevenly.

## Repository structure

```
prepare_data.py            sample + preprocess Pushshift dumps -> monthly/yearly slices -> compass text
train_model.py             train the TWEC compass and one aligned model per slice
compass/twec.py            TWEC on stock gensim 4 (port, see below)
compass/twec_original.py   the 2019 TWEC class used at the time (needs a patched gensim 3 fork)
compass/LICENSE_TWEC       MIT licence of TWEC
analysis/semantic_change.py  neighbours, sense-similarity curves, PCA plot (reconstruction)
test_models.py             exploratory nearest-neighbour checks on the compass (Dec 2019)
scripts/make_toy_corpus.py synthetic drifting corpus for smoke tests
results/figures_2020/      figures produced in spring 2020 (recovered from the old laptop)
thesis/                    submitted thesis PDF
third_party/histwords_viz/ histwords plotting scripts copied in 2019 (reference only, not runnable)
data/README.md             where the data comes from / where copies are on R:
```

## Setup (Python 3.12)

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

### Why TWEC is ported, not installed

TWEC (<https://github.com/valedica/twec>) is not on PyPI; it and its successor
`cade` require a patched gensim 3.x fork (`valedica/gensim`) that adds a
`learn_hidden` switch to Word2Vec. On Python 3.12 that fork installs only as a
source build without its C extension and fails to import (old `smart_open` /
`scipy` APIs). `compass/twec.py` re-implements the same algorithm on stock
gensim 4: the compass is a normal CBOW `Word2Vec` on the concatenated corpus;
each slice is trained as tagless PV-DM `Doc2Vec` (`dm_mean=1`, i.e. exactly
CBOW) with the vocabulary trimmed to the compass, the output layer `syn1neg`
copied from the compass and frozen through gensim's Doc2Vec
`learn_hidden=False` path. The API (`TWEC(...)`, `train_compass`,
`train_slice`) matches the original; slices are saved as gensim `KeyedVectors`
(`*.kv`). Only CBOW slices are supported (the thesis uses CBOW).

## How to run

```bash
# 1. data (see data/README.md): Pushshift dumps in data/raw/RC_YYYY-MM.xz
python prepare_data.py --raw-dir data/raw --per-month 2000000 --by-year
#    -> data/comments_monthly/, data/slices/comments-YYYY.txt, data/compass/compass.txt

# 2. train (thesis settings are the defaults: CBOW, 300 dims, 10 epochs, window 5, ns 5, min_count 2)
python train_model.py --compass data/compass/compass.txt --slices data/slices --out models

# 3. analyse a word
python analysis/semantic_change.py swipe --models models \
    --sense card=debit,credit,card --sense ui=tapping,scroll,rotate
```

Smoke test without Reddit data (runs in seconds):

```bash
python scripts/make_toy_corpus.py --out data/toy/slices
python prepare_data.py --slice-dir data/toy/slices --compass data/toy/compass.txt
python train_model.py --compass data/toy/compass.txt --slices data/toy/slices --out data/toy/models --size 50 --workers 1
python analysis/semantic_change.py swipe --models data/toy/models --out data/toy/analysis \
    --sense card=debit,credit,card --sense ui=tap,scroll,screen
```

On the toy corpus the *card* sense of *swipe* falls from ~0.78 to ~-0.09 and the
*ui* sense rises from ~-0.12 to ~0.79 across four slices, while control words
(*card, screen*) keep a cross-slice self-similarity of ~0.9, which shows the
slices are aligned.

## Code vs. thesis

| Thesis (May 2020) | Code in this repo |
|---|---|
| up to 2M random comments / month | original script: first ~1M of a p=0.5 sample per month (biased towards the start of each month); now `--per-month`, `--keep-prob` |
| yearly slices 2006-2019 | original script: monthly slices 2006-2017; `--by-year` and default years 2006-2019 added |
| CBOW, 300 dims, 10 iterations, window 5, negative 5, min_count 2 | original `train_model.py`: window 10, ns 10 (others the same); defaults now follow the thesis |
| top-10 neighbours, PCA in 2019 space, mean similarity of 3-word senses | analysis scripts not found anywhere; `analysis/semantic_change.py` is a reconstruction from section 3.5 |
| frequency plot (Fig. 1), sense curves with exponential trend lines (Figs 4, 6, ...) | frequency counts and trend fitting not reconstructed; the 2020 figures are in `results/figures_2020/` |

The thesis models and the yearly corpus were not found on R:, so the thesis
figures cannot be regenerated without re-downloading Pushshift data.

## History

Started in Dec 2019 as a fork of `valedica/twec`, so the git history begins
with the TWEC authors' commits (2018-2019). The Reddit text files that were
committed in `comment_data/` (~500 MB) and TWEC's example arXiv data/models
were removed from the history when this repo was consolidated in 2026.

## Citation

Pöyhönen, T. (2020). *"This is lit, fam": Diachronic word embeddings and
classifying semantic change.* BA thesis, University of Helsinki.
<https://teemursu.github.io/assets/documents/ba_thesis_teemu_poyhonen.pdf>

Please also cite TWEC: Di Carlo, V., Bianchi, F., & Palmonari, M. (2019).
Training Temporal Word Embeddings with a Compass. *AAAI 2019*, 33(01), 6326-6334.
<https://doi.org/10.1609/aaai.v33i01.33016326>
