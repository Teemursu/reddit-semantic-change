"""Temporal Word Embeddings with a Compass (TWEC) on stock gensim 4.

Minimal port of the TWEC class by Valerio Di Carlo, Federico Bianchi and
Matteo Palmonari (https://github.com/valedica/twec, MIT licence, see
compass/LICENSE_TWEC; the 2019 version used for the thesis is kept in
compass/twec_original.py).

Why a port: the original code needs a patched gensim 3.x fork
(github.com/valedica/gensim) that adds a ``learn_hidden`` switch to
Word2Vec. That fork has no wheels and does not import on Python 3.12
(old smart_open / scipy APIs, Cython extension needs a C compiler), and
the successor package ``cade`` depends on the same fork.

How it works here: stock gensim 4 already has a ``learn_hidden`` switch,
but only in the Doc2Vec Cython routines. A Doc2Vec model in PV-DM mode with
``dm_mean=1`` and documents that carry *no tags* is exactly CBOW Word2Vec
(context = mean of the window words). So:

1. the compass is an ordinary gensim ``Word2Vec`` (CBOW or SG) trained on the
   concatenation of all slices -- identical to the original;
2. each slice is a ``_FrozenHiddenCBOW`` (Doc2Vec subclass) whose vocabulary is
   trimmed to the compass vocabulary, whose output layer (``syn1neg``) is
   copied from the compass and kept frozen (``learn_hidden=False``), and whose
   input vectors are trained from random init (TWEC ``init_mode="hidden"``).

Slices therefore live in the compass's coordinate system and can be compared
directly with cosine similarity. Only CBOW + negative sampling (the setting
used in the thesis, ``sg=0``) is supported for slices.
"""

import logging
import os

import numpy as np
from gensim import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec_inner import train_document_dm
from gensim.models.word2vec import LineSentence, PathLineSentences, Word2Vec

logger = logging.getLogger(__name__)


class _FrozenHiddenCBOW(Doc2Vec):
    """CBOW word2vec whose hidden->output weights (syn1neg) are not updated."""

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            tally += train_document_dm(
                self, doc.words, [], alpha, work, neu1,
                learn_doctags=False, learn_words=True, learn_hidden=False,
                doctag_vectors=self.dv.vectors, doctags_lockf=self.dv.vectors_lockf,
            )
        return tally, self._raw_word_count(job)


class _Tagless:
    """Wrap a sentence iterable as TaggedDocuments without tags (restartable)."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for words in self.sentences:
            yield TaggedDocument(words, [])


class TWEC:
    """Handles alignment between multiple slices of temporal text.

    Same constructor arguments as the original TWEC class (size, sg, siter,
    diter, ns, window, alpha, min_count, workers, opath, init_mode).
    """

    def __init__(self, size=100, sg=0, siter=5, diter=5, ns=10, window=5, alpha=0.025,
                 min_count=5, workers=2, test="test", opath="model", init_mode="hidden",
                 seed=1):
        if init_mode not in ("hidden", "both"):
            raise ValueError("init_mode must be 'hidden' or 'both' in this port")
        self.size = size
        self.sg = sg
        self.trained_slices = dict()
        self.gvocab = {}
        self.static_iter = siter
        self.dynamic_iter = diter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        self.workers = workers
        self.test = test
        self.opath = opath
        self.init_mode = init_mode
        self.seed = seed
        self.compass = None
        os.makedirs(self.opath, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.opath, "log.txt"),
                            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    # -- compass ---------------------------------------------------------
    def train_compass(self, compass_text, overwrite=False):
        """Train (or load) the atemporal compass on a file or a folder of files."""
        path = os.path.join(self.opath, "compass.model")
        if os.path.isfile(path) and not overwrite:
            self.compass = Word2Vec.load(path)
            print("Compass loaded from file.")
        else:
            sentences = PathLineSentences(compass_text)
            sentences.input_files = [s for s in sentences.input_files
                                     if not os.path.basename(s).startswith(".")]
            print("Training the compass.")
            self.compass = Word2Vec(
                sentences, sg=self.sg, vector_size=self.size, alpha=self.static_alpha,
                epochs=self.static_iter, negative=self.negative, hs=0, window=self.window,
                min_count=self.min_count, workers=self.workers, seed=self.seed)
            self.compass.save(path)
        self.gvocab = self.compass.wv.key_to_index

    # -- slices ----------------------------------------------------------
    def _trim_rule(self, word, count, min_count):
        return utils.RULE_KEEP if word in self.gvocab else utils.RULE_DISCARD

    def train_slice(self, slice_text, save=True):
        """Train one temporal slice aligned to the compass. Returns the model's KeyedVectors."""
        if self.compass is None:
            raise RuntimeError("Missing compass: call train_compass() first")
        if self.sg:
            raise NotImplementedError("Slice training is implemented for CBOW (sg=0) only")
        print("Training temporal embeddings: slice {}.".format(slice_text))
        docs = _Tagless(LineSentence(slice_text))
        model = _FrozenHiddenCBOW(
            dm=1, dm_mean=1, vector_size=self.size, window=self.window, alpha=self.dynamic_alpha,
            min_count=self.min_count, negative=self.negative, hs=0, workers=self.workers,
            epochs=self.dynamic_iter, seed=self.seed)
        model.build_vocab(docs, trim_rule=self._trim_rule)
        idx = [self.compass.wv.key_to_index[w] for w in model.wv.index_to_key]
        model.syn1neg = self.compass.syn1neg[idx].copy()
        if self.init_mode == "both":
            model.wv.vectors = self.compass.wv.vectors[idx].copy()
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

        name = os.path.splitext(os.path.basename(slice_text))[0]
        kv = model.wv
        self.trained_slices[name] = kv
        if save:
            kv.save(os.path.join(self.opath, name + ".kv"))
        return kv
