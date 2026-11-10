"""Quick sanity checks on a trained compass model (exploratory, Dec 2019).

    python test_models.py models/compass.model

Ported to gensim 4: the compass is loaded as a Word2Vec model; the unused
gensim test-data loads were dropped and the undefined `wv` was replaced by the
compass vectors. Words missing from the vocabulary are skipped.
"""
import sys

from gensim.models import Word2Vec

path = sys.argv[1] if len(sys.argv) > 1 else "models/compass.model"
compass_wv = Word2Vec.load(path).wv


def show(label, *args, **kwargs):
    words = list(kwargs.get("positive", [])) + list(kwargs.get("negative", [])) + [a for a in args if isinstance(a, str)]
    missing = [w for w in words if w not in compass_wv]
    if missing:
        print(label, "-- not in vocabulary:", missing)
        return
    print(label, compass_wv.most_similar(*args, **kwargs))


show("king - man + woman:", positive=['king', 'woman'], negative=['man'], topn=5)
print()
for w in ['noob', 'wtf', 'email', 'hangry']:
    show(w + ":", positive=[w], topn=5)
print()
show("Most similar to 'lol' ", positive=['lol'], topn=10)

pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
    ('dog', 'poop'),
    ('rock', 'hard')
]
for w1, w2 in pairs:
    if w1 in compass_wv and w2 in compass_wv:
        print('%r\t%r\t%.2f' % (w1, w2, compass_wv.similarity(w1, w2)))

show("feminism:", positive=['feminism'], topn=20)
show("mansplaining:", positive=['mansplaining'], topn=20)
show("binary + trans - encoding:", positive=['binary', 'trans'], negative=['encoding'], topn=20)
for w in ['wholesome', 'dank', 'flex', 'salty', 'woke', 'fam', 'swag']:
    show(w.capitalize() + ": ", positive=[w], topn=10)
