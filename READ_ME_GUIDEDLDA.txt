
This is a read me on how to get guided_lda working when the package will not successfully build wheels.

This is in no way meant to undermine the author of the guided_lda package or to replace it in anyway and is 
simply meant as a work around until the issue can be fully resolved.

1. Pull down the repository.

2. Install the original LDA package. 
	https://pypi.org/project/lda/
	
3. Drop the *.py files from the GuidedLDA_WorkAround repo in the lda folder under site-packages for your specific enviroment.

4. Profit...

EXAMPLE:


import numpy as np
from lda import guidedlda as glda
<s>from lda import glda_datasets as gldad
X = gldad.load_data(gldad.NYT)
vocab = gldad.load_vocab(gldad.NYT)</s>

Update courtesy of @senjed
import lda.datasets as gldad
X = gldad.load_reuters()
vocab = gldad.load_reuters_vocab()

word2id = dict((v, idx) for idx, v in enumerate(vocab))
print(X[:10])




print("TESTING....")

seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                   ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                   ['music', 'write', 'art', 'book', 'world', 'film'],
                   ['political', 'government', 'leader', 'official', 'state', 'country',
                    'american','case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

model = glda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)





n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
# Topic 0: game play team win season player second point start victory
# Topic 1: company percent market price business sell executive pay plan sale
# Topic 2: play life man music place write turn woman old book
# Topic 3: official government state political leader states issue case member country
# Topic 4: school child city program problem student state study family group
