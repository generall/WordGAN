

# WordGAN 

(Work in progress)

Generative Adversarial Net for generating replacements for distinct words in sentences.

## Required data

* **Word Embeddings** - regular word embeddings like word2vec, GloVe, fastText. 
ELMo, BERT and other context embeddings could not be used in this configuration
* **Word dictionary** - list of good words which could be used as replacement words
    * First 100_000 of Peter Norvig's compilation of the [1/3 million most frequent English words](http://norvig.com/ngrams/count_1w.txt).
    
* **Train corpora** - large text collection to train on.


## Build v2w vectors

We need reverse w2v for generator. 
We can do it by normalizing word embeddings.

Use following command:

* Build list of top-frequent words, presenting in word2vec

```bash
comm -12 \
<(cut -f 1 -d ' ' ./data/model.txt | sort ) \
<(cat ./data/count_1w.txt | cut -f 1 | sort) > ./data/common.txt
```
