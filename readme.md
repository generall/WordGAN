

# WordGAN 

(Work in progress)

Generative Adversarial Net for generating replacements for distinct words in sentences.

## Required data

* **Word Embeddings** - regular word embeddings like word2vec, GloVe, fastText. 
ELMo, BERT and other context embeddings could not be used in this configuration
* **Word dictionary** - list of good words which could be used as replacement words
    * First 100_000 of Peter Norvig's compilation of the [1/3 million most frequent English words](http://norvig.com/ngrams/count_1w.txt).
    
* **Train corpora** - large text collection to train on.


