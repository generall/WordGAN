

# WordGAN 

(Work in progress)

Generative Adversarial Net for generating replacements for distinct words in sentences.

## Required data

* **Word Embeddings** - regular word embeddings like word2vec, GloVe, fastText. 
ELMo, BERT and other context embeddings could not be used in this configuration
* **Word dictionary** - list of good words which could be used as replacement words
    * First 100_000 of Peter Norvig's compilation of the [1/3 million most frequent English words](http://norvig.com/ngrams/count_1w.txt).
    
* **Train corpora** - large text collection to train on.
    * For example [wikitext-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)   



## Train model with downloading all dependencies

* [colab](https://colab.research.google.com/drive/1E6zvwg5-Z8EG7S6KxcrwGxAGzJtDUgJ2)


# Overview

About sequential GANs - [Goodfellow's comment](https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/)

Setup
- Why generate synonyms with GANs
    - Language models don't optimized to use information about replaced words
- Why generator should have bigger context
    - cause it should force generator to use information from existing word
    - In other case it discriminator itself works exactly like language model - decide which sentences are probable
    and which are not. But if generator will use same information as discriminator
    it may start working as LM as well. To prevent this scenario we give
    discriminator more information about the context, so it enforces generator
    to use information from the target word.


