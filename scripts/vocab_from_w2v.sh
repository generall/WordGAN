#!/usr/bin/env bash
# This script builds AllenNLP compatible vocabulary file out of W2V model


VOCAB_FILE=${VOCAB_FILE:-'./data/tokens_vocab.txt'}
MODEL_TXT_FILE=${MODEL_TXT_FILE:-'./data/model.txt'}

echo '@@PADDING@@' > ${VOCAB_FILE}
echo '@@UNKNOWN@@' >> ${VOCAB_FILE}

cut -f 1 -d ' ' ${MODEL_TXT_FILE} | tail -n +2 >> ${VOCAB_FILE}

