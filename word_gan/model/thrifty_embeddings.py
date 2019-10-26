import h5py
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Embedding


class ThriftyEmbedding(Embedding):
    """
    Same as Embedding, but don't actually saves weights if not trainable
    """

    def __init__(self, trainable: bool = True, weights_file: str = None, **kwargs):
        kwargs['pretrained_file'] = weights_file
        super(ThriftyEmbedding, self).__init__(trainable=trainable, **kwargs)
        self.trainable = trainable
        self.weights_file = weights_file

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(ThriftyEmbedding, self).state_dict(destination, prefix, keep_vars)

        if not self.trainable:
            state_dict.pop(prefix + 'weight')

        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        strict = strict and self.trainable  # restore weights only in case they are trainable
        return super(ThriftyEmbedding, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                   missing_keys, unexpected_keys, error_msgs)

    # Custom logic requires custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ThriftyEmbedding':  # type: ignore
        weights_file = params.get('weights_file')
        params.params['pretrained_file'] = weights_file
        obj = super(ThriftyEmbedding, cls).from_params(vocab, params)
        return obj
