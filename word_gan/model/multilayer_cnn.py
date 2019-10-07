from typing import Optional, Tuple, List

from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation


@Seq2VecEncoder.register("multilayer_cnn")
class MultilayerCnnEncoder(Seq2VecEncoder):
    """
    A ``MultilayerCnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    pooling : ``str``, optional (default=``"max"``)
        What method to use for downsampling:
        * "max" - For Max Pooling
        * "avg" - For Average Pooling
    """

    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 layers: int = 1,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),  # pylint: disable=bad-whitespace
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None,
                 pooling='max') -> None:
        super(MultilayerCnnEncoder, self).__init__()
        self._pooling = pooling
        self._layers = layers
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name('relu')()
        self._output_dim = output_dim

        self._num_ngram_filters = len(self._ngram_filter_sizes)

        self._convolution_layers: List[List[Conv1d]] = []
        for layer_num in range(self._layers):
            layer_cnns = [Conv1d(in_channels=self._embedding_dim if layer_num == 0 else self._num_filters,
                                 out_channels=self._num_filters,
                                 kernel_size=ngram_size,
                                 padding=1)
                          for ngram_size in self._ngram_filter_sizes]

            self._convolution_layers.append(layer_cnns)

            for i, conv_layer in enumerate(layer_cnns):
                self.add_module(f'conv_layer_{layer_num}_{i}', conv_layer)

        pool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(pool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = pool_output_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        prev_tensors = [tokens] * self._num_ngram_filters

        for layer_num in range(self._layers):
            next_tensors = []
            for prev_tensor, convolution_layer in zip(prev_tensors, self._convolution_layers[layer_num]):
                next_tensors.append(self._activation(convolution_layer(prev_tensor)))

            prev_tensors = next_tensors

        if self._pooling == 'max':
            filter_outputs = [tensor.max(dim=2)[0] for tensor in prev_tensors]
        elif self._pooling == 'avg':
            filter_outputs = [tensor.mean(dim=2)[0] for tensor in prev_tensors]
        else:
            raise NotImplementedError(f"Pooling {self._pooling} is not implemented")

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        pool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(pool_output)
        else:
            result = pool_output
        return result
