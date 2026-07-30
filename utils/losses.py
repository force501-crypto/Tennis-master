"""Loss functions and class-weight helpers for vision training."""
import numpy as np
import mxnet as mx

from mxnet.gluon.loss import Loss


def get_class_weights(class_counts, method='none', beta=0.9999):
    """Return mean-normalised class weights for the requested weighting method."""
    counts = np.asarray(class_counts, dtype=np.float32)
    if np.any(counts <= 0):
        raise ValueError('All classes must have at least one training sample: {}'.format(class_counts))

    if method == 'none':
        weights = np.ones_like(counts)
    elif method == 'inverse_sqrt':
        weights = 1.0 / np.sqrt(counts)
    elif method == 'effective_num':
        if not 0.0 <= beta < 1.0:
            raise ValueError('beta must be in [0, 1) for effective-number weighting')
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError('Unknown class weighting method: {}'.format(method))

    return weights / weights.mean()


class FocalSoftmaxCrossEntropyLoss(Loss):
    """Sparse softmax cross entropy with optional class weights and focal scaling."""

    def __init__(self, class_weights, gamma=0.0, axis=-1, batch_axis=0, **kwargs):
        super(FocalSoftmaxCrossEntropyLoss, self).__init__(
            weight=None, batch_axis=batch_axis, **kwargs)
        if gamma < 0:
            raise ValueError('gamma must be non-negative')
        self._gamma = gamma
        self._axis = axis
        self.class_weights = self.params.get_constant(
            'class_weights', mx.nd.array(class_weights, dtype='float32'))

    def hybrid_forward(self, F, pred, label, class_weights):
        label = label.astype('int32')
        log_probabilities = F.log_softmax(pred, axis=self._axis)
        log_pt = F.pick(log_probabilities, label, axis=self._axis, keepdims=False)
        pt = F.exp(log_pt)
        sample_weights = F.take(class_weights, label)
        focal_scale = F.power(1.0 - pt, self._gamma)
        return -sample_weights * focal_scale * log_pt
