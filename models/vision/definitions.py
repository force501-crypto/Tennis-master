"""
Defines the model specifications
"""
import mxnet as mx
from mxnet.gluon import HybridBlock, nn

from utils.layers import TimeDistributed


class FrameModel(HybridBlock):
    def __init__(self, backbone, num_classes=-1, swap=False, **kwargs):
        """
        A framewise model (just the backbone CNN with a single dense layer to the classes)

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes
        """
        super(FrameModel, self).__init__(**kwargs)
        self.swap = swap
        with self.name_scope():
            self.backbone = backbone
            self.classes = None
            if num_classes > 0:
                self.classes = nn.Dense(num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        if self.swap:
            x = F.swapaxes(x, 1, 2)
        x = self.backbone(x)
        if self.classes:
            x = self.classes(x)
        return x


class TemporalPooling(HybridBlock):
    def __init__(self, model, num_classes=-1, pool='max', feats=False, **kwargs):
        """
        A temporal pooling model

        Args:
            model: the CNN model
            num_classes (int): the number of classes,
                               -1 meaning the model output is a softmax (default)
                               0 means we apply the pool between the model softmax
        """
        super(TemporalPooling, self).__init__(**kwargs)
        self.pool = pool
        self.feats = feats
        with self.name_scope():
            self.classes = None
            if model is not None:
                if num_classes == 0:
                    self.td = TimeDistributed(model.backbone)
                    self.classes = model.classes
                else:
                    self.td = TimeDistributed(model)
                    if num_classes > 0:
                        self.classes = nn.Dense(num_classes, flatten=True)
            else:
                self.classes = nn.Dense(num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        if not self.feats:
            x = self.td(x)
        if self.pool == 'mean':
            x = F.mean(x, axis=1)
        else:
            x = F.max(x, axis=1)
        if self.classes:
            x = self.classes(x)
        return x


class CNNRNN(HybridBlock):
    def __init__(self, model, num_classes=-1, hidden_size=128, type='gru', **kwargs):
        """
        A temporal CNN+RNN(GRU) model

        Args:
            model: the CNN model
            num_classes (int): the number of classes,
                               -1 meaning the model output is a softmax learnt from scratch (default)
                               0 means we take the backbone classes layer
            hidden_size (int): the hidden size of the GRU (default is 128)
            type (str): the unit type, either gur or lstm, gru default
        """
        super(CNNRNN, self).__init__(**kwargs)
        self.feats = model is None
        with self.name_scope():
            if model is not None:
                self.td = TimeDistributed(model.backbone)
            if type == 'lstm':
                self.rnn = mx.gluon.rnn.LSTM(hidden_size, layout="NTC", bidirectional=True)
            else:
                self.rnn = mx.gluon.rnn.GRU(hidden_size, layout="NTC", bidirectional=True)
            self.classes = None
            if num_classes == 0:
                self.classes = model.classes
            elif num_classes > 0:
                self.classes = nn.Dense(num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        if not self.feats:
            x = self.td(x)
        x = self.rnn(x)
        x = F.max(x, axis=1)
        if self.classes:
            x = self.classes(x)
        return x


class TemporalResidualBlock(HybridBlock):
    """A residual dilated temporal convolution block."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2, **kwargs):
        super(TemporalResidualBlock, self).__init__(**kwargs)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        padding = dilation * (kernel_size - 1) // 2
        with self.name_scope():
            self.conv1 = nn.Conv1D(channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1D(channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return F.relu(x + residual)


class TemporalConvNet(HybridBlock):
    """Classify the centre frame of a feature window with a dilated TCN."""

    def __init__(self, model, num_classes, window, hidden_size=128, num_layers=4,
                 kernel_size=3, dropout=0.2, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        if window < 2:
            raise ValueError('TCN requires a temporal window greater than one')
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        if hidden_size < 1:
            raise ValueError('hidden_size must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')
        self.feats = model is None
        self.center = window // 2
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(
            2 ** layer_index for layer_index in range(num_layers))
        with self.name_scope():
            if model is not None:
                self.td = TimeDistributed(model.backbone)
            self.input_projection = nn.Conv1D(hidden_size, kernel_size=1)
            self.temporal_blocks = nn.HybridSequential()
            for layer_index in range(num_layers):
                self.temporal_blocks.add(TemporalResidualBlock(
                    hidden_size, kernel_size=kernel_size,
                    dilation=2 ** layer_index, dropout=dropout))
            self.classes = nn.Dense(num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        if not self.feats:
            x = self.td(x)
        # Collapse any spatial feature dimensions while preserving N and T.
        x = F.reshape(x, shape=(0, 0, -1))
        x = F.swapaxes(x, 1, 2)  # NTC -> NCT for Conv1D
        x = self.input_projection(x)
        x = self.temporal_blocks(x)
        x = F.slice_axis(x, axis=2, begin=self.center, end=self.center + 1)
        x = F.squeeze(x, axis=2)
        return self.classes(x)


class Debug(HybridBlock):
    def __init__(self, **kwargs):
        """
        Useful model for debugging
        """
        super(Debug, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=4, kernel_size=2)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        return x


class TwoStreamModel(HybridBlock):
    def __init__(self, model_rgb, model_flow, num_classes, **kwargs):
        """
        A two stream model (just the model CNNs concatenated before a single dense layer to the classes)

        Args:
            model_rgb: the rgb model CNN model
            model_flow: the flow model CNN model
            num_classes (int): the number of classes
        """
        super(TwoStreamModel, self).__init__(**kwargs)
        with self.name_scope():
            self.features_rgb = model_rgb
            self.features_flow = model_flow
            self.classes = nn.Dense(num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        rgb = F.slice_axis(x, axis=-3, begin=0, end=3)
        flow = F.slice_axis(x, axis=-3, begin=3, end=6)
        flow = self.features_flow(flow)
        if self.features_rgb is not None:
            rgb = self.features_rgb(rgb)
            x = F.concat(rgb, flow, dim=-1)
        else:
            x = flow
        x = self.classes(x)
        return x


if __name__ == '__main__':
    # just for debugging
    from mxnet import gluon, autograd

    mod = Debug()
    td = TimeDistributed(mod)
    td.initialize()
    td.hybridize()
    mse_loss = gluon.loss.L2Loss()
    with autograd.record():
        out = td(mx.nd.ones((3, 2, 3, 2, 2)))
        loss = mse_loss(out, mx.nd.ones((3, 2, 4, 1, 1)))
    loss.backward()
