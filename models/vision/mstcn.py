"""Multi-stage temporal convolutional network for framewise event segmentation.

This is the temporal model used by the unified model 0010 pipeline.
"""
from mxnet.gluon import HybridBlock, nn


def _validate_temporal_parameters(channels, num_layers, kernel_size, dropout):
    if channels < 1:
        raise ValueError('channels must be positive')
    if num_layers < 1:
        raise ValueError('num_layers must be positive')
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer')
    if not 0.0 <= dropout < 1.0:
        raise ValueError('dropout must be in [0, 1)')


class DilatedResidualLayer(HybridBlock):
    """One dilated temporal convolution followed by a residual projection."""

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.5, **kwargs):
        super(DilatedResidualLayer, self).__init__(**kwargs)
        padding = dilation * (kernel_size - 1) // 2
        with self.name_scope():
            self.dilated_conv = nn.Conv1D(
                channels, kernel_size=kernel_size, padding=padding,
                dilation=dilation)
            self.projection = nn.Conv1D(channels, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

    def hybrid_forward(self, F, x):
        residual = x
        x = F.relu(self.dilated_conv(x))
        x = self.projection(x)
        x = self.dropout(x)
        return residual + x


class DualDilatedResidualLayer(HybridBlock):
    """MS-TCN++ dual-dilation layer used by the prediction generation stage."""

    def __init__(self, channels, dilation_a, dilation_b, kernel_size=3,
                 dropout=0.5, **kwargs):
        super(DualDilatedResidualLayer, self).__init__(**kwargs)
        padding_a = dilation_a * (kernel_size - 1) // 2
        padding_b = dilation_b * (kernel_size - 1) // 2
        with self.name_scope():
            self.conv_a = nn.Conv1D(
                channels, kernel_size=kernel_size, padding=padding_a,
                dilation=dilation_a)
            self.conv_b = nn.Conv1D(
                channels, kernel_size=kernel_size, padding=padding_b,
                dilation=dilation_b)
            self.fusion = nn.Conv1D(channels, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

    def hybrid_forward(self, F, x):
        residual = x
        branch_a = F.relu(self.conv_a(x))
        branch_b = F.relu(self.conv_b(x))
        x = F.concat(branch_a, branch_b, dim=1)
        x = self.fusion(x)
        x = self.dropout(x)
        return residual + x


class PredictionGeneration(HybridBlock):
    """Initial MS-TCN++ stage operating on pre-extracted visual features."""

    def __init__(self, num_classes, channels=64, num_layers=7,
                 kernel_size=3, dropout=0.5, **kwargs):
        super(PredictionGeneration, self).__init__(**kwargs)
        _validate_temporal_parameters(
            channels, num_layers, kernel_size, dropout)
        with self.name_scope():
            self.input_projection = nn.Conv1D(channels, kernel_size=1)
            self.layers = nn.HybridSequential()
            for layer_index in range(num_layers):
                self.layers.add(DualDilatedResidualLayer(
                    channels,
                    dilation_a=2 ** layer_index,
                    dilation_b=2 ** (num_layers - 1 - layer_index),
                    kernel_size=kernel_size,
                    dropout=dropout))
            self.classifier = nn.Conv1D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.input_projection(x)
        x = self.layers(x)
        return self.classifier(x)


class RefinementStage(HybridBlock):
    """Refine the previous stage's per-frame class probabilities."""

    def __init__(self, num_classes, channels=64, num_layers=7,
                 kernel_size=3, dropout=0.5, **kwargs):
        super(RefinementStage, self).__init__(**kwargs)
        _validate_temporal_parameters(
            channels, num_layers, kernel_size, dropout)
        with self.name_scope():
            self.input_projection = nn.Conv1D(channels, kernel_size=1)
            self.layers = nn.HybridSequential()
            for layer_index in range(num_layers):
                self.layers.add(DilatedResidualLayer(
                    channels, dilation=2 ** layer_index,
                    kernel_size=kernel_size, dropout=dropout))
            self.classifier = nn.Conv1D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.input_projection(x)
        x = self.layers(x)
        return self.classifier(x)


class MultiStageTemporalConvNet(HybridBlock):
    """MS-TCN++ style network returning logits from every prediction stage.

    Input shape is ``(batch, time, ...)``. Any dimensions after time are
    flattened, so saved DenseNet feature maps and flat feature vectors are both
    accepted. Output shape is ``(batch, stages, classes, time)``.
    """

    def __init__(self, num_classes, num_stages=3, channels=64, num_layers=7,
                 kernel_size=3, dropout=0.5, **kwargs):
        super(MultiStageTemporalConvNet, self).__init__(**kwargs)
        if num_classes < 2:
            raise ValueError('num_classes must be at least two')
        if num_stages < 1:
            raise ValueError('num_stages must be positive')
        _validate_temporal_parameters(
            channels, num_layers, kernel_size, dropout)

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.refinement_receptive_field = 1 + (kernel_size - 1) * (
            2 ** num_layers - 1)
        dual_dilation_sum = sum(max(
            2 ** layer_index,
            2 ** (num_layers - 1 - layer_index))
            for layer_index in range(num_layers))
        self.prediction_receptive_field = (
            1 + (kernel_size - 1) * dual_dilation_sum)
        self.per_stage_receptive_field = max(
            self.prediction_receptive_field,
            self.refinement_receptive_field)

        with self.name_scope():
            self.prediction_generation = PredictionGeneration(
                num_classes, channels=channels, num_layers=num_layers,
                kernel_size=kernel_size, dropout=dropout)
            self.refinement_stages = nn.HybridSequential()
            for _ in range(num_stages - 1):
                self.refinement_stages.add(RefinementStage(
                    num_classes, channels=channels, num_layers=num_layers,
                    kernel_size=kernel_size, dropout=dropout))

    def hybrid_forward(self, F, x):
        x = F.reshape(x, shape=(0, 0, -1))
        x = F.swapaxes(x, 1, 2)  # NTC -> NCT

        output = self.prediction_generation(x)
        stage_outputs = [output]
        for stage in self.refinement_stages:
            output = stage(F.softmax(output, axis=1))
            stage_outputs.append(output)
        return F.stack(*stage_outputs, axis=1)
