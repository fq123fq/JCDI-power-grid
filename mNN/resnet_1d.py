import torch
from torch import nn

from .select import select_norm_layer, select_activation

def calc_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

class ResNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, rezero = False,
        kernel_size = 3, bottlneck_features = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if bottlneck_features is None:
            bottlneck_features = features

        self.block = nn.Sequential(
            nn.Conv1d(
                features, bottlneck_features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1,
            ),
            select_norm_layer(norm, bottlneck_features),
            select_activation(activ),

            nn.Conv1d(
                bottlneck_features, features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1
            ),
            select_norm_layer(norm, features),
        )

        self.block_out = select_activation(activ)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.block(x)
        z = x + self.re_alpha * y

        return self.block_out(z)

    def extra_repr(self):
        return f're_alpha = {self.re_alpha}'

class ResNetBlockv2(nn.Module):

    def __init__(
        self, features, activ, norm, rezero = False,
        kernel_size = 3, bottlneck_features = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if bottlneck_features is None:
            bottlneck_features = features

        self.block = nn.Sequential(
            select_norm_layer(norm, bottlneck_features),
            select_activation(activ),

            nn.Conv1d(
                features, bottlneck_features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1,
            ),

            select_norm_layer(norm, features),
            select_activation(activ),

            nn.Conv1d(
                bottlneck_features, features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1
            ),
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.block(x)
        z = x + self.re_alpha * y

        return self.block_out(z)

    def extra_repr(self):
        return f're_alpha = {self.re_alpha}'

class ResNetStem(nn.Module):

    def __init__(
        self, input_shape, features, norm = None,
        kernel_size = 4, padding = 0, stride = 4
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                input_shape[0], features,
                kernel_size = kernel_size, padding = padding, stride = stride
            ),
            select_norm_layer(norm, features),
        )

        self._input_shape  = input_shape
        self._output_shape = (
            features,
            calc_conv_output_size(input_shape[1], kernel_size, padding, stride)
        )

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return self.net(x)

class ResNetEncoder(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def _make_stem_block(self, block_spec, curr_shape):
        # pylint: disable=no-self-use
        block = ResNetStem(curr_shape, **block_spec)
        curr_shape = block.output_shape

        return (block, curr_shape)

    def _make_resample_block(self, block_spec, curr_shape):
        # pylint: disable=no-self-use
        if isinstance(block_spec, (list, tuple)):
            size, kwargs = block_spec
        else:
            size   = block_spec
            kwargs = {}

        features   = curr_shape[0]
        block      = nn.Upsample(size, **kwargs)
        curr_shape = (size, features)

        return (block, curr_shape)

    def _make_resnet_block(self, block_spec, curr_shape):
        if isinstance(block_spec, (list, tuple)):
            n_blocks, kwargs = block_spec
        else:
            n_blocks = block_spec
            kwargs   = {}

        features = curr_shape[0]
        block    = nn.Sequential(
            *[ ResNetBlock(
                features,
                activ  = self._activ,
                norm   = self._norm,
                rezero = self._rezero,
                **kwargs
            )
            for _ in range(n_blocks) ]
        )

        return (block, curr_shape)

    def _make_resnet_block_v2(self, block_spec, curr_shape):
        if isinstance(block_spec, (list, tuple)):
            n_blocks, kwargs = block_spec
        else:
            n_blocks = block_spec
            kwargs   = {}

        features = curr_shape[0]
        block    = nn.Sequential(
            ResNetBlockv2(
                features,
                activ  = self._activ,
                norm   = self._norm,
                rezero = self._rezero,
                **kwargs
            ) for _ in range(n_blocks)
        )

        return (block, curr_shape)

    def __init__(
        self, input_shape, block_specs, activ, norm, rezero = True
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        super().__init__()

        curr_shape  = input_shape
        self.blocks = nn.ModuleList()

        self._activ  = activ
        self._norm   = norm
        self._rezero = rezero

        block_idx    = 0
        skip_indices = set()
        skip_shapes  = []

        for (block_type, block_spec) in block_specs:
            if block_type == 'stem':
                block, curr_shape \
                    = self._make_stem_block(block_spec, curr_shape)

            elif block_type == 'resample':
                block, curr_shape \
                    = self._make_resample_block(block_spec, curr_shape)

            elif block_type == 'resnet':
                block, curr_shape \
                    = self._make_resnet_block(block_spec, curr_shape)

            elif block_type == 'resnet-v2':
                block, curr_shape \
                    = self._make_resnet_block_v2(block_spec, curr_shape)

            elif block_type == 'skip':
                skip_indices.add(block_idx)
                skip_shapes.append(curr_shape)
                continue

            else:
                raise ValueError(f"Unknown block type: {block_type}")

            self.blocks.append(block)
            block_idx += 1

        self._input_shape  = input_shape
        self._output_shape = curr_shape
        self._skip_shapes  = skip_shapes
        self._skip_indices = skip_indices

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def skip_indices(self):
        return self._skip_indices

    @property
    def skip_shapes(self):
        return self._skip_shapes

    def forward(self, x, return_skips = False):
        if return_skips:
            skips = []

        y = x

        for idx, block in enumerate(self.blocks):
            if return_skips and (idx in self._skip_indices):
                skips.append(y)

            y = block(y)

        if return_skips:
            return (y, skips)

        return y

