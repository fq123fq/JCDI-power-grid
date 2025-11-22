from torch import nn

from .resnet_1d import ResNetEncoder
from .powergrid_trans import PowerGridTransformer

class PowerGridHybrid(nn.Module):

    def __init__(
        self, traj_shape,n_events, n_params, resnet_params, transformer_params
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        basic_traj_encoder = ResNetEncoder(traj_shape, **resnet_params)
        #basic_traj_encoder1 = ResNetEncoder(traj_shape, **resnet_params)
        #basic_traj_encoder2 = ResNetEncoder(traj_shape, **resnet_params)
        #basic_traj_encoder3 = ResNetEncoder(traj_shape, **resnet_params)

        basic_traj_encoders = nn.ModuleList([
            ResNetEncoder(traj_shape, **resnet_params) for _ in range(n_events)
        ])
        
        
        #traj_encoder = nn.Sequential(
        #    basic_traj_encoder,
        #    nn.AdaptiveAvgPool1d(1),
        #    nn.Flatten()
        #)
        traj_encoders=basic_traj_encoders


        traj_enc_size = basic_traj_encoder.output_shape[0]
        traj_enc_num_each = basic_traj_encoder.output_shape[1]

        print("basic_traj_encoder.output_shape",basic_traj_encoder.output_shape)

        self.net = PowerGridTransformer(
            traj_encoders, traj_enc_size,traj_enc_num_each,n_events, n_params, **transformer_params
        )

    def forward(self, x, cond, time):
        # x    : (N, n_params)
        # cond : (N, C_cond, L_cond)
        # time : (N, 1)

        return self.net(x, cond, time)

