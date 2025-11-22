import torch
from torch import nn

class TimeEmbeddingSin(nn.Module):

    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self._features = features

        self._sin_in = nn.Linear(1, features)
        self._out    = nn.Sequential(
            nn.Linear(features, features),
            nn.GELU(),
        )

    def forward(self, t):
        # t : (N, 1)
        result = self._sin_in(t)
        result = torch.sin(result)
        return self._out(result)

class PowerGridTransformer(nn.Module):

    def __init__(
        self, traj_encoders, traj_enc_size,traj_enc_num_each,n_events, n_params, features, features_ffn,
        n_heads, n_layers, norm_first = True, time_embed = 'linear',
        **trans_kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        #self.traj_encoder      = traj_encoder
        self.traj_encoders      = traj_encoders

        #self.traj_enc_to_token = nn.Linear(traj_enc_size, features)
        self.traj_enc_to_tokens = nn.ModuleList([
            nn.Linear(traj_enc_size, features) for _ in range(traj_enc_num_each*n_events)
        ])
        self.traj_enc_num_each=traj_enc_num_each
        self.n_events=n_events

        self.n_params       = n_params
        self.param_encoders = nn.ModuleList([
            nn.Linear(1, features) for _ in range(n_params)
        ])
        self.param_decoders = nn.ModuleList([
            nn.Linear(features, 1) for _ in range(n_params)
        ])

        if time_embed == 'sin':
            self.time_encoder = TimeEmbeddingSin(features)
        elif time_embed == 'linear':
            self.time_encoder = nn.Linear(1, features)

        trans_layer = nn.TransformerEncoderLayer(
            d_model         = features,
            nhead           = n_heads,
            dim_feedforward = features_ffn,
            norm_first      = norm_first,
            batch_first     = True,
            **trans_kwargs
        )

        self.transformer = nn.TransformerEncoder(trans_layer, n_layers)

    def forward(self, x, cond, time):
        # x    : (N, n_params)
        # cond : (N, C_cond, L_cond)
        # time : (N)

        #print("x.shape",x.shape)

        # params : List[(N, 1)]
        params = x.split(1, dim = 1)

        # param_tokens : (N, n_params, features)
        param_tokens = torch.stack(
            [ enc(x) for (x, enc) in zip(params, self.param_encoders) ],
            dim = 1
        )
        #param_tokens=param_tokens.squeeze(1)
        #param_tokens=param_tokens.squeeze(1)

        #print("param_tokens.shape",param_tokens.shape)


        #trajs:(N,2,length)
        cond_trajs=torch.unbind(cond,dim=1)
        #print("len(cond_trajs)",len(cond_trajs))
        #print("cond_trajs[0].shape",cond_trajs[0].shape)

        # traj_token : (N, 1, features)
        #traj_enc   = self.traj_encoder(cond)
        #traj_token = self.traj_enc_to_token(traj_enc)
        #traj_token = self.traj_enc_to_token(traj_enc).unsqueeze(1)

        traj_encs = torch.stack(
            [ enc(x) for (x, enc) in zip(cond_trajs, self.traj_encoders) ],
            dim = 1
        )
        #print("traj_encs.shape",traj_encs.shape)
        #print("len(traj_encs)",len(traj_encs))
        #print("traj_encs[0].shape",traj_encs[0].shape)

        traj_encs=torch.unbind(traj_encs,dim=1)
        
        #print("len(traj_encs)",len(traj_encs))
        #print("traj_encs[0].shape",traj_encs[0].shape)

        traj_encs_each1=traj_encs[0].split(1,dim=2)
        traj_encs_each2=traj_encs[1].split(1,dim=2)
        traj_encs_each3=traj_encs[2].split(1,dim=2)

        #print("len(traj_encs_each1)",len(traj_encs_each1))

        traj_encs_all=[]
        for i in range(self.traj_enc_num_each):
            traj_encs_all.append(traj_encs_each1[i])
        for i in range(self.traj_enc_num_each):
            traj_encs_all.append(traj_encs_each2[i])
        for i in range(self.traj_enc_num_each):
            traj_encs_all.append(traj_encs_each3[i])


        #print("len(traj_encs_all)",len(traj_encs_all))
        #print("traj_encs_all[0].shape",traj_encs_all[0].shape)

        #traj_encs = traj_encs.split(1, dim = 2)
        
        #traj_tokens = torch.stack(
        #    [ enc(x) for (x, enc) in zip(traj_encs, self.traj_enc_to_tokens) ],
        #    dim = 1
        #)

        traj_tokens = torch.stack(
            [ enc(x.squeeze(2)) for (x, enc) in zip(traj_encs_all, self.traj_enc_to_tokens) ],
            dim = 1
        )

        #print("traj_tokens.shape",traj_tokens.shape)
        #traj_tokens = traj_tokens.unsqueeze(1)


        #print("traj_token.shape",traj_token.shape)

        # time_token : (N, 1, features)
        time = time.unsqueeze(1).to(dtype = torch.float32)
        time_token = self.time_encoder(time)
        #time_token = self.time_encoder(time).unsqueeze(1)
        #time_token = self.time_encoder(time).squeeze(1)

        #print("time_token.shape",time_token.shape)

        # tokens  : (N, n_params + 2, features)
        #tokens = torch.cat([param_tokens, traj_token, time_token], dim = 1)
        tokens = torch.cat([param_tokens, traj_tokens, time_token], dim = 1)

        # out_tokens  : (N, n_params + 2, features)
        out_tokens = self.transformer(tokens)

        # out_param_tokens  : (N, n_params, features)
        out_param_tokens = out_tokens[:, :self.n_params, :]

        # out_params : List[(N, 1)]
        out_params = [
            dec(x).squeeze(2)
                for (x, dec) in zip(
                    out_param_tokens.split(1, dim = 1), self.param_decoders
                )
        ]

        # result : (N, n_params)
        result = torch.cat(out_params, dim = 1)
        #print("result.shape",result.shape)

        return result

