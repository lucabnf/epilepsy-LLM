# File that defines all the modules needed to define the BENDR archiecture
import numpy as np
import torch
from torch import nn
import copy
from utils import _make_mask

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename, map_location={'cuda:0':'cuda:3'}) #added map_location='cpu' to avoid cuda:out of memory error
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, 
                 encoder_h=256, 
                 enc_width=(3, 2, 2, 2, 2, 2), # 100% size
                #  enc_width=(3, 2, 2), # 50%
                 dropout=0., 
                 projection_head=False, 
                 enc_downsample=(3, 2, 2, 2, 2, 2)): # 100%
                #  enc_downsample=(3, 2, 2)): # 50%

        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
                # nn.ReLU()
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
                # nn.ReLU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = np.ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)


class EncodingAugment(nn.Module):
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())
        # self.relative_position = nn.Sequential(conv, nn.ReLU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename, map_location={'cuda:0':'cuda:3'}) #added map_location='cpu' to avoid cuda:out of memory error
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        # print("Initialized mask embedding and position encoder from ", filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):

    # def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu', # 100% 
    def __init__(self, in_features, hidden_feedforward=3076, heads=4, layers=4, dropout=0.15, activation='gelu', # 50%
    # def __init__(self, in_features, hidden_feedforward=3076, heads=2, layers=2, dropout=0.15, activation='gelu', # 25%
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super(BENDRContextualizer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())
            # self.relative_position = nn.Sequential(conv, nn.ReLU())
            # self.relative_position = nn.Sequential(conv, nn.Sigmoid())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
            # Permute([0, 2, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        # Multi-gpu workaround, wastes memory
        x = x.clone()

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        # print("# nan: ,", torch.isnan(x).sum())
        # print('# zeros: ', torch.numel(x)-torch.count_nonzero(x))
        # print('range: [{}, {}]'.format(torch.min(x), torch.max(x)))
        if self.position_encoder:
            x = x + self.relative_position(x) ## here's the problem!!!!!
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename, map_location={'cuda:0':'cuda:3'}) #added map_location='cpu' to avoid cuda:out of memory error
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)