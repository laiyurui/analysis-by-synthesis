import torch
from torch import nn
from typing import TypeVar, Any
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from one_lip_ae.conv import LipschitzConv2d, BCOP
from functools import partial

T_co = TypeVar('T_co', covariant=True)


def get_one_lip_model(args):
    input_spatial_shape = (28, 28)
    input_channels = 1
    # if args.ae == 'one_lip':
    print('initalizing 1-Lip AE')

    groupsort = GroupSort(group_size=2, axis=1)
    inner_conv_module = partial(BCOP, ortho_mode="bjorck", mask_half=True, projection=False, bjorck_iters=20)
    conv_module = partial(LipschitzConv2d, conv_module=inner_conv_module)
    encoder = OneLipEncoder(
        input_spatial_shape=input_spatial_shape,
        in_channels=input_channels,
        out_channels=args.n_latents_per_class,
        conv_module=conv_module,
        linear_module=BjorckLinear,
        activation=groupsort)
    decoder = OneLipDecoder(
        input_spatial_shape=None,
        in_channels=args.n_latents_per_class,
        out_channels=1,
        conv_module=conv_module,
        linear_module=BjorckLinear,
        activation=groupsort)

    model = AE(encoder, decoder, lip_up_factor=args.lip_up_factor)
    model.to(args.device)

    # LargeConvNet needs to know the input shape to calculate the Lipschitz constant
    forward_example = torch.zeros((1, input_channels) + input_spatial_shape).to(args.device)
    model.encoder(forward_example)
    print(f"Lipschitz-constant: {get_l2_lipschitz_constant(model.encoder):.2f}")

    forward_example = torch.zeros((1, args.n_latents_per_class, 1, 1)).to(args.device)
    model.decoder(forward_example)
    print(f"Lipschitz-constant: {get_l2_lipschitz_constant(model.decoder):.2f}")
    # elif args.ae == 'vanilla':
    #     print('initalizing vanilla ae')
    #     encoder = OneLipEncoder(
    #         input_spatial_shape=input_spatial_shape,
    #         in_channels=input_channels,
    #         out_channels=args.n_latents,
    #         conv_module=torch.nn.Conv2d,
    #         linear_module=torch.nn.Linear,
    #         activation=torch.nn.ReLU())
    #     decoder = OneLipDecoder(
    #         input_spatial_shape=None,
    #         in_channels=args.n_latents,
    #         out_channels=1,
    #         conv_module=torch.nn.Conv2d,
    #         linear_module=BjorckLinear,
    #         activation=torch.nn.ReLU())
    #
    #     model = AE(encoder, decoder)
    #     model.to(args.device)
    # else:
    #     raise NotImplementedError
    print()
    return model


class Decoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_latents, 32, 4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def channel_2_dim(b, up_fact=2):
    # moves channels to spatial dimensions
    # b must be tensor of shape bs, n_ch, n_x, n_y
    bs, n_ch, n_x, n_y = b.shape
    assert n_ch // up_fact == n_ch / up_fact, f"n_ch must be multiple of up_fact**2 {up_fact**2}"
    b = b.reshape(bs, n_ch // up_fact**2, up_fact, up_fact, n_x, n_y).permute(0, 1, 4, 2, 5, 3).\
        reshape(bs, n_ch // up_fact**2, n_x * up_fact, n_y * up_fact)
    return b


class OneLipDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_module,
        linear_module,
        activation,
        input_spatial_shape,
    ):
        super().__init__()
        cf = 1
        self.cache = None
        self.activation = activation
        self.n_latents = in_channels
        self.fc1 = linear_module(in_features=in_channels, out_features=64 * cf)
        self.fc2 = linear_module(in_features=64 * cf, out_features=512 * cf)

        self.conv1 = conv_module(in_channels=32 * cf, out_channels=256 * cf, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_module(in_channels=64 * cf, out_channels=128 * cf, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv_module(in_channels=32 * cf, out_channels=16 * cf, kernel_size=3, stride=1, padding=1)  # padding is broken
        self.conv4 = conv_module(in_channels=4 * cf, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input) -> T_co:
        fc1 = self.activation(self.fc1(input[:, :, 0, 0]))
        fc2 = self.activation(self.fc2(fc1))[:, :, None, None]
        fc2_up = channel_2_dim(fc2, up_fact=4)  # (bs, 512/ 16, 4, 4)

        conv1_up = channel_2_dim(self.activation(self.conv1(fc2_up)))   # (bs,.., 8, 8)
        conv2_up = channel_2_dim(self.activation(self.conv2(conv1_up)))  # (bs,.., 16, 16)
        conv3_up = self.activation(self.conv3(conv2_up))
        conv3_up = channel_2_dim(conv3_up[:, :, 1:-1, 1:-1])    # padding does not work, (bs, ..., 28, 28)
        conv4 = self.conv4(conv3_up)    # (bs, ..., 28, 28)
        out = conv4
        self.cache = [var.detach().cpu() for var in [fc1, fc2_up, fc2_up, conv1_up, conv2_up, conv3_up, conv4, out]]
        return out


class OneLipEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_module,
        linear_module,
        activation,
        input_spatial_shape,
    ):
        super().__init__()
        self.conv1 = conv_module(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = conv_module(
            in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1,
        )
        self.conv3 = conv_module(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
        )
        self.conv4 = conv_module(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1,
        )
        self.linear1 = linear_module(
            in_features=(input_spatial_shape[0] // 4)
            * (input_spatial_shape[1] // 4)
            * 64,
            out_features=512,
        )
        self.linear2 = linear_module(in_features=512, out_features=512)
        self.linear3 = linear_module(in_features=512, out_features=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.linear1(x.flatten(start_dim=1))
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x


class AE(nn.Module):
    def __init__(self, encoder, decoder, lip_up_factor=1.):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_latents = self.decoder.n_latents
        self.lip_up_factor = lip_up_factor

    def forward(self, inputs):
        z = self.encoder(inputs - 0.5)[:, :, None, None]
        rec = self.decoder(z) * self.lip_up_factor
        return rec, z, torch.ones_like(z)


class GroupSort(nn.Module):
    """GroupSort activation function

    based on https://github.com/cemanil/LNets
    """

    def __init__(self, group_size, axis=-1):
        super().__init__()
        self.group_size = group_size
        self.axis = axis

    def lipschitz_constant(self):
        return 1

    def forward(self, x):
        return group_sort(x, self.group_size, self.axis)


def group_sort(x, group_size, axis):
    assert x.size(axis) % group_size == 0
    assert group_size == 2

    a, b = x.split(x.size(axis) // 2, axis)
    a, b = torch.max(a, b), torch.min(a, b)
    return torch.cat([a, b], dim=axis)


class BjorckLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            bjorck_beta=0.5,
            bjorck_iters=20,
            bjorck_order=1,
            power_iteration_scaling=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.bjorck_beta = bjorck_beta
        self.bjorck_iters = bjorck_iters
        self.bjorck_order = bjorck_order
        self.bjorck_weight = None
        self.power_iteration_scaling = power_iteration_scaling

    def train(self, mode=True):
        # overwrites the parent class
        super().train(mode=mode)
        if not mode:
            # in eval mode, we only recompute it if it's None
            self.bjorck_weight = None

    def reset_parameters(self):
        # overwrites the parent class
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def update_bjorck_weight(self):
        self.bjorck_weight = bjorck_orthonormalize(
            self.weight,
            beta=self.bjorck_beta,
            iters=self.bjorck_iters,
            order=self.bjorck_order,
            power_iteration_scaling=self.power_iteration_scaling,
            default_scaling=not self.power_iteration_scaling,
        )

    def singular_values(self):
        if self.bjorck_weight is None:
            with torch.no_grad():
                self.update_bjorck_weight()
        _, s, _v = torch.svd(self.bjorck_weight.detach(), compute_uv=False)
        return s

    def lipschitz_constant(self):
        return self.singular_values().max().item()

    def forward(self, x):
        if self.training:
            self.update_bjorck_weight()
        elif self.bjorck_weight is None:
            # first forward pass after calling eval()
            with torch.no_grad():
                self.update_bjorck_weight()
        return F.linear(x, self.bjorck_weight, self.bias)


def get_l2_lipschitz_constant(model):
    l_constant = 1.0
    for module in model.children():
        if isinstance(module, nn.Flatten):
            continue
        l_constant *= module.lipschitz_constant()
    return l_constant



def bjorck_orthonormalize(
        w, beta=0.5, iters=20, order=1, power_iteration_scaling=False, default_scaling=False
):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if order != 1:
        raise ValueError("only first order Bjorck is supported")

    if w.shape[-2] < w.shape[-1]:
        return bjorck_orthonormalize(
            w.transpose(-1, -2),
            beta=beta,
            iters=iters,
            order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling,
        ).transpose(-1, -2)

    if power_iteration_scaling:
        with torch.no_grad():
            _, s, _ = power_iteration(w)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)

    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w



def power_iteration(A, u=None, n_iters=10):
    shape = list(A.shape)
    shape[-1] = 1
    shape = tuple(shape)
    if u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    assert tuple(u.shape) == shape

    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    return u, s, v
