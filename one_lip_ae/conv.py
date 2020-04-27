# Jonas Rauber, April 2020

# mostly copied from https://github.com/ColinQiyangLi/LConvNet

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange


def block_orth(p1, p2):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    assert p1.shape == p2.shape
    n = p1.size(0)
    kernel2x2 = {}
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    kernel2x2[0, 0] = p1.mm(p2)
    kernel2x2[0, 1] = p1.mm(eye - p2)
    kernel2x2[1, 0] = (eye - p1).mm(p2)
    kernel2x2[1, 1] = (eye - p1).mm(eye - p2)

    return kernel2x2


def matrix_conv(m1, m2):
    """Matrix convolution.
    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.
    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0, 0]).size(0)
    if n != (m2[0, 0]).size(0):
        raise ValueError(
            "The entries in matrices m1 and m2 " "must have the same dimensions!"
        )
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))  # noqa: E741
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(
                (n, n), device=m1[0, 0].device, dtype=m1[0, 0].dtype
            )
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += m1[index1, index2].mm(
                            m2[i - index1, j - index2]
                        )
    return result


def dict_to_tensor(x, k1, k2):
    return torch.stack([torch.stack([x[i, j] for j in range(k2)]) for i in range(k1)])


def convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    if ksize == 1:
        return ortho.unsqueeze(-1).unsqueeze(-1)
    p = block_orth(sym_projs[0], sym_projs[1])
    for _ in range(1, ksize - 1):
        p = matrix_conv(p, block_orth(sym_projs[_ * 2], sym_projs[_ * 2 + 1]))
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = ortho.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)


def cyclic_pad_2d(x, pads, even_h=False, even_w=False):
    """
    Implemenation of cyclic padding for 2-D image input
    """
    pad_change_h = -1 if even_h else 0
    pad_change_w = -1 if even_w else 0
    pad_h, pad_w = pads
    if pad_h != 0:
        v_pad = torch.cat(
            (x[..., :, -pad_h:, :], x, x[..., :, : pad_h + pad_change_h, :]), dim=-2
        )
    elif pad_change_h != 0:
        v_pad = torch.cat((x, x[..., :, :pad_change_h, :]), dim=-2)
    else:
        v_pad = x
    if pad_w != 0:
        h_pad = torch.cat(
            (
                v_pad[..., :, :, -pad_w:],
                v_pad,
                v_pad[..., :, :, : pad_w + pad_change_w],
            ),
            dim=-1,
        )
    elif pad_change_w != 0:
        h_pad = torch.cat((v_pad, v_pad[..., :, :, :+pad_change_w]), dim=-1)
    else:
        h_pad = v_pad
    return h_pad


def conv2d_cyclic_pad(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Implemenation of cyclic padding followed by a normal convolution
    """
    kh, kw = weight.size(-2), weight.size(-1)
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return F.conv2d(x, weight, bias, stride, padding, dilation)


# The following two functions are directly taken from https://arxiv.org/pdf/1805.10408.pdf
def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def power_iteration(A, init_u=None, n_iters=10, return_uv=True):
    """
    Power iteration for matrix
    """
    shape = list(A.shape)
    # shape[-2] = shape[-1]
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return s


def bjorck_orthonormalize(
    w, beta=0.5, iters=20, order=1, power_iteration_scaling=False, default_scaling=False
):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

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
            s = power_iteration(w, return_uv=False)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)
    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w


class PixelUnshuffle2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def lipschitz_constant(self):
        return 1.0

    def forward(self, x):
        return rearrange(
            x,
            "b c (w k1) (h k2) -> b (c k1 k2) w h",
            k1=self.kernel_size[0],
            k2=self.kernel_size[1],
        )


class LipschitzConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_module=None,
    ):
        super().__init__()
        # compute the in_channels based on stride
        # the invertible downsampling is applied before the convolution
        true_in_channels = in_channels * stride * stride
        true_out_channels = out_channels
        assert kernel_size % stride == 0
        # compute the kernel size of the actual convolution based on stride
        true_kernel_size = kernel_size // stride
        self.shuffle = PixelUnshuffle2d(stride)
        self.conv = conv_module(
            true_in_channels,
            true_out_channels,
            kernel_size=true_kernel_size,
            stride=1,
            padding=true_kernel_size // 2,
            bias=True,
        )

    def forward(self, x):
        x = self.shuffle(x)
        x = self.conv(x)
        return x

    def lipschitz_constant(self):
        l_constant = 1.0
        for module in self.children():
            l_constant *= module.lipschitz_constant()
        return l_constant


class BCOP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        bias=True,
        mask_half=True,
        projection=False,
        ortho_mode="bjorck",
        bjorck_iters=20,
        power_iteration_scaling=True,
        frozen=False,
    ):
        super().__init__()
        assert stride == 1
        assert not projection and ortho_mode == "bjorck"
        assert mask_half
        self.mask_half = mask_half
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_channels = max(self.in_channels, self.out_channels)
        self.num_kernels = 2 * (kernel_size - 1) + 1
        self.ortho_mode = ortho_mode
        self.bjorck_iters = bjorck_iters
        self.power_iteration_scaling = power_iteration_scaling
        self.frozen = frozen

        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.param_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, self.max_channels),
            requires_grad=not self.frozen,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
        self.mask = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                    torch.zeros(
                        self.num_kernels - 1,
                        1,
                        self.max_channels - self.max_channels // 2,
                    ),
                ),
                dim=-1,
            ).float(),
            requires_grad=False,
        )

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels), requires_grad=not self.frozen
            )
        else:
            self.bias = None

        self.reset_parameters()
        self.weight = None

    def train(self, mode=True):
        # overwrites the parent class
        super().train(mode=mode)
        if not mode:
            # in eval mode, we only recompute it if it's None
            self.weight = None

    def singular_values(self):
        if self.weight is None:
            self.update_weight()
        svs = conv_singular_values_numpy(
            self.buffer_weight.detach().cpu().numpy(), self._input_shape
        )
        return svs

    def lipschitz_constant(self):
        return self.singular_values().max().item()

    def reset_parameters(self):
        ortho_weights = [
            torch.empty(self.max_channels, self.max_channels)
            for i in range(self.num_kernels)
        ]
        stdv = 1.0 / (self.max_channels ** 0.5)
        for index, ortho_weight in enumerate(ortho_weights):
            nn.init.orthogonal_(ortho_weight, gain=stdv)
            self.param_matrices.data[index] = ortho_weight

        std = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def update_weight(self):
        # orthognoalize all the matrices using Bjorck
        ortho = bjorck_orthonormalize(
            self.param_matrices,
            iters=self.bjorck_iters,
            power_iteration_scaling=self.power_iteration_scaling,
            default_scaling=not self.power_iteration_scaling,
        )

        # compute the symmetric projectors
        H = ortho[0, : self.in_channels, : self.out_channels]
        PQ = ortho[1:]
        PQ = PQ * self.mask
        PQ = PQ @ PQ.transpose(-1, -2)

        # compute the resulting convolution kernel using block convolutions
        self.weight = convolution_orthogonal_generator_projs(
            self.kernel_size, self.in_channels, self.out_channels, H, PQ
        )
        self.buffer_weight = self.weight

    def forward(self, x):
        # cache the input shape for self.singular_values()
        self._input_shape = x.shape[2:]

        if self.training or self.weight is None:
            self.update_weight()

        # detach the weight when we are using the cached weights from previous steps
        weight = self.weight
        if not self.training:
            weight = weight.detach()

        # apply cyclic padding to the input and perform a standard convolution
        return conv2d_cyclic_pad(x, weight, self.bias)
