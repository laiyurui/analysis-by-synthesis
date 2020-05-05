import warnings
import numpy as np
import torch
import torch.nn.functional as F


def get_loss_function(args):
    if args.base_model == 'one_lip_ae':
        return 'reg_ae'
    elif args.base_model == 'vae':
        return 'vae'
    else:
        raise NotImplementedError(f'base_model {args.base_model} needs be assiciated with loss function')


def samplewise_loss_function(x, rec_x, mu, logvar, beta, loss_f='vae'):
    """This is the loss function used during inference to calculate the logits.

    This function must only operate on the last the dimensions of x and rec_x.
    There can be varying number of additional dimensions before them!
    """
    input_size = int(np.prod(x.shape[-3:]))
    if len(x.shape) == 5 and len(rec_x.shape) == 5 and x.shape[1] == 1 and rec_x.shape[0] == 1:
        # alternative implementation that is much faster and more memory efficient
        # when each sample in x needs to be compared to each sample in rec_x
        assert x.shape[-3:] == rec_x.shape[-3:]
        x = x.reshape(x.shape[0], input_size)
        y = rec_x.reshape(rec_x.shape[1], input_size)

        x2 = torch.norm(x, p=2, dim=-1, keepdim=True).pow(2)  # x2 shape (bs, 1)
        y2 = torch.norm(y, p=2, dim=-1, keepdim=True).pow(2)  # y2 shape (1, nsamples)
        # note that we could cache the calculation of y2, but
        # it's so fast that it doesn't matter

        L2squared = x2 + y2.t() - 2 * torch.mm(x, y.t())
    else:
        if len(x.shape) != 4 or x.shape != rec_x.shape:
            warnings.warn('samplewise_loss_function possibly not been optimized for this')
            raise Exception

        d = rec_x - x
        d.pow_(2)
        L2squared = d.sum((-1, -2, -3))

    if loss_f == 'vae':
        latent_reg = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=((-1, -2, -3)))  # KLD
        rec_loss = L2squared
    elif loss_f == 'reg_ae':
        latent_reg = (mu**2).sum((-1, -2, -3)).sqrt()
        rec_loss = L2squared.sqrt()
    else:
        raise NotImplementedError

    # note that the KLD sum is over the latents, not over the input size
    return rec_loss + beta * latent_reg


def vae_loss_function(x, rec_x, mu, logvar, beta):
    """Loss function to train a VAE summed over all elements and batch."""
    diff = rec_x - x
    L2squared = torch.sum(diff.pow(2))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L2squared + beta * KLD


def reg_ae_loss_function(x, rec_x, mu, logvar, beta):
    """Loss function to train a LAE summed over all elements and batch."""
    L2 = ((rec_x - x)**2).flatten(2).sum((-1, -2, -3)).sqrt().sum()
    latent_reg = (mu**2).sum((-1, -2, -3)).sqrt().sum()
    return L2 + beta * latent_reg


def abs_loss_function(x, labels, logits, recs, mus, logvars, beta, loss_f='vae'):
    """Loss function of the full ABS model

    Args:
        x (Tensor): batch of inputs
        labels (Tensor): batch of labels corresponding to the inputs
        logits (Tensor): batch of logits (logit for each sample (dim 0) and class (dim 1))
        recs (Tensor): reconstruction from each VAE (dim 0) for each sample (dim 1)
        mus (Tensor): mu from each VAE (dim 0) for each sample (dim 1)
        logvars (Tensor): logvar from each VAE (dim 0) for each sample (dim 1)
    """
    N = len(x)
    C = len(recs)

    assert labels.size() == (N,)
    assert recs.size()[:2] == (C, N)
    assert mus.size()[:2] == (C, N)
    assert logvars.size()[:2] == (C, N)

    assert labels.min().item() >= 0
    assert labels.max().item() < C

    loss = 0
    if loss_f == 'vae':
        loss_f = vae_loss_function
    elif loss_f == 'reg_ae':
        loss_f = reg_ae_loss_function
    else:
        raise NotImplementedError

    for c, rec, mu, logvar in zip(range(C), recs, mus, logvars):
        # train each VAE on samples from one class
        samples = (labels == c)
        if samples.sum().item() == 0:
            # batch does not contain samples for this VAE
            continue

        loss += loss_f(x[samples], rec[samples], mu[samples], logvar[samples], beta) / float(len(x[samples]))

    ce = F.cross_entropy(logits, labels)
    total_loss = loss + ce
    return total_loss, loss, ce
