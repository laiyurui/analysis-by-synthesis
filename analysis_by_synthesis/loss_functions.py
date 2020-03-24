import warnings
import numpy as np
import torch
import torch.nn.functional as F


def samplewise_loss_function(x, rec_x, z, logvar, beta, prior_log_rate=5., KL_prior='gaussian',
                             marg_ent_weight=0.1):
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
        L2squared = L2squared / input_size / len(x)
    else:
        if len(x.shape) != 4 or x.shape != rec_x.shape:
            warnings.warn('samplewise_loss_function possibly not been optimized for this')
            raise Exception

        d = rec_x - x
        d.pow_(2)
        L2squared = d.sum((-1, -2, -3)) / input_size / len(x)

    h = 0
    if KL_prior == 'gaussian':
        KLD = -0.5 * torch.sum(1 + logvar - z.pow(2) - logvar.exp(), dim=((-1, -2, -3))) / input_size / len(x)  # z = mu
    elif KL_prior == 'exponential':
        KLD = torch.sum(z - prior_log_rate + torch.exp(prior_log_rate - z) - 1, dim=(-1, -2, -3)) / input_size / len(x)  # z is log_rates
        p = torch.exp(z)
        p = p / torch.sum(p, dim=(-1, -2, -3), keepdim=True) / input_size / len(x)
        h = torch.sum(- p * torch.log(p + 1e-10), dim=(-1,-2, -3))  # marginal entropy
    else:
        raise NotImplementedError
    # note that the KLD sum is over the latents, not over the input size
    loss =  L2squared + beta * KLD
    if marg_ent_weight != 0:
        loss -= marg_ent_weight * h
    return loss


def vae_loss_function(x, rec_x, z, logvar, beta, prior_log_rate=5, KL_prior='gaussian',
                      marg_ent_weight=0.1):
    """Loss function to train a VAE summed over all elements and batch."""
    input_size = int(np.prod(x.shape[-3:]))

    L2squared = ((rec_x - x)**2).sum() / input_size / len(x)

    h = 0
    if KL_prior == 'gaussian':
        KLD = -0.5 * torch.sum(1 + logvar - z.pow(2) - logvar.exp()) / input_size / len(x) # z = mu
    elif KL_prior == 'exponential':
        KLD = torch.sum(z - prior_log_rate + torch.exp(prior_log_rate - z) - 1) / input_size / len(x)  # z is log_rates
        p = torch.exp(z)
        p = p / torch.sum(p, dim=(-1, -2, -3), keepdim=True) / input_size / len(x)

        h = torch.sum(- p * torch.log(p + 1e-10))  # marginal entropy
    else:
        raise NotImplementedError
    loss = L2squared + beta * KLD
    if marg_ent_weight != 0:
        loss -= marg_ent_weight * h
    return loss


def abs_loss_function(x, labels, logits, recs, zs, logvars, beta, KL_prior='gaussian',
                      marg_ent_weight=0.1):
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
    assert zs.size()[:2] == (C, N)
    assert logvars.size()[:2] == (C, N)

    assert labels.min().item() >= 0
    assert labels.max().item() < C

    loss = 0
    for c, rec, z, logvar in zip(range(C), recs, zs, logvars):
        # train each VAE on samples from one class
        samples = (labels == c)
        if samples.sum().item() == 0:
            # batch does not contain samples for this VAE
            continue
        loss += vae_loss_function(x[samples], rec[samples], z[samples], logvar[samples], beta, KL_prior=KL_prior,
                                  marg_ent_weight=marg_ent_weight) / N

    ce = F.cross_entropy(logits, labels)
    total_loss = loss + ce
    return total_loss, loss, ce
