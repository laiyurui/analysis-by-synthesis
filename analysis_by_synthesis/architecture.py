import torch
from torch import nn

from .loss_functions import samplewise_loss_function


class Encoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.conv_mu = nn.Conv2d(64, n_latents, 5)
        self.conv_logvar = nn.Conv2d(64, n_latents, 5)

    def forward(self, x):
        shared = self.shared(x)
        mu = self.conv_mu(shared)
        logvar = self.conv_logvar(shared)
        return mu, logvar


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
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        return self.sigmoid(x)


class ColorEncoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.conv_mu = nn.Conv2d(64, n_latents, 5)
        self.conv_logvar = nn.Conv2d(64, n_latents, 5)

    def forward(self, x):
        shared = self.shared(x)
        mu = self.conv_mu(shared)
        logvar = self.conv_logvar(shared)
        return mu, logvar


class ColorDecoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_latents, 32, 4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 3, 4),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        return self.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, n_latents, color=False, KL_prior='gaussian', threshold=None):
        super().__init__()
        self.KL_prior = KL_prior
        self.threshold = threshold

        self.n_latents = n_latents
        if color:
            self.encoder = ColorEncoder(self.n_latents)
            self.decoder = ColorDecoder(self.n_latents)
        else:
            self.encoder = Encoder(self.n_latents)
            self.decoder = Decoder(self.n_latents)

    def reparameterize(self, z, logvar):
        if self.KL_prior == 'gaussian':
            if self.training:  # z is mu in this case
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return eps.mul(std).add_(z)  # mu
            else:
                return z   # mu
        elif self.KL_prior == 'exponential':   # z is log_rates of exponential
            if self.training:
                # inverse transform sampling - Exponential Distribution
                eps = torch.empty_like(z).uniform_(0, 1)
                return - 1 / torch.exp(z) * torch.log(1 - eps)
            else:
                return 1 / torch.exp(z)
        else:
            raise Exception(f'prior {self.KL_prior} not know')

    def forward(self, x):
        z, logvar = self.encoder(x)
        z_reparam = self.reparameterize(z, logvar)
        if self.KL_prior == 'exponential':   # z is log_rates of exponential
            z_reparam = torch.relu(z_reparam - self.threshold)
            logvar = z_reparam
        return self.decoder(z_reparam), z, logvar


class ABS(nn.Module):
    """ABS model implementation that performs variational inference
    and can be used for training."""

    def __init__(self, n_classes, n_latents_per_class, beta, color=False, logit_scale=350., KL_prior='gaussian',
                 marg_ent_weight=0.1, threshold=None):
        super().__init__()

        self.beta = beta
        self.marg_ent_weight = marg_ent_weight
        self.vaes = nn.ModuleList([VAE(n_latents_per_class, color, KL_prior=KL_prior,
                                       threshold=threshold) for _ in range(n_classes)])
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.KL_prior = KL_prior

        self.encoder_parameters = [item for vae in self.vaes for item in list(vae.encoder.parameters())]
        self.decoder_parameters = [item for vae in self.vaes for item in list(vae.decoder.parameters())]

    def forward(self, x):
        outputs = [vae(x) for vae in self.vaes]
        recs, mus, logvars = zip(*outputs)
        recs, mus, logvars = torch.stack(recs), torch.stack(mus), torch.stack(logvars)
        losses = [samplewise_loss_function(x, recs.detach(), mus.detach(), logvars.detach(), self.beta,
                                           KL_prior=self.KL_prior, marg_ent_weight=self.marg_ent_weight)
                  for recs, mus, logvars in outputs]
        losses = torch.stack(losses)
        assert losses.dim() == 2
        logits = -losses.transpose(0, 1)
        logits = logits * self.logit_scale
        return logits, recs, mus, logvars
