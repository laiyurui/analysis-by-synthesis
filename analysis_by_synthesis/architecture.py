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


class VAE(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.n_latents = n_latents
        self.encoder = Encoder(self.n_latents)
        self.decoder = Decoder(self.n_latents)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class ABS(nn.Module):
    """ABS model implementation that performs variational inference
    and can be used for training."""

    def __init__(self, n_classes, n_latents_per_class, beta, logit_scale=350.):
        super().__init__()

        self.beta = beta
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        
        vaes = [VAE(n_latents_per_class) for _ in range(n_classes)]
        self.vaes = nn.ModuleList(vaes)
        
        self.encoder_parameters = [item for vae in vaes for item in list(vae.encoder.parameters())]
        self.decoder_parameters = [item for vae in vaes for item in list(vae.decoder.parameters())]

    def forward(self, x):
        outputs = [vae(x) for vae in self.vaes]
        recs, mus, logvars = zip(*outputs)
        recs, mus, logvars = torch.stack(recs), torch.stack(mus), torch.stack(logvars)
        losses = [samplewise_loss_function(x, recs.detach(), mus.detach(), logvars.detach(), self.beta)
                  for recs, mus, logvars in outputs]
        losses = torch.stack(losses)
        assert losses.dim() == 2
        logits = -losses.transpose(0, 1)
        logits = logits * self.logit_scale
        return logits, recs, mus, logvars
