import torch
from torchvision.utils import make_grid


def sample(model, device, epoch, writer):
    if writer is None:
        return
    model.eval()
    n_latents = model.base_models[0].n_latents
    with torch.no_grad():
        zs = torch.randn(12, n_latents, 1, 1).to(device)
        samples = torch.cat([base_model.decoder(zs).cpu() for base_model in model.base_models]).clamp(0, 1)
        grid = make_grid(samples, nrow=12)
        writer.add_image(f'samples', grid, epoch)
