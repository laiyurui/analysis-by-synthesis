import tqdm
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from .loss_functions import abs_loss_function
from .utils.util_functions import get_best_non_target_logit
from .utils import count_correct
import numpy as np


def test(model, args, device, test_loader, epoch, writer=None, max_batches=None):
    model.eval()
    suffix = '-' + model.name if hasattr(model, 'name') else ''

    N = len(test_loader.dataset)

    loss = 0
    correct = 0
    all_espilons = []
    with torch.no_grad():
        # using a context manager for tqdm prevents messed up console outputs
        with tqdm.tqdm(test_loader) as t:
            for i, (data, targets) in enumerate(t):
                data = data.to(device)
                targets = targets.to(device)
                logits, recs, mus, logvars = model(data)
                loss += abs_loss_function(data, targets, logits, recs,
                                          mus, logvars, args.beta,
                                          args.loss_f)[0].item() * len(data)
                all_espilons.append(get_epsilon_bound(logits, targets, args))
                correct += count_correct(logits, targets)

                if i == 0 and writer is not None:
                    # up to 8 samples
                    n = min(data.size(0), 8)

                    # flatten VAE and batch dim into a single dim
                    shape = (-1,) + recs.size()[2:]

                    grid = torch.cat([data[:n], recs[:, :n].reshape(shape).clamp(0, 1)])
                    grid = make_grid(grid, nrow=n)
                    writer.add_image(f'reconstructions/test{suffix}', grid, epoch)

                if i == max_batches:
                    # limit testing to a subset by passing max_batches
                    N = i * args.test_batch_size + len(data)
                    break

    loss /= N
    accuracy = 100 * correct / N
    all_espilons = torch.cat(all_espilons).cpu()
    median_eps = torch.median(all_espilons)
    print(f'====> Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{N} ({accuracy:.0f}%) {suffix[1:]}, '
          f'Median Eps {median_eps:.4f}, \n')

    if writer is not None:
        writer.add_scalar(f'loss/test{suffix}', loss, epoch)
        writer.add_scalar(f'accuracy/test{suffix}', accuracy, epoch)

        writer.add_scalar(f'robustness/median_eps{suffix}', median_eps, epoch)
        if len(suffix) == 0:
            writer.add_scalar(f'robustness/eps_0p1_eps{suffix}', torch.mean((all_espilons >= 0.1).float()), epoch)
            writer.add_scalar(f'robustness/eps_0p3_eps{suffix}', torch.mean((all_espilons >= 0.3).float()), epoch)
            fig, ax = plt.subplots(1, 1)
            ax.hist(all_espilons, bins=np.linspace(0, 3, 40))
            writer.add_figure(f'robustness/epsilon_hist{suffix}', fig, epoch)


def get_epsilon_bound(logits, targets, args):
    true_logits = logits[range(len(targets)), targets]
    best_other_logits, _ = get_best_non_target_logit(logits, targets)
    margin = true_logits - best_other_logits
    sqrt_2 = torch.tensor(2.).to(logits.device).sqrt()

    eps_bounds = torch.max(torch.zeros_like(margin), margin) / (sqrt_2 * (1 + args.lip_up_factor + args.beta))
    return eps_bounds