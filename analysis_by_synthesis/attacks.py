import torch
import numpy as np
import tqdm
from torchvision.utils import make_grid


def eval_robustness(model, args, device, test_loader, step, attack, writer=None, max_batches=None):
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as t:
            for i, (data, targets) in enumerate(t):
                data = data.to(device)
                targets = targets.to(device)
                print(attack)
                advs = attack(data, targets, n_ft_steps=2)

                dists = [adv_i['distance'] for adv_i in advs]
                median_pert = np.median(dists)

                # plot adversarials
                if i == 0 and writer is not None:
                    n = min(data.shape[0], 8)
                    imgs = [adv_i['img'] for adv_i in advs][:n**2]
                    # flatten VAE and batch dim into a single dim
                    grid = torch.stack(imgs)
                    grid = make_grid(grid, nrow=n)
                    writer.add_image(f'adversarials/adv_{i}', grid, step)
                    writer.add_scalar(f'robustness/{attack.name}', median_pert, step)
                break
    return


class LineSearchAttack:
    def __init__(self, abs_model, device='gpu'):
        self.abs = abs_model
        self.name = 'LatentDescent'

    def __call__(self, x, l, n_coarse_steps=10, n_ft_steps=10):
        device = x.device
        bs = x.shape[0]
        best_other = 0
        best_advs = [{'original_label': -1, 'adversarial_label': None,
                      'distance': np.inf, 'img': torch.zeros(x.shape[1:]).to(device)}
                     for _ in range(bs)]
        coarse_steps = torch.zeros(bs).to(device)

        n_adv_found = 0
        for i, coarse_step in enumerate(torch.linspace(0, 1., n_coarse_steps).to(device)):
            current_adv = (1 - coarse_step) * x + coarse_step * best_other
            best_other, current_label = self.get_best_prototypes(current_adv, l)
            for j, (current_adv_i, pred_l_i, l_i) in enumerate(zip(current_adv, current_label, l)):
                if best_advs[j]['original_label'] == -1 and pred_l_i != l_i:
                    self.update_adv(best_advs[j], current_adv_i, pred_l_i, l_i, x[j])
                    coarse_steps[i] = coarse_step
                    n_adv_found += 1
            if n_adv_found == bs:
                break
        best_advs_imgs = torch.cat([a['img'][None] for a in best_advs])
        coarse_steps_old = coarse_steps[:, None, None, None]

        # binary search
        best_advs_imgs_old = best_advs_imgs.clone()
        sign, step = - torch.ones(bs, 1, 1, 1).to(device), 0.5
        for i in range(n_ft_steps):
            coarse_steps = coarse_steps_old + step * sign
            current_adv = (1 - coarse_steps) * x + coarse_steps * best_advs_imgs_old
            _, current_label = self.get_best_prototypes(current_adv, l)

            for j, (pred_l_i, l_i) in enumerate(zip(current_label, l)):
                if pred_l_i == l_i:
                    sign[j] = 1
                else:
                    self.update_adv(best_advs[j], current_adv[j], pred_l_i, l_i, x[j])

                    sign[j] = -1
            step /= 2

        return best_advs

    def get_best_prototypes(self, x: torch.Tensor, l: torch.Tensor):
        bs = l.shape[0]
        logits, recs, l_v_classes, logvars = self.abs.forward(x)
        _, pred_classes = torch.max(logits, dim=1)
        logits[range(bs), l] = - np.inf
        _, pred_classes_other = torch.max(logits, dim=1)
        best_other_reconst = recs[pred_classes_other.squeeze(), range(bs)]
        best_other_reconst = self.post_process_reconst(best_other_reconst, x)

        return best_other_reconst, pred_classes.squeeze()

    def update_adv(self, best_adv, current_adv, pred_l, orig_l, orig_x):
        best_adv['img'] = current_adv.data.clone()
        best_adv['original_label'] = orig_l.cpu()
        best_adv['adversarial_label'] = pred_l.cpu()
        best_adv['distance'] = torch.sqrt(torch.sum((current_adv - orig_x).cpu()**2))

    def post_process_reconst(self, reconst, x):
        return reconst






