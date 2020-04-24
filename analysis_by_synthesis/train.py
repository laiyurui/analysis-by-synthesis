import torch
from torchvision.utils import make_grid

from .loss_functions import abs_loss_function
from .utils import count_correct


def train(model, args, device, train_loader, optimizer, epoch, writer=None):
    model.train()

    epoch_loss = 0
    epoch_correct = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # training
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits, recs, mus, logvars = model(data)
        loss, vae_loss, ce_loss = abs_loss_function(data, targets, logits, recs, mus, logvars, args.beta,
                                                    KL_prior=args.KL_prior, marg_ent_weight=args.marg_ent_weight)
        loss.backward()
        optimizer.step()

        # logging
        loss = loss.item()
        epoch_loss += loss * len(data)
        correct = count_correct(logits, targets)
        accuracy = 100 * correct / len(data)
        epoch_correct += correct

        if writer is not None:
            step = (epoch - 1) * len(train_loader.sampler) + batch_idx * args.batch_size

            writer.add_scalar('loss/train', loss, step)
            writer.add_scalar('accuracy/train', accuracy, step)

            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/vae-loss', vae_loss.item(), step)
            writer.add_scalar('train/ce-loss', ce_loss.item(), step)

            writer.add_scalar('model/logit-scale', model.logit_scale, step)

            # some shenanigans on the mu's
            writer.add_scalar(f'train_latents/mean', torch.mean(mus), step)
            writer.add_scalar(f'train_latents/percantage_larger0', (mus > 0).type(torch.float32).mean(), step)
            writer.add_scalar(f'train_latents/percantage_larger0p1', (mus > 0.1).type(torch.float32).mean(), step)

            if batch_idx == 0:
                # up to 8 samples
                n = min(data.size(0), args.n_classes)

                # flatten VAE and batch dim into a single dim
                shape = (-1,) + recs.size()[2:]

                grid = torch.cat([data[:n], recs[:, :n].reshape(shape)])
                grid = make_grid(grid, nrow=n)
                writer.add_image(f'reconstructions/train', grid, step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:5d}/{} ({:2.0f}%)]\tLoss: {:3.4f} ({:2.0f}%)'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.sampler),
                100 * batch_idx / len(train_loader), loss, accuracy))

    epoch_loss /= len(train_loader.sampler)
    epoch_accuracy = epoch_correct / len(train_loader.sampler)
    if writer is not None:
        step = epoch * len(train_loader.sampler)
        writer.add_scalar('loss/train-epoch', epoch_loss, step)
        writer.add_scalar('accuracy/train-epoch', epoch_accuracy, step)
    print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy * 100:.3f}')
