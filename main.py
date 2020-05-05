#!/usr/bin/env python3
from os.path import join
import torch
from torch import optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import git
import datetime
from pytz import timezone

from analysis_by_synthesis.datasets import get_dataset, get_dataset_loaders
from analysis_by_synthesis.inference import RobustInference
from analysis_by_synthesis.loss_functions import get_loss_function
from analysis_by_synthesis.architecture import ABS, get_base_model
from analysis_by_synthesis.args import get_args
from analysis_by_synthesis.train import train
from analysis_by_synthesis.test import test
from analysis_by_synthesis.attacks import LineSearchAttack, eval_robustness
from analysis_by_synthesis.sample import sample


def main():
    assert not hasattr(torchvision.datasets.folder, 'find_classes'), 'torchvision master required'

    args = get_args()

    if args.test_only:
        args.initial_evaluation = True
        args.epochs = 0

    first_epoch = 0 if args.initial_evaluation else 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    # load the train and test set
    train_set, test_set = get_dataset(args.dataset, args.no_augmentation)
    train_loader, test_loader = get_dataset_loaders(train_set, test_set, use_cuda, args)
    samples_per_epoch = len(train_loader.sampler)

    # create the ABS model
    color = True if args.dataset in ['cifar', 'gtsrb'] else False
    base_model_f = lambda: get_base_model(args)
    args.loss_f = get_loss_function(args)
    model = ABS(n_classes=args.n_classes, beta=args.beta, color=color, base_model_f=base_model_f,
                loss_f=args.loss_f).to(device)
    model.eval()

    # load weights
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # create wrappers that perform robust inference
    kwargs = {
        'fraction_to_dismiss': args.fraction_to_dismiss,
        'lr': args.inference_lr,
        'radius': args.clip_to_sphere,
    }
    robust_inference1 = RobustInference(model, device, n_samples=80, n_iterations=0, **kwargs)
    robust_inference2 = RobustInference(model, device, n_samples=8000, n_iterations=0, **kwargs)
    robust_inference3 = RobustInference(model, device, n_samples=8000, n_iterations=50, **kwargs)

    # create wrapper to create attack
    attack = LineSearchAttack(robust_inference2)

    # create optimizer
    all_params = list(model.parameters())
    per_parameter_options = [
        {'params': all_params, 'lr': args.lr},
    ]
    optimizer = optim.Adam(per_parameter_options)

    # create writer for TensorBoard
    time_stamp = '--' + datetime.datetime.now(tz=timezone("Europe/Berlin")).strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(args.logdir + time_stamp) if args.logdir is not None else None

    # write arguments to TensorBoard
    if writer is not None:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))

        # add git commit and branch to Tensorboard
        repo = git.Repo(search_parent_directories=True)
        writer.add_text('git/commit', repo.head.object.hexsha)
        writer.add_text('git/branch', repo.active_branch.name)
    
    # main loop
    for epoch in range(first_epoch, args.epochs + 1):
        if epoch > 0:
            # train for one epoch
            train(model, args, device, train_loader, optimizer, epoch, writer=writer)

            # model changed, so make sure reconstructions are regenerated
            robust_inference1.invalidate_cache()
            robust_inference2.invalidate_cache()
            robust_inference3.invalidate_cache()

        # common params for calls to test
        param_args = (args, device, test_loader, epoch)
        param_kwargs = {'writer': writer}

        # some evaluations can happen after every epoch because they are cheap
        test(model, *param_args, **param_kwargs)
        test(robust_inference1, *param_args, **param_kwargs)
        test(robust_inference2, *param_args, **param_kwargs)

        # expensive evaluations happen from time to time and at the end
        if epoch % args.epochs_full_evaluation == 0 or epoch == args.epochs:
            test(robust_inference3, *param_args, **param_kwargs)
            eval_robustness(robust_inference3, *param_args, attack, **param_kwargs)

        sample(model, device, epoch, writer)

    # save the model
    if args.logdir is not None:
        path = join(args.logdir, 'model.pth')
        torch.save(model.state_dict(), path)
        print(f'model saved to {path}')
    if args.save is not None:
        torch.save(model.state_dict(), args.save)
        print(f'model saved to {args.save}')

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
