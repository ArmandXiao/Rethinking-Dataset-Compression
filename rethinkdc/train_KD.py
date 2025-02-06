import os
import sys
import argparse
import yaml


def get_args():
    from rich.table import Table
    from rich.console import Console
    from rich.box import HEAVY
    import pkg_resources

    def create_custom_help_with_groups(parser):
        console = Console()
        
        # Define consistent column widths
        ARGUMENT_WIDTH = 20
        HELP_WIDTH = 60
        TYPE_WIDTH = 8
        DEFAULT_WIDTH = 25

        if parser.description:
            console.print(f"\n[bold blue]{parser.description}[/bold blue]")

        # Print usage
        usage = parser.format_usage()
        if usage:
            console.print(usage)

        console.print("[bold]Example Usage:[/bold]\n \trethinkdc he-yang/2025-rethinkdc-imagenet-random-ipc-10 --soft --ipc 10 --output-dir ./random_ipc10_soft \n")

        def create_group_table(title, actions, show_header=False):  # Added show_header parameter
            table = Table(
                title=title,
                box=HEAVY,
                show_header=show_header,  # Now controlled by parameter
                header_style="bold magenta",
                title_style="bold",
                show_edge=False,
                pad_edge=False,
                width=ARGUMENT_WIDTH + HELP_WIDTH + TYPE_WIDTH + DEFAULT_WIDTH + 3
            )
            
            # Add columns with fixed widths
            table.add_column("Argument", style="cyan", width=ARGUMENT_WIDTH)
            table.add_column("Help", style="green", width=HELP_WIDTH)
            table.add_column("Type", style="yellow", width=TYPE_WIDTH)
            table.add_column("Default", style="magenta", width=DEFAULT_WIDTH)

            for action in actions:
                arg_names = ', '.join(action.option_strings) if action.option_strings else action.dest
                help_text = action.help or ''
                arg_type = str(action.type.__name__) if action.type else 'str'
                default = str(action.default) if action.default is not None else ''
                    
                table.add_row(arg_names, help_text, arg_type, default)

            return table

        # Main Arguments - First table with headers
        main_actions = [action for action in parser._actions
                        if action.container is parser and not any(
                action in g._group_actions for g in parser._action_groups)]
        if main_actions:
            console.print(create_group_table("Main Arguments", main_actions, show_header=True))

        # Argument Groups - Subsequent tables without headers
        for group in parser._action_groups:
            if group.title != 'positional arguments' and group.title != 'optional arguments':
                if hasattr(group, '_mutually_exclusive'):
                    title = f"{group.title} (Mutually Exclusive)"
                else:
                    title = group.title

                group_actions = group._group_actions
                if group_actions:
                    console.print(create_group_table(title, group_actions, show_header=False))

        if parser.epilog:
            console.print(f"\n[italic]{parser.epilog}[/italic]")

        console.print("\n")


    parser = argparse.ArgumentParser(description='Rethinking Large-scale Dataset Compression', 
                                     epilog="For more information, please visit the project repository: https://github.com/ArmandXiao/Rethinking-Dataset-Compression")
    
    parser.print_help = lambda: create_custom_help_with_groups(parser)

    
    config_group = parser.add_argument_group('Configuration Options')
    mutually_exclusive_config = config_group.add_mutually_exclusive_group()

    mutually_exclusive_config.add_argument(
        '--soft',
        action='store_true',
        help='Use standard_soft_config.yaml (Example: rethinkdc PATH --soft)'
    )
    mutually_exclusive_config.add_argument(
        '--hard',
        action='store_true',
        help='Use standard_hard_config.yaml (Example: rethinkdc PATH --hard)'
    )
    mutually_exclusive_config.add_argument(
        '--yaml',
        type=str,
        default=None,
        help='Custom config file (Exmpale: rethinkdc YOUR_PATH_TO_CONFIG.yaml)'
    )

    # --- Data Group ---
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('train_dir', type=str,
                        nargs='?',
                        help='path to training dataset or huggingface dataset key')
    data_group.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    data_group.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    data_group.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')
    data_group.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    data_group.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')
    data_group.add_argument('--hf-cache-dir', type=str, default='./hf_cache', help='cache dir for huggingface dataset')
    data_group.add_argument('--mode', type=str, default='fkd_save', help='mode for training')

    # --- Training Options Group ---
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    training_group.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    training_group.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    training_group.add_argument('--sgd-setting', default=False, action='store_true', help='using sgd evaluation settting (lr=0.1, scheduler=cos)')
    training_group.add_argument('--hard-label', default=False, action='store_true', help='use hard label')
    training_group.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    training_group.add_argument('--epochs', type=int, default=300, help='total epoch')


    # --- Model Options Group ---
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str,
                        default='resnet18', help='student model name')
    model_group.add_argument('--teacher-model', type=str, default=None,
                        help='teacher model name')
    model_group.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')

    # --- Mixup/CutMix Options Group ---
    mix_group = parser.add_argument_group('Mixup/CutMix Options')

    mix_group.add_argument('--mix-type', default=None, type=str,
                         help='choices in {mixup, cutmix, None}')
    mix_group.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    mix_group.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    mix_group.add_argument('--ipc', default=50, type=int, help='number of images per class')


    # --- Wandb Options Group ---
    wandb_group = parser.add_argument_group('Wandb Options')  # Corrected group name
    wandb_group.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    wandb_group.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')

    # If no arguments are provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args_cli = parser.parse_args()
    
    final_args = vars(args_cli).copy()

    yaml_file = None
    if args_cli.soft:
        yaml_file = pkg_resources.resource_filename('rethinkdc', 'dc_config/standard_soft_config.yaml')
    elif args_cli.hard:
        yaml_file = pkg_resources.resource_filename('rethinkdc', 'dc_config/standard_hard_config.yaml')
    else:
        yaml_file = args_cli.yaml
    print(f"Loading configuration from {yaml_file}")

    with open(yaml_file, 'r') as file:
        yaml_args = yaml.safe_load(file)
        
        # First update with YAML values
        for key, value in yaml_args.items():
            if key in final_args:
                final_args[key] = value
            else:
                print(f"Warning: Unknown parameter in YAML: {key}")


    # override with command line arguments that were explicitly set
    for key, value in vars(args_cli).items():
        # check if this argument was explicitly set on the command line
        if key != 'yaml' and sys.argv[1:] and any(f"--{key.replace('_', '-')}" in arg for arg in sys.argv[1:]):
            final_args[key] = value

    final_args['val_dir'] = os.getenv('IMAGENET_VAL_DIR', '/default/path/if/not/set')

    if 'mode' not in final_args:
        final_args['mode'] = 'fkd_save'

    return argparse.Namespace(**final_args)

def main():
    
    args = get_args()

    from rethinkdc.imagenet_ipc import ImageFolderIPC
    from rethinkdc.utils import AverageMeter, accuracy, get_parameters, print_args_rich
    from rethinkdc.utils_fkd import mix_aug
    
    import math
    import time
    import shutil
    import numpy as np
    import wandb

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.optim.lr_scheduler import LambdaLR

    from collections import defaultdict

    def train(model, args, epoch=None):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        optimizer = args.optimizer
        scheduler = args.scheduler
        loss_function_kl = nn.KLDivLoss(reduction='batchmean')
        loss_function = nn.CrossEntropyLoss()

        model.train()
        if not args.hard_label:
            args.teacher_model.eval()
        t1 = time.time()
        max_prob_list = []
        for batch_idx, (data, target) in enumerate(args.train_loader):
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            images, _, _, _ = mix_aug(data, args)

            output = model(images)
            
            if not args.hard_label:
                soft_label = args.teacher_model(images).detach()
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                output = F.log_softmax(output/args.temperature, dim=1)
                soft_label = F.softmax(soft_label/args.temperature, dim=1)

                max_prob, _ = torch.max(soft_label, dim=1)
                max_prob_list.append(max_prob.mean().item())

                loss = loss_function_kl(output, soft_label)
                # loss = loss * args.temperature * args.temperature
            else:
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                loss = loss_function(output, target)

            n = images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch_idx == 0:
                optimizer.zero_grad()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(args.train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        metrics = {
            "train/loss": objs.avg,
            "train/Top1": top1.avg,
            "train/Top5": top5.avg,
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/max_prob_crop": np.mean(max_prob_list)}
        wandb_metrics.update(metrics)


        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                    'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                    'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                    'train_time = {:.6f}'.format((time.time() - t1))
        print(printInfo)
        t1 = time.time()
        return np.mean(max_prob_list)

    def validate(model, args, epoch=None):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss_function = nn.CrossEntropyLoss()

        model.eval()
        t1 = time.time()

        # Initialize per-class accuracy tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for data, target in args.val_loader:
                target = target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = loss_function(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                n = data.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                # Calculate per-class accuracy
                _, predicted = torch.max(output, 1)
                correct = (predicted == target).squeeze()
                for i in range(n):
                    label = target[i].item()
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        # Calculate per-class accuracy
        per_class_accuracy = {label: class_correct[label] / class_total[label] 
                            for label in class_total}

        logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'val_time = {:.6f}'.format(time.time() - t1)
        print(logInfo)

        metrics = {
            'val/loss': objs.avg,
            'val/top1': top1.avg,
            'val/top5': top5.avg,
            'val/epoch': epoch,
        }

        wandb.log(metrics)

        return top1.avg, per_class_accuracy


    def save_checkpoint(state, is_best, output_dir=None,epoch=None):
        print('==> Do not save checkpoint to save space')
        return
        if epoch is None:
            path = output_dir + '/' + 'checkpoint.pth.tar'
        else:
            path = output_dir + f'/checkpoint_{epoch}.pth.tar'
        torch.save(state, path)

        if is_best:
            path_best = output_dir + '/' + 'model_best.pth.tar'
            shutil.copyfile(path, path_best)



    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1],
               config={"tracking": False},
               settings=wandb.Settings(_disable_stats=True))

    print_args_rich(args)

    print('=> args.output_dir', args.output_dir)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")


    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])


    if "he-y" in args.train_dir:  # use huggingface dataset
        from datasets import load_dataset
        from rethinkdc.imagenet_ipc import HFDatasetAdapter
        dataset_hf = load_dataset(args.train_dir, cache_dir=args.hf_cache_dir)
        train_dataset = HFDatasetAdapter(dataset_hf, transform=train_transforms)
        print(f"=> Load data from Huggingface: total images = {len(train_dataset)}, choose images = {args.ipc}")
    else:   # use local dataset
        assert os.path.exists(args.train_dir)
        train_dataset = ImageFolderIPC(
            args.train_dir,
            transform=train_transforms,
            image_number=args.ipc
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size/4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')


    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = torchvision.models.__dict__[args.model](pretrained=False)
    model = nn.DataParallel(model).cuda()
    model.train()

    if not args.hard_label:
        # load teacher model
        print("=> loading teacher model '{}'".format(args.teacher_model))
        teacher_model = torchvision.models.__dict__[args.teacher_model](pretrained=True)
        teacher_model = nn.DataParallel(teacher_model).cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        args.teacher_model = teacher_model

    if args.sgd_setting:
        # pre-defined as pruning settings
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        assert args.cos, "CosineAnnealing Scheduler is used in SGD setting"
    else:  # default to use AdamW
        optimizer = torch.optim.AdamW(get_parameters(model),
                                        lr=args.adamw_lr,
                                        weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)


    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    

    max_prob_crop_list = []
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        max_prob_crop = train(model, args, epoch)
        max_prob_crop_list.append(max_prob_crop)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            top1, _ = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)
    wandb.log({'max_prob_crop.avg': np.mean(max_prob_crop_list)})


# if __name__ == "__main__":
#     # Only parse arguments first
#     args = get_args()
    
#     # Only proceed with heavy imports and initialization if not showing help
#     if not any(arg in sys.argv for arg in ['-h', '--help']):
#         import multiprocessing as mp
#         mp.set_start_method('spawn')
#         main(args)
