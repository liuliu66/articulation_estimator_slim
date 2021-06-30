import argparse
import os
import json
import time
from tqdm import tqdm
import open3d as o3d
import torch

from datasets import ArticulationDataset, train_pipelines
from datasets.batch_collator import BatchCollator
from models import ArticulationEstimator
from utils import WarmupMultiStepLR, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Articulation train Estimator')
    parser.add_argument('--use_rgb', default=True, help='use rgb as point feature')
    parser.add_argument('--n_max_parts', default=13, type=int, help='use rgb as point feature')
    parser.add_argument('--distributed', action='store_true', help='distributed training or test')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.use_rgb:
        in_channels = 6
    else:
        in_channels = 3
    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_gpus = 4
    args.distributed = num_gpus > 1

    output_dir = 'output'
    if not os.path.lexists(output_dir):
        os.makedirs(output_dir)

    # build estimator
    estimator = ArticulationEstimator(in_channels=in_channels,
                                      n_max_parts=args.n_max_parts)
    device = torch.device('cuda')
    estimator.to(device)
    if num_gpus > 1:
        estimator = torch.nn.DataParallel(estimator, device_ids=list(range(num_gpus)))

    # build dataloader
    instance_per_gpu = 16
    workers_per_gpu = 4
    data_root = 'data/box_data/'
    train_dataset = ArticulationDataset(ann_file=data_root + 'train.txt',
                                        img_prefix=data_root,
                                        intrinsics_path=data_root + 'camera_intrinsic.json',
                                        n_max_parts=args.n_max_parts,
                                        pipeline=train_pipelines)
    sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, instance_per_gpu, drop_last=True
    )
    batch_collator = BatchCollator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=workers_per_gpu,
        batch_sampler=batch_sampler,
        collate_fn=batch_collator,
    )

    # training config
    base_lr = 0.001
    params = []
    optimizer = torch.optim.Adam(estimator.parameters(), base_lr)
    scheduler = WarmupMultiStepLR(optimizer,
                                  [2000, 4000, 6000, 8000],
                                  gamma=0.7
                                  )

    # start train
    max_epoch = 100
    total_iterations = 10000
    frequency_print = 20
    checkpoint_period = 20
    step = 0
    loss_print = []
    print('start training...')
    estimator.train()
    for epoch in range(max_epoch):
        for i, input_data in enumerate(train_loader):
            step += 1
            for k in input_data.keys():
                if k == 'img_meta':
                    continue
                input_data[k] = input_data[k].to(device)

            optimizer.zero_grad()
            output = estimator(return_loss=True, **input_data)
            for loss_type in output.keys():
                if not '_loss' in loss_type:
                    continue
                output[loss_type] = torch.mean(output[loss_type])
            loss = sum(output[loss_type] for loss_type in output if '_loss' in loss_type)
            loss.backward()
            optimizer.step()
            scheduler.step()

            output_print = dict()
            for loss_type in output.keys():
                if not '_loss' in loss_type:
                    continue
                output_print[loss_type] = output[loss_type].detach().cpu().numpy()
            loss_print.append(output_print)

            if step % frequency_print == 0:
                print_str = 'Epoch: {:d} | Step: {:d} | Batch Loss: {:6f}'.format(epoch, step, loss.detach().cpu().numpy())
                for loss_type in output_print:
                    print_str += ' | {}: {:6f}'.format(loss_type, output_print[loss_type])
                print(print_str)

        if epoch % checkpoint_period == 0:
            checkpoint_filename = os.path.join(output_dir, 'epoch_{}.pth'.format(epoch))
            save_checkpoint(estimator, checkpoint_filename)

        if step > total_iterations:
            checkpoint_filename = os.path.join(output_dir, 'final.pth'.format(epoch))
            save_checkpoint(estimator, checkpoint_filename)
            print('training finished')
            break


if __name__ == '__main__':
    main()