from pathlib import Path
import argparse
import json
import time
from torch import nn, optim
import torch
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from models.vgg_1d import VGG16

parser = argparse.ArgumentParser(description='ELIC PreTraining')
parser.add_argument('--data-dir', type=Path, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--gamma', default=0.7, type=float, metavar='L',
                    help='balance parameter of the loss')
parser.add_argument('--projector', default='64-64', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

NUM_LEADS = 8


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ELIC(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_features = int(args.projector.split('-')[-1])
        self.backbone_group = list()
        for i in range(NUM_LEADS):
            backbone = VGG16(ch_in=1, alpha=0.125)
            backbone.fc = nn.Identity()
            self.backbone_group.append(backbone.to(self.device))

        sizes = [64] + list(map(int, args.projector.split('-')))
        self.projector_group = list()
        for j in range(NUM_LEADS):
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector_group.append(nn.Sequential(*layers).to(self.device))
        self.bn = nn.BatchNorm1d(512, affine=False).to(self.device)

    def forward(self, y1, y2):
        h1_list = list()
        h2_list = list()
        z1_list = list()
        z2_list = list()
        for i in range(NUM_LEADS):
            h1_list.append(self.backbone_group[i](y1[:, [i], :]))
            h2_list.append(self.backbone_group[i](y2[:, [i], :]))

            z1_list.append(self.projector_group[i](h1_list[-1]))
            z2_list.append(self.projector_group[i](h2_list[-1]))

        z1 = torch.concat(z1_list, dim=-1)
        z2 = torch.concat(z2_list, dim=-1)

        z1_2d = z1.view([z1.shape[0], NUM_LEADS, z1.shape[1] // NUM_LEADS])
        z2_2d = z2.view([z2.shape[0], NUM_LEADS, z2.shape[1] // NUM_LEADS])

        inv_loss = self.calc_inv_loss(z1_2d[:, torch.randperm(NUM_LEADS)[0], :],
                                      z2_2d[:, torch.randperm(NUM_LEADS)[0], :])

        cov_loss = self.calc_cov_loss(z1, z2)
        cov_loss /= NUM_LEADS

        loss = self.args.gamma * inv_loss + (1 - self.args.gamma) * cov_loss

        return loss, inv_loss, cov_loss

    @staticmethod
    def calc_inv_loss(p1, p2):
        temperature = 0.1
        p1 = torch.nn.functional.normalize(p1, dim=1)
        p2 = torch.nn.functional.normalize(p2, dim=1)
        logits = p1 @ p2.T
        logits /= temperature
        n = p2.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        loss_ = torch.nn.functional.cross_entropy(logits, labels)
        return loss_

    def calc_cov_loss(self, x, y):
        lambd = 0.0051
        c = self.bn(x).T @ self.bn(y)
        c.div_(self.args.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss


def main(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = ELIC(args)

    param_weights = []
    param_biases = []
    for md in model.backbone_group:
        for param in md.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    for param in model.bn.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)

    for md in model.projector_group:
        for param in md.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    parameters = param_weights + param_biases
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    t = transforms.Compose([
        RandomResizeCropTimeOut(),
        ToTensor()
    ])
    dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([t, t]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
        pin_memory=True)

    start_time = time.time()
    min_loss = 999999999
    for epoch in range(0, args.epochs):
        total_loss = 0
        total_inv_loss = 0
        total_cov_loss = 0
        ep_start_time = time.time()
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(non_blocking=True)
            y2 = y2.cuda(non_blocking=True)
            optimizer.zero_grad()
            loss, inv_loss, cov_loss = model.forward(y1, y2)
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             loss=loss.item(),
                             inv_loss=inv_loss.item(),
                             cov_loss=cov_loss.item(),
                             time=int(time.time() - start_time))
                print(json.dumps(stats))

            total_inv_loss += inv_loss.item()
            total_cov_loss += cov_loss.item()
            total_loss += loss.item()

        total_loss /= len(loader)
        total_inv_loss /= len(loader)
        total_cov_loss /= len(loader)
        ep_end_time = time.time()

        print("\nEpoch end. Time: %f - Average loss %f - inv loss %f - cov loss %f.\n" % (
            ep_end_time - ep_start_time, total_loss, total_inv_loss, total_cov_loss))

        if total_loss < min_loss:
            min_loss = total_loss
            print("\n Save model at loss %f. \n" % min_loss)
            torch.save({
                'backbone_state_dict': model.backbone_group,
            }, args.checkpoint_dir / 'encoder_group_best.pth')

    torch.save({
        'backbone_state_dict': model.backbone_group,
    }, args.checkpoint_dir / 'encoder_group.pth')


if __name__ == '__main__':
    pt_args = parser.parse_args()
    main(pt_args)
