# Copyright (c) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2021 mfinzi
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Dataset code adapted from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/datasets.py -- MIT License

import argparse
import time

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# Dataset definition
class O5Synthetic(object):
    """
    Simply replaces the sin function with cos from the original O5synthetic task and sets d=3
    """

    def __init__(self, N=1024):
        super().__init__()
        d = 5
        self.dim = 2 * d
        self.X = torch.randn(size=(N, 2, d))
        x1, x2 = self.X[:, 0, :], self.X[:, 1, :]
        self.Y = (
            torch.sin(torch.norm(x1, dim=-1))
            + 0.5 * torch.norm(x2, dim=-1) ** 3
            + torch.einsum("ij, ij -> i", x1, x2) / (torch.norm(x1, dim=-1) * torch.norm(x2, dim=-1))
        )

        self.X = self.X.reshape(N, -1)
        self.Y = self.Y[..., None]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


# G-RepsNet model definition
class GRepsNet_O5(tnn.Module):
    def __init__(self, c_in=2, c_h=100, c_out=1):
        super().__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.fc1 = tnn.Linear(c_in, c_h, bias=False)
        self.fc2 = tnn.Linear(c_h, c_h, bias=False)
        self.fc3 = tnn.Linear(c_h, c_h, bias=False)
        self.fc4 = tnn.Linear(c_h, c_h, bias=False)
        self.fc5 = tnn.Linear(c_h, c_out, bias=False)
        self.relu = tnn.ReLU()

    def o5_nonlin(self, x):
        EPS = 1e-5
        non_lin = self.relu(torch.norm(x, dim=1, keepdim=True) - 0.3) + EPS
        x = non_lin * x
        return x

    def forward(self, x):
        # dim x [batch_size, 2, 5]  # V + V
        x = x.reshape(-1, 2, 5).permute(0, 2, 1)  # dim x [batch_size, 5, 2]  # V + V

        x = self.fc1(x)  # dim [bs, 5, 100]
        x_res0 = x
        x = self.o5_nonlin(x)

        x = self.fc2(x) + x_res0  # dim [bs, 5, 100]
        x_res1 = x
        x = self.o5_nonlin(x)
        x = self.fc3(x) + x_res1
        x = torch.norm(x, dim=1)  # dim [bs, 1, 100]
        x_res2 = x
        x = self.fc4(x) + x_res2  # dim [bs, 1]
        x = self.relu(x)
        x = self.fc5(x)
        return x


def main(args):
    print(f"model: {args.model}")
    trainset = O5Synthetic(args.traindata_size)
    testset = O5Synthetic(args.testdata_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    BS = args.batch_size
    lr = args.lr
    NUM_EPOCHS = args.num_epochs

    if args.model == "grep":
        model = GRepsNet_O5(c_in=2, c_h=args.channels, c_out=1).to(device)
    else:
        raise NotImplementedError

    def num_params(model):
        return sum([p.numel() for n, p in model.named_parameters()])

    num_parameters = num_params(model)
    print(f"num params: {num_parameters}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def loss(x, y):
        x = x.float()
        yhat = model(x.to(device))
        return ((yhat - y.to(device)) ** 2).mean()

    def train_op(x, y):
        optimizer.zero_grad()
        lossval = loss(x, y)
        lossval.backward()
        optimizer.step()
        return lossval

    def eval(model, loader):
        test_loss = 0.0
        for i, data in enumerate(loader):
            x, y = data
            x, y = x.to(device).float(), y.to(device).float()
            out = model(x)
            loss = ((out - y.to(device)) ** 2).mean().item()
            test_loss += loss
        test_loss /= i + 1
        return test_loss

    trainloader = DataLoader(trainset, batch_size=BS, shuffle=True)
    testloader = DataLoader(testset, batch_size=BS, shuffle=True)

    test_losses = []
    train_losses = []

    start_time = time.time()
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_epoch_loss = 0.0

        for i, data in enumerate(trainloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            train_epoch_loss += train_op(x, y).item()

        train_epoch_loss /= i + 1
        train_losses.append(train_epoch_loss)

        if (epoch + 1) % 5 == 0:
            test_losses.append(eval(model, testloader))

    end_time = time.time()
    train_time_per_epoch = (end_time - start_time) / NUM_EPOCHS
    print(f"Elapsed training time per epoch = {train_time_per_epoch} seconds")

    start_eval_time = time.time()
    final_test_loss = eval(model, testloader)
    end_eval_time = time.time()
    print(f"Eval time per sample = {(end_eval_time - start_eval_time)/len(testloader)} seconds")
    print(f"Final test loss {args.model}: {final_test_loss}")
    return final_test_loss, train_time_per_epoch, num_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for O(5) synthetic dataset")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--traindata_size", default=1000, type=int)
    parser.add_argument("--testdata_size", default=1000, type=int)
    parser.add_argument("--model", default="grep", choices=["grep"])
    parser.add_argument("--channels", default=100, type=int)

    args = parser.parse_args()

    if args.model != "grep":
        raise NotImplementedError
    main(args)
