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
class ParticleInteraction(object):
    """Electron muon e^4 interaction"""

    def __init__(self, N=1024):
        super().__init__()
        self.dim = 4 * 4
        self.X = np.random.randn(N, self.dim) / 4
        P = self.X.reshape(N, 4, 4)
        p1, p2, p3, p4 = P.transpose(1, 0, 2)
        𝜂 = np.diag(np.array([1.0, -1.0, -1.0, -1.0]))
        dot = lambda v1, v2: ((v1 @ 𝜂) * v2).sum(-1)  # Lorentzian dot product
        Le = p1[:, :, None] * p3[:, None, :] - (dot(p1, p3) - dot(p1, p1))[:, None, None] * 𝜂  # dim [N, 4, 4]
        L𝜇 = (p2 @ 𝜂)[:, :, None] * (p4 @ 𝜂)[:, None, :] - (dot(p2, p4) - dot(p2, p2))[
            :, None, None
        ] * 𝜂  # dim [N, 4, 4]
        M = 4 * (Le * L𝜇).sum(-1).sum(-1)
        self.Y = M
        self.Y = self.Y[..., None]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


# G-RepsNet model definition
class GRepsNet_SO13(tnn.Module):
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

    def lorentz_dot(self, x, y):
        # dim x [batch_size, 4, channels]
        x_shape = x.shape
        g_uv = torch.tensor([1.0, -1.0, -1.0, -1.0], device=x.device)  # dim [4]
        xy_dot = torch.einsum("ijk, j, ijk -> ik", x, g_uv, y).reshape(-1, 1, x_shape[-1])  # dim [batch_size, 1, 100]
        return xy_dot

    def so13_nonlin(self, x):
        EPS = 1e-4
        x_norm = torch.sqrt(torch.abs(self.lorentz_dot(x, x))) + EPS
        x = x_norm * x
        return x

    def forward(self, x):
        # dim x [batch_size, 1, 4]  # V
        x = x.reshape(-1, self.c_in, 4).permute(0, 2, 1)  # dim x [batch_size, 4, self.c_in]  # V

        x = self.fc1(x)  # dim [bs, 4, 100]
        x_res0 = x
        x = self.so13_nonlin(x)

        x = self.fc2(x) + x_res0  # dim [bs, 4, 100]
        x_res1 = x
        x = self.so13_nonlin(x)
        x = self.fc3(x) + x_res1

        x_shape = x.shape  # dim [batch_size, 4, 100]
        x = self.lorentz_dot(x, x).reshape(x_shape[0], x_shape[-1])
        x_res2 = x
        x = self.fc4(x) + x_res2  # dim [bs, 1]
        x = self.relu(x)
        x = self.fc5(x)
        return x


def main(args):
    print(f"model: {args.model}")
    trainset = ParticleInteraction(args.traindata_size)
    testset = ParticleInteraction(args.testdata_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    BS = args.batch_size
    lr = args.lr
    NUM_EPOCHS = args.num_epochs

    if args.model == "grep":
        model = GRepsNet_SO13(c_in=4, c_h=args.channels, c_out=1).to(device)
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
    parser = argparse.ArgumentParser(description="Argument parser for Lorentz() synthetic dataset")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--traindata_size", default=1000, type=int)
    parser.add_argument("--testdata_size", default=1000, type=int)
    parser.add_argument("--model", default="grep", choices=["grep"])
    parser.add_argument("--channels", default=100, type=int)

    args = parser.parse_args()

    main(args)
