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
class Inertia(object):
    def __init__(self, N=1024, k=5):
        super().__init__()
        self.dim = (1 + 3) * k
        self.X = np.random.randn(N, self.dim)
        self.X[:, :k] = np.log(1 + np.exp(self.X[:, :k]))  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = np.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] * (r2 * I - ri[..., None] * ri[..., None, :])).sum(1)
        self.Y = inertia.reshape(-1, 9)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


# G-RepsNet model definition
class GRepsNet_O3(tnn.Module):
    def __init__(self, c_in=2, c_h=100, c_out=1):
        super().__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.t0_ln1a, self.t0_ln1b = tnn.Linear(c_in + c_in, c_h, bias=True), tnn.Linear(c_h, c_h, bias=True)
        self.t1_ln1a, self.t1_ln1b = tnn.Linear(c_in, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)
        self.t2_ln1a, self.t2_ln1b = tnn.Linear(c_in + c_in, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)

        self.t0_ln2a, self.t0_ln2b = tnn.Linear(c_h + c_h + c_h, c_h, bias=True), tnn.Linear(c_h, c_h, bias=True)
        self.t1_ln2a, self.t1_ln2b = tnn.Linear(c_h, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)
        self.t2_ln2a, self.t2_ln2b = tnn.Linear(c_h + c_h + c_h, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)

        self.t0_ln3a, self.t0_ln3b = tnn.Linear(c_h + c_h + c_h, c_h, bias=True), tnn.Linear(c_h, c_h, bias=True)
        self.t1_ln3a, self.t1_ln3b = tnn.Linear(c_h, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)
        self.t2_ln3a, self.t2_ln3b = tnn.Linear(c_h + c_h + c_h, c_h, bias=False), tnn.Linear(c_h, c_h, bias=False)

        self.fc4 = tnn.Linear(c_h, c_out, bias=False)
        self.relu = tnn.ReLU()

    def higher_order_mixing(self, x, layer_type, weights=None):
        assert layer_type in ["01->012", "012->012"]

        if layer_type == "01->012":
            # dim [BS, 1+3, -1]
            x_shape = x.shape
            t0_lna, t0_lnb, t1_lna, t1_lnb, t2_lna, t2_lnb = weights
            x_T0, x_T1 = x[:, :1, :], x[:, 1:4, :]

            # 0 mixing
            # 1->0
            x_T1_0 = torch.norm(x_T1, dim=1, keepdim=True)
            x_T0_out = torch.cat([x_T0.clone(), x_T1_0], dim=-1)  # dim [BS, 1, 2*ch]

            x_T0_out = t0_lnb(self.relu(t0_lna(x_T0_out)))  # dim [BS, 1, ch]

            # 1 mixing
            # 0->1
            x_T1_out = t1_lna(x_T1)
            x_T1_out = x_T0_out * x_T1_out / torch.norm(x_T1_out)
            x_T1_out = t1_lnb(x_T1_out)

            # 2 mixing
            # 0->2
            x_T0_T2 = x_T0 * (torch.eye(3, device=x_T0.device).reshape(1, 9, 1))  # dim [BS, 9, ch]

            # 1->2
            x_T1_T2 = x_T1.reshape(x_shape[0], 3, 1, x_shape[2]) * x_T1.reshape(
                x_shape[0], 1, 3, x_shape[2]
            )  # dim [BS, 3, 3, ch]
            x_T1_T2 = x_T1_T2.reshape(x_shape[0], 9, x_shape[2])  # dim [BS, 9, ch]

            x_T2 = torch.cat([x_T0_T2, x_T1_T2], dim=-1)

            # concat for final x
            x_T2 = t2_lna(x_T2)
            x_T2 = x_T0_out * x_T2 / torch.norm(x_T2)
            x_T2 = t2_lnb(x_T2)

            x_T2_out = x_T2
            out = torch.cat([x_T0_out, x_T1_out, x_T2_out], dim=-2)
            return out

        elif layer_type == "012->012":
            # dim [BS, 1+3+9, -1]
            t0_lna, t0_lnb, t1_lna, t1_lnb, t2_lna, t2_lnb = weights
            x_shape = x.shape
            x_T0, x_T1, x_T2 = x[:, :1, :], x[:, 1:4, :], x[:, 4:13, :]

            # 0 mixing
            x_T1_0 = torch.norm(x_T1, dim=1, keepdim=True)
            x_T2_0 = torch.norm(x_T2, dim=1, keepdim=True)
            x_T0_out = torch.cat([x_T0.clone(), x_T1_0, x_T2_0], dim=-1)  # dim [BS, 1, 3*ch]
            x_T0_out = t0_lnb(self.relu(t0_lna(x_T0_out)))  # dim [BS, 1, ch]

            # 1 mixing
            x_T1_out = t1_lna(x_T1)  # dim [BS, 1, ch]
            x_T1_out = x_T0_out * x_T1_out / torch.norm(x_T1_out)  # dim [BS, 1, ch]
            x_T1_out = t1_lnb(x_T1_out)  # dim [BS, 1, ch]

            # 0->2
            x_T1_T2 = x_T1.reshape(x_shape[0], 3, 1, -1) * x_T1.reshape(x_shape[0], 1, 3, -1)  # dim [BS, 3, 3, ch]
            x_T1_T2 = x_T1_T2.reshape(x_shape[0], 9, -1)  # dim [BS, 9, ch]
            x_T0_T2 = x_T0 * (torch.eye(3, device=x_T0.device).reshape(1, 9, 1))  # dim [BS, 9, ch]

            # 2->2
            x_T2 = torch.cat([x_T0_T2, x_T1_T2, x_T2], dim=-1)  # dim [BS, 9,3*ch]

            x_T2 = t2_lna(x_T2)
            x_T2 = x_T0_out * x_T2 / torch.norm(x_T2)
            x_T2_out = t2_lnb(x_T2) + x_T2

            out = torch.cat([x_T0_out, x_T1_out, x_T2_out], dim=-2)
            return out

    def forward(self, x):
        # dim x [batch_size, 5*1, 5*3]  # 5*V0 + 5*V1
        x_shape = x.shape
        x_scalars = x[:, :5].reshape(-1, 5, 1)  # dim [BS, 5*1]
        x_vectors = x[:, 5:].reshape(-1, 5, 3)  # dim [BS, 5*3]
        x = torch.cat([x_scalars, x_vectors], dim=2)  # dim [BS, 5, (1+3)]
        x = x.reshape(-1, 5, 1 + 3).permute(0, 2, 1)  # dim x [batch_size, 1+3, 5]  # V + V

        # 01 -> 012
        x = self.higher_order_mixing(
            x,
            layer_type="01->012",
            weights=[self.t0_ln1a, self.t0_ln1b, self.t1_ln1a, self.t1_ln1b, self.t2_ln1a, self.t2_ln1b],
        )

        x_res = x
        x = self.higher_order_mixing(
            x,
            layer_type="012->012",
            weights=[self.t0_ln2a, self.t0_ln2b, self.t1_ln2a, self.t1_ln2b, self.t2_ln2a, self.t2_ln2b],
        )
        x = x + x_res.clone()

        x_res = x
        x = self.higher_order_mixing(
            x,
            layer_type="012->012",
            weights=[self.t0_ln3a, self.t0_ln3b, self.t1_ln3a, self.t1_ln3b, self.t2_ln3a, self.t2_ln3b],
        )
        x = x + x_res.clone()

        x = x[:, 4:, :]  # take only T2 output from [T0, T1, T2]

        x = self.fc4(x).reshape(x_shape[0], 9)
        x = x.reshape(x_shape[0], 9)
        return x


def main(args):
    print(f"model: {args.model}")
    trainset = Inertia(args.traindata_size)
    testset = Inertia(args.testdata_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    BS = args.batch_size
    lr = args.lr
    NUM_EPOCHS = args.num_epochs

    if args.model == "grep":
        model = GRepsNet_O3(c_in=5, c_h=args.channels, c_out=1).to(device)
    else:
        raise NotImplementedError

    def num_params(model):
        return sum([p.numel() for n, p in model.named_parameters()])

    print(f"num params: {num_params(model)}")
    num_parameters = num_params(model)

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
    parser = argparse.ArgumentParser(description="Argument parser for O(3) inertia dataset")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="use 1e-3 for grep")
    parser.add_argument("--traindata_size", default=1000, type=int)
    parser.add_argument("--testdata_size", default=1000, type=int)
    parser.add_argument("--model", default="grep", choices=["grep"])
    parser.add_argument("--channels", default=100, type=int)

    args = parser.parse_args()

    if args.model != "grep":
        raise NotImplementedError
    main(args)
