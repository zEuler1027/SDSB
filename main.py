import math
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import Tensor, nn
from typing import Tuple
from torch.nn import Module
import tqdm
from model import DSB
from noiser import FlowNoiser


class Checkerboard(Dataset):
    def __init__(self, size=8, grid_size=4):
        self.size = size
        self.grid_size = grid_size
        self.checkboard = torch.tensor([[i, j] for i in range(grid_size) for j in range(grid_size) if (i + j) % 2 == 0])

        grid_pos = torch.randint(low=0, high=self.checkboard.shape[0], size=(self.size,), dtype=torch.int64)
        self.data = torch.rand(size=(self.size, 2), dtype=torch.float32) + self.checkboard[grid_pos].float()
        self.data = self.data / self.grid_size * 2 - 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class Pinwheel(Dataset):
    def __init__(self, npar: int):
        self.size = npar

        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 7
        num_per_class = math.ceil(npar / num_classes)
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        x = .4 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

        self.init_sample = torch.from_numpy(x).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tensor:
        return self.init_sample[idx]


data_size = 2 ** 20
pinwheel_dataset = Pinwheel(data_size)
checkerboard_dataset = Checkerboard(size=data_size, grid_size=8)

from torch.utils.data import DataLoader


def create_data_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = False,
    num_workers: int = 0, pin_memory: bool = False
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )


batch_size = 2 ** 12
pinwheel_data_loader = create_data_loader(pinwheel_dataset, batch_size, num_workers=0, pin_memory=True)
checkerboard_data_loader = create_data_loader(checkerboard_dataset, batch_size, num_workers=0, pin_memory=True)

import matplotlib.pyplot as plt


def show_2d_data(data: Tensor):
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.show()
    plt.close()

import os
import time
import tqdm
import matplotlib.animation as anime

from torch.optim import AdamW


class Runner:
    def __init__(
        self, device,
        data_loader_0: DataLoader, data_loader_1: DataLoader,
        noiser: Module, backward_model: Module, forward_model: Module,
        lr: float = 1e-3, weight_decay: float = 0.,
        save_path: str = '.'
    ):
        self.device = device
        self.save_path = save_path

        self.data_loader_0, self.data_loader_1 = data_loader_0, data_loader_1
        self.data_iter_0, self.data_iter_1 = iter(data_loader_0), iter(data_loader_1)

        self.num_batches = len(data_loader_0)
        self.batch_size = data_loader_0.batch_size
        # 每一对数据 (x_0, x_1) 都对应有完整的一条轨迹(x_{k-1}, x_k, x_{k+1}..),
        # 因此先缓存下一批数据避免每次迭代都重新计算
        self.cache_size = self.cnt = self.num_batches * self.batch_size * 4

        self.loss_fn = nn.MSELoss()

        self.noiser = noiser
        self.backward_model = backward_model.to(device)
        self.forward_model = forward_model.to(device)

        self.backward_optimizer = AdamW(
            self.backward_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.forward_optimizer = AdamW(
            self.forward_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.model_dict = {
            'backward': self.backward_model,
            'forward': self.forward_model
        }
        self.optimizer_dict = {
            'backward': self.backward_optimizer,
            'forward': self.forward_optimizer
        }

        self.direction = 'backward'

    def get_paired_data(self) -> Tuple[Tensor]:
        try:
            x_0, x_1 = next(self.data_iter_0), next(self.data_iter_1)
        except StopIteration:
            self.data_iter_0 = iter(self.data_loader_0)
            self.data_iter_1 = iter(self.data_loader_1)
            x_0, x_1 = next(self.data_iter_0), next(self.data_iter_1)

        return x_0.to(self.device), x_1.to(self.device)

    def get_batch(self, epoch: int) -> Tuple[Tensor]:
        # 当缓存里的数据已经取完, 则重新计算
        if self.cnt + self.batch_size > self.cache_size:
            self.x_cache, self.gt_cache, self.t_cache = [], [], []

            num_pairs = math.ceil(self.cache_size / (self.batch_size * self.noiser.num_timesteps))
            pbar = tqdm.trange(num_pairs, desc=f"Caching data on epoch {epoch} for {self.direction} model..")
            for _ in pbar:
                x_0, x_1 = self.get_paired_data()
                with torch.no_grad():
                    # 首个 epoch 训练 backward model
                    if epoch == 0:
                        if self.direction != 'backward':
                            raise RuntimeError("Epoch 0 should be backward model.")
                        # 由于 forward model 还未训练, 因此只能根据线性插值计算 backward model 的 data & target.
                        x_all, gt_all, t_all = self.noiser.trajectory_F(x_0, x_1)
                    else:
                        # 若当前训练的是 forward model, 则其 data & target 由 backward model 推理生成;
                        # 反之, 若训练的是 backward model, 则 data & target 由 forward model 推理生成.
                        model = self.model_dict['backward' if self.direction == 'forward' else 'forward'].eval()
                        x_start = x_1 if self.direction == 'forward' else x_0
                        x_all, gt_all, t_all = model.inference(x_start)

                self.x_cache.append(x_all)
                self.gt_cache.append(gt_all)
                self.t_cache.append(t_all)

            self.x_cache = torch.cat(self.x_cache).cpu()
            self.gt_cache = torch.cat(self.gt_cache).cpu()
            self.t_cache = torch.cat(self.t_cache).cpu()

            self.cnt = 0
            self.cache_indices = torch.randperm(self.x_cache.size(0))

        # 每次取1个 batch 并记录取到哪里
        indices = self.cache_indices[self.cnt:self.cnt + self.batch_size]
        self.cnt += self.batch_size

        x_batch = self.x_cache[indices].to(self.device)
        gt_batch = self.gt_cache[indices].to(self.device)
        t_batch = self.t_cache[indices].to(self.device)

        return x_batch, gt_batch, t_batch

    def train(
        self, n_epochs: int, repeat_per_epoch: int,
        log_interval: int = 128, eval_interval: int = 1024
    ):
        steps_per_epoch = self.num_batches * repeat_per_epoch
        self.cache_size = min(self.cache_size, steps_per_epoch * self.batch_size)

        for epoch in range(n_epochs):
            self.noiser.train()

            # 两个 model 交替训练
            self.direction = 'backward' if epoch % 2 == 0 else 'forward'
            model, optimizer = self.model_dict[self.direction], self.optimizer_dict[self.direction]

            model.train()
            optimizer.zero_grad()

            self.cnt = self.cache_size
            pbar = tqdm.tqdm(total=steps_per_epoch)

            # 使用 ema loss 来观察整体趋势
            ema_loss, ema_loss_w = None, lambda x: min(0.99, x / 10)

            for step in range(steps_per_epoch):
                x_t, gt, t = self.get_batch(epoch)
                pred = model(x_t, t)
                loss = self.loss_fn(pred, gt)
                loss.backward()
                loss = loss.item()

                optimizer.step()
                optimizer.zero_grad()

                ema_loss = loss if ema_loss is None \
                    else (ema_loss * ema_loss_w(step) + loss * (1 - ema_loss_w(step)))

                if (step + 1) % log_interval == 0 or step == steps_per_epoch - 1:
                    info = f'Epoch: [{epoch}]/[{n_epochs}]; Step: {step}; Direction: {self.direction}; Loss: {loss:.14f}; Ema Loss: {ema_loss:.14f}'
                    pbar.set_description(info, refresh=False)
                    pbar.update(step + 1 - pbar.n)

                # 训练到一定步数就推理看看效果
                if (step + 1) % eval_interval == 0:
                    self.evaluate(epoch, step + 1)
                    model.train()

            self.evaluate(epoch, steps_per_epoch, last_step=True)

    @torch.no_grad()
    def evaluate(self, epoch: int, step: int, last_step: bool = False):
        self.backward_model.eval()
        self.forward_model.eval()

        x_0, x_1 = self.get_paired_data()
        qs = self.backward_model.inference(x_1, sample=True)[0]
        # 首个 epoch forward model 还未训练, 因此插值计算出整条 forward 轨迹.
        ps = self.forward_model.inference(x_0, sample=True)[0] \
            if epoch else self.noiser.trajectory_F(x_0, x_1, sample=True)[0]

        # 画出两个 model 最后1步生成的样本
        self.draw_sample(qs[-1], epoch, step, subfix='_0')
        self.draw_sample(ps[-1], epoch, step, subfix='_1')

        # 画出两个方向对应的整条轨迹
        if (epoch + 1) % 2 == 0 and last_step:
            self.draw_trajectory(qs, epoch, subfix='_0')
            self.draw_trajectory(ps, epoch, subfix='_1')

    def draw_sample(
        self, sample: Tensor,
        epoch: int, step: int,
        xrange=(-1, 1), yrange=(-1, 1),
        subfix: str = None
    ):
        sample = sample.cpu().numpy()

        save_path = os.path.join(
            self.save_path,
            f'sample' + (subfix if subfix is not None else ''),
        )
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 10))
        plt.scatter(sample[:, 0], sample[:, 1], s=1)
        plt.xlim(xrange[0] - 0.1, xrange[1] + 0.1)
        plt.ylim(yrange[0] - 0.1, yrange[1] + 0.1)

        plt.savefig(os.path.join(save_path, f'ep{epoch}_it{step}.jpg'))
        plt.close()

    def draw_trajectory(self, xs: Tensor, epoch: int, xrange=(-1, 1), yrange=(-1, 1), subfix: str = None):
        save_path = os.path.join(self.save_path,
            f'trajectory' + (subfix if subfix is not None else ''),
            f'ep{epoch}'
        )
        os.makedirs(save_path, exist_ok=True)

        k = 1000
        xs = xs.cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.scatter(
            x=np.reshape(xs[:, :k, 0], -1),
            y=np.reshape(xs[:, :k, 1], -1),
            s=1, cmap='viridis', vmin=0, vmax=1,
            c=np.reshape(np.repeat(np.expand_dims(np.arange(xs.shape[0]), 1), k, axis=1), -1) / xs.shape[0]
        )
        plt.xlim(xrange[0] - 0.1, xrange[1] + 0.1)
        plt.ylim(yrange[0] - 0.1, yrange[1] + 0.1)

        plt.savefig(os.path.join(save_path, 'trajectory.jpg'))
        plt.close()

        self.draw_animation(xs, save_path, xrange=xrange, yrange=yrange)

    def draw_animation(self, xs: Tensor, save_path: str, xrange=(-1, 1), yrange=(-1, 1)):
        clamp = lambda x, a, b: int(min(max(x, a), b))

        st = time.perf_counter()
        num_timesteps, batch_size = xs.shape[0], xs.shape[1]
        steps_per_second = clamp(num_timesteps / 100, 1, 10)
        frames_per_second = clamp(num_timesteps / 10, 1, 10)
        num_seconds = num_timesteps / frames_per_second / steps_per_second + 3

        print('plotting point cloud animation ......', end='', flush=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(xrange[0] - 0.1, xrange[1] + 0.1)
        ax.set_ylim(yrange[0] - 0.1, yrange[1] + 0.1)
        scatter = ax.scatter([], [], s=1, c=[], cmap='viridis', vmin=0, vmax=1)

        def animate(j):
            j = min((j + 1) * steps_per_second, xs.shape[0])
            cc = np.arange(j) / (num_timesteps - 1)
            cc = np.reshape(np.repeat(np.expand_dims(cc, axis=1), batch_size, axis=1), -1)
            scatter.set_offsets(np.reshape(xs[j - 1], (-1, 2)))
            scatter.set_array(cc[-batch_size:])
            return scatter,

        ani = anime.FuncAnimation(fig, animate, frames=int(num_seconds*frames_per_second), interval=1000/frames_per_second, repeat=False, blit=True)
        try:
            ani.save(os.path.join(save_path, 'trajectory.mp4'), writer=anime.FFMpegWriter(fps=frames_per_second, codec='h264'), dpi=100)
        except:
            ani.save(os.path.join(save_path, 'trajectory.gif'), writer=anime.PillowWriter(fps=frames_per_second), dpi=100)

        plt.close(fig)
        print(f' done! ({time.perf_counter() - st:.3f}s)')

dim_in = dim_out = 2

training_timesteps = 16
inference_timesteps = 16
gamma_min = 1e-4
gamma_max = 1e-3
simplify = True
reparam = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

noiser = FlowNoiser(
    device,
    training_timesteps, inference_timesteps,
    gamma_min, gamma_max,
    simplify=simplify, reparam=reparam
)
backward_model = DSB(device, 'b', noiser, 2, 2)
forward_model = DSB(device, 'f', noiser, 2, 2)

lr = 1e-3  #@param {'type': 'number'}
save_path = 'exp2d'  #@param {'type': 'string'}

runner = Runner(
    device,
    checkerboard_data_loader, pinwheel_data_loader,
    noiser, backward_model, forward_model,
    lr=lr, save_path=save_path
)

n_epochs = 41  #@param {'type': 'integer'}
repeat_per_epoch = 8  #@param {'type': 'integer'}

runner.train(n_epochs, repeat_per_epoch)
