import torch
import torch.nn as nn

from typing import Dict, Tuple, Union

from torch import Tensor
from torch.nn import Module


def align_shape(x: Tensor, coeff):
    if isinstance(coeff, Dict):
        for k, v in coeff.items():
            v = align_shape(x, v)
            coeff[k] = v
    elif isinstance(coeff, Tensor):
        while len(coeff.shape) < len(x.shape):
            coeff = coeff.unsqueeze(-1)
    else:
        raise ValueError("coeff must be either a dict or a tensor")
    
    return coeff


class FlowNoiser(Module):
    def __init__(
        self,
        device,
        training_timesteps: int,
        inference_timesteps: int,
        gamma_min: float,
        gamma_max: float,
        simplify: bool = True,
        reparam: str = None
    ):
        super(FlowNoiser, self).__init__()
        self.device = device
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.training_timesteps = training_timesteps
        self.training_timestep_map = torch.arange(
            0, training_timesteps, 1,
            dtype=torch.long, device=device
        )
        self.inference_timesteps = inference_timesteps
        self.inference_timestep_map = torch.arange(
            0, inference_timesteps, training_timesteps // inference_timesteps,
            dtype=torch.long, device=device
        )
        self.num_timesteps = training_timesteps
        self.timestep_map = self.training_timestep_map

        self.simplify = simplify
        if simplify and reparam is not None:
            if reparam not in ('FR', 'TR'):
                raise ValueError("reparam must be either 'FR' or 'TR'")
        self.reparam = reparam

    def train(self, mode=True):
        self.num_timesteps = self.training_timesteps if mode else self.inference_timesteps
        self.timestep_map = self.training_timestep_map if mode else self.inference_timestep_map

    def eval(self):
        self.train(mode=False)

    def coefficient(self, t: Union[Tensor, int]) -> Dict:
        if isinstance(t, Tensor):
            t = t.max()

        if t >= len(self.timestep_map):
            coeff_0, coeff_1 = torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)
        else:
            timestep = self.timestep_map[t].float()
            coeff_1 = timestep / self.num_timesteps
            coeff_0 = 1. - coeff_1

        return {'coeff_0': coeff_0, 'coeff_1': coeff_1}
    
    def prepare_gammas(self) -> Tensor:
        gammas = torch.linspace(
            self.gamma_min, self.gamma_max, self.num_timesteps // 2,
            device=self.device
        )
        self.gammas = torch.cat([gammas, gammas.flip(0)])

    def forward(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Union[Tensor, int]
    ) -> Tensor:
        coeff = align_shape(x_0, self.coefficient(t))
        coeff_0, coeff_1 = coeff['coeff_0'], coeff['coeff_1']
        x_t = coeff_0 * x_0 + coeff_1 * x_1

        return x_t
    
    @torch.no_grad()
    def trajectory(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        trajectory = [x_0.clone()]
        for t in range(self.num_timesteps):
            x_t = self.forward(x_0, x_1, t)
            trajectory.append(x_t.clone())
        
        return torch.stack(trajectory)

    def forward_F(self, x: Tensor, x_1: Tensor, t: Union[Tensor, int]) -> Tensor:
        coeff_0_t = align_shape(x, self.coefficient(t)['coeff_0'])
        coeff_0_t_plus_one = align_shape(x, self.coefficient(t + 1)['coeff_0'])

        vec = (x_1 - x) / coeff_0_t
        F_x = x + (coeff_0_t - coeff_0_t_plus_one) * vec

        return F_x
    
    @torch.no_grad()
    def trajectory_F(self, x_0: Tensor, x_1: Tensor, sample: bool = False) -> Tuple[Tensor]:
        self.prepare_gammas()
        ones = torch.ones((x_0.size(0), ), dtype=torch.long, device=self.device)

        x = x_0
        x_all, gt_all, t_all = [], [], []

        for t in range(0, self.num_timesteps):
            t_ts = ones * t
            t_all.append(t_ts)

            F_x = self.forward_F(x, x_1, t_ts)

            if sample and t == self.num_timesteps -1:
                x_next = F_x
            else:
                x_next = F_x + (2. * self.gammas[t].sqrt() * torch.randn_like(x))
            x_all.append(x_next.clone())

            # S-DSB
            if self.simplify:
                if self.reparam == 'TR':
                    gt_all.append(x_0)
                elif self.reparam == 'FR':
                    vec = (x_0 - x_next) / self.coefficient(t + 1)['coeff_1']
                    gt_all.append(vec)
                else:
                    gt_all.append(x.clone())
            else:
                F_x_next = self.forward_F(x, x_1, t_ts)
                gt_all.append(x_next + F_x -F_x_next)

            x = x_next

        x_all = torch.stack([x_0] + x_all).cpu() if sample else torch.cat(x_all).cpu()
        gt_all = torch.cat(gt_all).cpu()
        t_all = torch.cat(t_all).cpu()

        return x_all, gt_all, t_all 


if __name__ == "__main__":
    training_timesteps = 16
    inference_timesteps = 16
    gamma_min = 1e-4
    gamma_max = 1e-3
    simplify = True
    reparam = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noiser = FlowNoiser(
        device,
        training_timesteps, inference_timesteps,
        gamma_min, gamma_max,
        simplify=simplify, reparam=reparam
    )
