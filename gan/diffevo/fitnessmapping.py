"""
This module contains classes of fitness mapping function.
"""
import torch
import logging

class Identity:
    """Identity fitness mapping function."""
    def __init__(self, l2_factor=0.0):
        self.l2_factor = l2_factor

    def l2(self, x):
        if isinstance(x, float):
            x = torch.tensor([x])  # 将标量转换为张量
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or float, but got {type(x)}")
        return torch.norm(x, dim=-1) ** 2

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x) * torch.exp(-1.0 * self.l2(x) * self.l2_factor)


class Energy(Identity):
    """Fitness mapping function that treats the fitness as energy.

    Args:
        temperature: float, the temperature of the system.

    Returns:
        p: torch.Tensor, the probability of the fitness. Compute by exp(-x / temperature).
    """
    def __init__(self, temperature=1.0, l2_factor=0.0):
        super().__init__(l2_factor=l2_factor)
        self.temperature = temperature

    def forward(self, x):
        logging.debug(f"Forward input x: {x}, type: {type(x)}")
        if isinstance(x, float):
            x = torch.tensor([x])
            power = -x / self.temperature
            if power.dim() == 0:  # Check if power is still a scalar tensor
                power = power.unsqueeze(0)
            power = power - power.max() + 5  # This ensures power is a tensor
            p = torch.exp(power)
        # Additional logic...
        return p


class Power(Identity):
    """Fitness mapping function that returns the power of the fitness.

    Args:
        power: float, the power of the fitness.
        temperature: float, the temperature of the system.
    
    Returns:
        p: torch.Tensor, the probability of the fitness. Compute by (x / temperature) ** power.
    """
    def __init__(self, power=1.0, temperature=1.0, l2_factor=0.0):
        super().__init__(l2_factor=l2_factor)
        self.power = power
        self.temperature = temperature
    
    def forward(self, x):
        return torch.pow(x / self.temperature, self.power)