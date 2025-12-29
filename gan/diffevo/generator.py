import torch
import torch.nn as nn
import torch.nn.functional as F
from .kde import KDE
from .gmm import GMM
import numpy as np


class BayesianEstimator:
    def __init__(self, x, fitness, alpha, density='kde', h=0.1):
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be list, np.ndarray, or torch.Tensor, but got {type(x)}")

        if x.ndim > 2:
            x = x.view(-1, x.size(-1))

        if x.ndim != 2:
            raise ValueError(f"x should be a 2D tensor, but got shape {x.shape}")

        if isinstance(fitness, (list, np.ndarray)):
            fitness = torch.tensor(fitness)
        elif not isinstance(fitness, torch.Tensor):
            raise TypeError(f"Expected fitness to be list, np.ndarray, or torch.Tensor, but got {type(fitness)}")

        if fitness.ndim != 1:
            raise ValueError(f"Fitness should be a 1D tensor, but got shape {fitness.shape}")

        assert x.size(0) == fitness.size(0), "First dimension of x and fitness should match."

        self.x = x.to(x.device if x.is_cuda else "cpu")
        self.fitness = fitness.to(x.device if x.is_cuda else "cpu")
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if density not in ['uniform', 'kde', 'gmm']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

    def append(self, estimator):
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)

    def density(self, x):
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0]) / x.shape[0]
        elif self.density_method == 'kde':
            return KDE(x, h=self.h)
        elif self.density_method == 'gmm':
            return GMM(self.x)

    @staticmethod
    def norm(x):
        if x.shape[-1] == 1:
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 3))

    def _estimate(self, x_t, p_x_t):
        device = self.x.device
        x_t = x_t.to(device)
        p_x_t = p_x_t.to(device)

        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)

        prob = (self.fitness.to(device) + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)

        target_shape = self.x.shape
        while len(prob.shape) < len(target_shape):
            prob = prob.unsqueeze(-1)

        if prob.shape != target_shape:
            prob = prob.expand_as(self.x)

        z = torch.sum(prob, dim=0, keepdim=True)
        origin = torch.sum(prob * self.x, dim=0) / (z + 1e-9)

        return origin

    def estimate(self, x_t):
        p_x_t = self.density(x_t)
        results = []
        for i in range(len(x_t)):
            result = self._estimate(x_t[i], p_x_t[i])
            results.append(result)
        return torch.stack(results)

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'
class ConsistencyGenerator:
    def __init__(self, x, fitness, alpha, teacher_model,
                 density='kde', h=0.1,
                 lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1):
        super().__init__(x, fitness, (alpha, alpha), density, h)  # 固定单步扩散参数α一致
        self.fitness = fitness
        self.alpha = alpha
        self.teacher_model = teacher_model.eval()
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.model = None
        self.estimator = BayesianEstimator(x, fitness, alpha, density, h)

    def generate(self, inputs, targets, noise=1.0):
        """生成新的架构参数"""
        # 1. 贝叶斯估计获取架构参数分布
        x0_est = self.estimator(self.x)  # [N, arch_params]

        # 2. 学生模型前向传播
        student_logits = self._model_forward(x0_est, inputs)

        # 3. 交叉熵损失（硬标签损失）
        ce = self.ce_loss(student_logits, targets)

        # 4. 教师模型软标签蒸馏（软标签损失）
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.log_softmax(student_logits, dim=1)
        kl = self.kl_loss(student_probs, teacher_probs)

        # 5. 一致性损失
        consistency_loss = self._compute_consistency_loss(x0_est, inputs)

        # 7. 综合损失函数
        total_loss = (self.lambda_ce * ce +
                      self.lambda_kl * kl +
                      self.lambda_consistency * consistency_loss)
        total_loss.backward()  # 反向传播更新生成器参数

        # 8. 单步DDIM生成（一步扩散过程）
        x_next = self._ddim_step(self.x, x0_est, self.alpha, noise=noise)
        return x_next.detach()  # 返回新生成的架构参数

    def _compute_consistency_loss(self, arch_params, inputs):
        """计算一致性损失"""
        # 对架构参数添加随机扰动
        perturbed_params = arch_params + torch.randn_like(arch_params) * self.perturb_scale

        # 获取原始参数和扰动参数的模型输出
        with torch.no_grad():
            original_outputs = self._model_forward(arch_params, inputs)
            perturbed_outputs = self._model_forward(perturbed_params, inputs)

        # 计算一致性损失
        if self.consistency_type == 'mse':
            return F.mse_loss(original_outputs, perturbed_outputs)
        elif self.consistency_type == 'kl':
            original_probs = F.softmax(original_outputs, dim=1)
            perturbed_log_probs = F.log_softmax(perturbed_outputs, dim=1)
            return self.kl_loss(perturbed_log_probs, original_probs)
        else:
            raise ValueError(f"Unsupported consistency type: {self.consistency_type}")

    def _ddim_step(self, x_t, x0_pred, alpha, noise=1.0):
        """单步DDIM采样过程"""
        alpha_t, alpha_prev = alpha
        sigma_t = 0  # 确定性采样

        # 计算噪声预测
        eps_pred = (x_t - x0_pred * alpha_t.sqrt()) / ((1 - alpha_t).sqrt() + 1e-10)

        # 一步DDIM更新
        if alpha_prev > 0:
            x_prev = (alpha_prev.sqrt() * x0_pred +
                      (1 - alpha_prev - sigma_t ** 2).sqrt() * eps_pred +
                      sigma_t * torch.randn_like(x_t) * noise)
        else:
            x_prev = x0_pred  # 当alpha_prev为0时，直接返回x0_pred

        return x_prev

    def _model_forward(self, arch_params, inputs):
        """将架构参数映射到模型并前向传播"""
        # 假设arch_params是一个扁平化的张量，需要重塑为架构矩阵
        # 这里需要根据实际架构表示进行调整
        batch_size = arch_params.shape[0]
        arch_matrix = arch_params.view(batch_size, -1)  # 重塑为[batch_size, num_params]

        # 将架构参数应用到模型
        self.model.copy_arch_parameters(arch_matrix)

        # 前向传播
        return self.model(inputs)

class SIMGenerator(BayesianEstimator):
    def __init__(self, x, fitness, alpha, teacher_model, density='kde', h=0.1, lambda_kl=0.5, lambda_ce=1.0):
        super().__init__(x, fitness, alpha, density, h)
        self.teacher_model = teacher_model.eval()
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.model = None

    def generate(self, inputs, targets, noise=1.0):
        x0_est = self(self.x)

        student_logits = self._model_forward(x0_est, inputs)

        ce = self.ce_loss(student_logits, targets)

        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.log_softmax(student_logits, dim=1)
        kl = self.kl_loss(student_probs, teacher_probs)

        score_diff = self._compute_score_difference(x0_est, inputs, student_logits, teacher_logits)
        c = 1.0  # 超参数
        sim_loss = torch.sqrt(torch.norm(score_diff, p=2) ** 2 + c ** 2) - c


        total_loss = self.lambda_ce * ce + self.lambda_kl * kl + sim_loss
        total_loss.backward()

        x_next = self._ddim_step(self.x, x0_est, self.alpha, noise)
        return x_next.detach()

    def _model_forward(self, arch_params, inputs):
        arch_matrix = arch_params.view(14, 8).to(inputs.device)
        self.model.copy_arch_parameters(arch_matrix)
        return self.model(inputs)

    def _compute_score_difference(self, arch_params, inputs, student_logits, teacher_logits):
        student_grad = torch.autograd.grad(student_logits.sum(), inputs, create_graph=True)[0]
        with torch.no_grad():
            teacher_grad = torch.autograd.grad(teacher_logits.sum(), inputs, create_graph=True)[0]
        return student_grad - teacher_grad

    def _ddim_step(self, xt, x0, alpha, noise):
        sigma = self._ddpm_sigma(alpha, alpha) * noise
        eps = (xt - (alpha ** 0.5) * x0) / (1.0 - alpha) ** 0.5
        if sigma is None:
            sigma = self._ddpm_sigma(alpha, alpha)
        x_next = (alpha ** 0.5) * x0 + ((1 - alpha - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
        return x_next

    def _ddpm_sigma(self, alphat, alphatp):
        return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5


class LatentBayesianEstimator(BayesianEstimator):
    def __init__(self, x: torch.tensor, latent: torch.tensor, fitness: torch.tensor, alpha, density='kde', h=0.1):
        super().__init__(x, fitness, alpha, density=density, h=h)
        self.z = latent

    def _estimate(self, z_t, p_z_t):
        mu = self.z * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(z_t, mu, sigma)

        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_z_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)

        return origin

    def estimate(self, z_t):
        p_z_t = self.density(self.z)
        results = []
        for i in range(len(z_t)):
            result = self._estimate(z_t[i], p_z_t[i])
            results.append(result)
        return torch.stack(results)


def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise
    eps = (xt - (alphat ** 0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next


def ddpm_sigma(alphat, alphatp):
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5

class BayesianGenerator:
    def __init__(self, x, fitness, alpha, density='kde', h=0.1):
        print(f"x shape before BayesianEstimator: {x.shape}")
        print(f"fitness shape before BayesianEstimator: {fitness.shape}")

        # Discard the second dimension of x if it is appropriate
        if x.dim() == 3 and x.size(1) == 100:
            x = x[:, 0, :]  # Take only the first slice along the second dimension

        #print(f"x shape after discarding a dimension: {x.shape}")

        if x.size(0) != fitness.size(0):
            raise ValueError(f"Mismatch in samples: x has {x.size(0)} samples but fitness has {fitness.size(0)} samples.")

        self.x = x
        self.fitness = fitness
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)

    def generate(self, noise=1.0, return_x0=False):
        x0_est = self.estimator(self.x)
        x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)


class LatentBayesianGenerator(BayesianGenerator):
    def __init__(self, x, latent, fitness, alpha, density='kde', h=0.1):
        self.x = x
        self.latent = latent
        self.fitness = fitness
        self.alpha, self.alpha_past = alpha
        self.estimator = LatentBayesianEstimator(self.x, self.latent, self.fitness, self.alpha, density=density, h=h)

    def generate(self, noise=1.0, return_x0=False):
        x0_est = self.estimator(self.latent)
        x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next
