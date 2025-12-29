import torch
import torch.nn as nn
from .kde import KDE
from .gmm import GMM
import numpy as np


class BayesianEstimator:
    def __init__(self, x, fitness, alpha, density='kde', h=0.3):
        # Ensure x is a numpy ndarray or torch tensor, then convert and reshape
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be list, np.ndarray, or torch.Tensor, but got {type(x)}")

        # Reshape x to a 2D tensor
        if x.ndim > 2:
            x = x.view(-1, x.size(-1))

        if x.ndim != 2:
            raise ValueError(f"x should be a 2D tensor, but got shape {x.shape}")

        # Ensure fitness is a numpy ndarray or torch tensor, then convert
        if isinstance(fitness, (list, np.ndarray)):
            fitness = torch.tensor(fitness)
        elif not isinstance(fitness, torch.Tensor):
            raise TypeError(f"Expected fitness to be list, np.ndarray, or torch.Tensor, but got {type(fitness)}")

        # Ensure fitness is 1D
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
            return GMM(self.x)(x)

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


class LatentBayesianEstimator(BayesianEstimator):
    def __init__(self, x: torch.tensor, latent: torch.tensor, fitness: torch.tensor, alpha, density='kde', h=0.3):
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
    def __init__(self, x, fitness, alpha, density='kde', h=0.3):
        print(f"x shape before BayesianEstimator: {x.shape}")
        print(f"fitness shape before BayesianEstimator: {fitness.shape}")

        # Discard the second dimension of x if it is appropriate
        if x.dim() == 3 and x.size(1) == 50:
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
    def __init__(self, x, latent, fitness, alpha, density='kde', h=0.3):
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

class SIMGenerator(BayesianEstimator):
    """集成SIM的单步生成器，融合教师模型软标签蒸馏"""
    def __init__(self, x, fitness, alpha, teacher_model, density='kde', h=0.3, lambda_kl=0.5, lambda_ce=1.0):
        super().__init__(x, fitness, (alpha, alpha), density, h)  # 固定单步扩散参数α一致
        self.teacher_model = teacher_model.eval()  # 冻结教师模型
        self.lambda_kl = lambda_kl  # KL散度权重
        self.lambda_ce = lambda_ce  # 交叉熵权重
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.model = None  # 假设model会在外部设置

    def generate(self, inputs, targets, noise=1.5):
        # 1. 贝叶斯估计获取架构参数预测
        x0_est = self.estimator(self.x)  # [N, arch_params]

        # 2. 学生模型前向传播
        student_logits = self._model_forward(x0_est, inputs)  # 生成架构对应的模型输出

        # 3. 交叉熵损失（硬标签损失）
        ce = self.ce_loss(student_logits, targets)

        # 4. 教师模型软标签蒸馏（软标签损失）
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.log_softmax(student_logits, dim=1)
        kl = self.kl_loss(student_probs, teacher_probs)

        # 5. SIM分数匹配损失（论文核心公式3.7变形）
        # 假设score_diff为学生/教师分数函数差异，此处简化实现
        score_diff = self._compute_score_difference(x0_est, inputs, student_logits, teacher_logits)
        #sim_loss = torch.norm(score_diff, p=2)  # 可替换为Pseudo - Huber距离
        c = 1.0  # 超参数
        sim_loss = torch.sqrt(torch.norm(score_diff, p=2) ** 2 + c ** 2) - c

        # 6. 综合损失函数（融合硬标签、软标签、分数匹配）
        total_loss = self.lambda_ce * ce + self.lambda_kl * kl + sim_loss
        total_loss.backward()  # 反向传播更新生成器参数

        # 7. 单步DDIM生成（固定单步扩散过程）
        x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha), noise=noise)
        return x_next.detach()  # 生成新种群参数

    def _model_forward(self, arch_params, inputs):
        """架构参数到模型的映射函数（需根据实际模型调整）"""
        # 示例：假设模型参数为112维，对应14x8的架构矩阵
        arch_matrix = arch_params.view(14, 8).to(inputs.device)
        self.model.copy_arch_parameters(arch_matrix)  # 假设model有参数复制接口
        return self.model(inputs)  # 返回学生模型输出

    def _compute_score_difference(self, arch_params, inputs, student_logits, teacher_logits):
        """简化的分数函数差异计算（需结合论文公式实现）"""
        # 此处需根据教师/学生模型的分数函数实际定义补充
        # 示例：假设分数函数为logits的梯度
        student_grad = torch.autograd.grad(student_logits.sum(), inputs, create_graph=True)[0]
        with torch.no_grad():
            teacher_grad = torch.autograd.grad(teacher_logits.sum(), inputs, create_graph=True)[0]
        return student_grad - teacher_grad  # 分数函数差异