import torch
import torch.nn as nn
from .kde import KDE
from .gmm import GMM
import numpy as np


class BayesianEstimator:
    def __init__(self, x, fitness, alpha, density='gmm', h=0.1):
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



class LatentBayesianEstimator(BayesianEstimator):
    def __init__(self, x: torch.tensor, latent: torch.tensor, fitness: torch.tensor, alpha, density='gmm', h=0.1):
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

class FlowGeneratorMatching:
    def __init__(self, x, theta_generator, alpha, teacher_model,
                 density='kde', h=0.1, lambda_kl=0.5, lambda_ce=1.0,
                 lambda_fm=1.0, reflow_based=True):
        """
        FGM生成器（基于2410.19310v1论文）
        Args:
            x: 初始架构参数（噪声或现有种群）
            theta_generator: 学生生成器参数（g_theta）
            alpha: 时间步参数 (t, t_prev)，用于DDIM
            teacher_model: 教师流模型（预训练的Network）
            density: 密度估计算法（kde/gmm）
            lambda_fm: FGM损失权重
            reflow_based: 是否基于ReFlow的条件向量场（u_t|x0=(x0-xt)/(1-t)）
        """
        self.x = x  # 输入噪声/架构参数（xt）
        self.theta = theta_generator  # 学生生成器参数（需要更新）
        self.alpha = alpha  # (alpha_t, alpha_t_prev)
        self.teacher_model = teacher_model.eval()
        self.lambda_fm = lambda_fm
        self.reflow_based = reflow_based
        self.device = x.device

        # 初始化在线流模型v_psi（用于近似学生向量场）
        self.psi = self._init_online_flow_model(x.size(-1)).to(self.device)
        self.optimizer_psi = torch.optim.Adam(self.psi.parameters(), lr=5e-5)
        self.optimizer_theta = torch.optim.Adam(self.theta.parameters(), lr=5e-5)

        # 密度估计器（复用BayesianEstimator）
        self.fitness = self._init_fitness(x)  # 初始适应度（NASWOT分数）
        self.estimator = BayesianEstimator(x, self.fitness, alpha[0], density=density, h=h)

    def _init_online_flow_model(self, input_dim):
        """初始化在线流模型v_psi（MLP，适配架构参数维度）"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)  # 输出与输入维度一致（向量场）
        )

    def _init_fitness(self, x):
        """初始化适应度（简化：用随机NASWOT分数，实际从multi-objective-search传递）"""
        return torch.randn(x.size(0), device=self.device)

    def _compute_reflow_conditional_field(self, x_t, x0):
        """ReFlow条件向量场 u_t|x0 = (x0 - x_t) / (1 - t)（FGM论文式3.5）"""
        t = self.alpha[0]
        if t >= 1.0:
            return torch.zeros_like(x_t)
        return (x0 - x_t) / (1 - t + 1e-10)  # 避免除以零

    def _compute_teacher_marginal_field(self, x_t):
        """教师边际向量场 u_t（简化：用教师模型前向传播获取，FGM论文式3.2）"""
        with torch.no_grad():
            self.teacher_model.eval()
            # 假设教师模型输出包含流向量场（需与Network类适配）
            if hasattr(self.teacher_model, 'get_flow_field'):
                return self.teacher_model.get_flow_field(x_t)
            #  fallback：用ReFlow边际场近似
            x0_teacher = self.teacher_model(x_t)  # 教师模型预测x0
            return self._compute_reflow_conditional_field(x_t, x0_teacher)

    def _train_psi(self, x0, x_t):
        """冻结theta，训练psi：最小化L_FM（v_psi与u_t|x0的L2损失，FGM论文式4.4）"""
        self.psi.train()
        self.optimizer_psi.zero_grad()
        
        # 计算条件向量场u_t|x0（ReFlow）
        u_t_cond = self._compute_reflow_conditional_field(x_t, x0)
        # 学生向量场v_psi（输入x_t）
        v_psi = self.psi(x_t)
        
        # L_FM损失
        l_fm = torch.norm(v_psi - u_t_cond, dim=-1).mean()
        l_fm.backward()
        self.optimizer_psi.step()
        return l_fm.item()

    def _train_theta(self, x_t):
        """冻结psi，训练theta：最小化L_FGM = L1 + L2（FGM论文式4.10）"""
        self.theta.train()
        self.optimizer_theta.zero_grad()
        self.psi.eval()

        # 1. 计算L1：E[||u_t - v_psi(sg(theta))||²]（stop-gradient on theta）
        x0_sg = self.theta(x_t).detach()  # sg(theta)：冻结theta
        v_psi_sg = self.psi(x_t).detach()  # 冻结psi
        u_t_marg = self._compute_teacher_marginal_field(x_t)
        l1 = torch.norm(u_t_marg - v_psi_sg, dim=-1).mean()

        # 2. 计算L2：E[2*(u_t - v_psi_sg)·(v_psi_sg - u_t|x0)]（FGM论文式4.9）
        u_t_cond = self._compute_reflow_conditional_field(x_t, x0_sg)
        cross_term = (u_t_marg - v_psi_sg) * (v_psi_sg - u_t_cond)
        l2 = 2 * cross_term.sum(dim=-1).mean()

        # 总FGM损失
        l_fgm = self.lambda_fm * (l1 + l2)
        l_fgm.backward()
        self.optimizer_theta.step()
        return l_fgm.item(), l1.item(), l2.item()

    def _ddim_one_step_sample(self, x_t, x0_est):
        """FGM一步DDIM采样（论文式4.10后续采样逻辑）"""
        alpha_t, alpha_t_prev = self.alpha
        sigma_t = 0.0  # 确定性采样（FGM用零噪声）
        
        # 计算噪声预测
        eps_pred = (x_t - alpha_t ** 0.5 * x0_est) / ((1 - alpha_t) ** 0.5 + 1e-10)
        # 一步更新
        x_next = (alpha_t_prev ** 0.5) * x0_est + \
                 ((1 - alpha_t_prev - sigma_t ** 2) ** 0.5) * eps_pred
        return x_next.detach()

    def generate(self, inputs, targets, noise=1.0):
        """FGM核心生成逻辑：交替训练psi→theta→一步DDIM生成"""
        # 1. 生成x0_est（学生生成器输出）
        x0_est = self.theta(self.x)

        # 2. 交替训练psi和theta（FGM论文算法1）
        # 2.1 训练psi（冻结theta）
        l_fm = self._train_psi(x0_est.detach(), self.x)
        # 2.2 训练theta（冻结psi）
        l_fgm, l1, l2 = self._train_theta(self.x)

        # 3. 一步DDIM生成新架构参数
        x_next = self._ddim_one_step_sample(self.x, x0_est)

        # 4. 更新适应度和密度估计器
        self.fitness = self._update_fitness(x_next)  # 从multi-objective-search获取NASWOT分数
        self.estimator.append(BayesianEstimator(x_next, self.fitness, self.alpha[0]))

        # 日志输出
        print(f"FGM Loss: L_FM={l_fm:.4f}, L_FGM={l_fgm:.4f} (L1={l1:.4f}, L2={l2:.4f})")
        return x_next

    def _update_fitness(self, x_next):
        """更新适应度（实际需从multi-objective-search传递NASWOT分数）"""
        return torch.randn(x_next.size(0), device=self.device)  # 占位：后续替换为真实NASWOT

    def get_optimized_theta(self):
        """获取训练后的学生生成器参数"""
        return self.theta.state_dict()

    def get_optimized_psi(self):
        """获取训练后的在线流模型参数"""
        return self.psi.state_dict()

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
    """集成SIM的单步生成器，融合教师模型软标签蒸馏"""
    def __init__(self, x, fitness, alpha, teacher_model, density='gmm', h=0.1, lambda_kl=0.5, lambda_ce=1.0):
        super().__init__(x, fitness, (alpha, alpha), density, h)  # 固定单步扩散参数α一致
        self.teacher_model = teacher_model.eval()  # 冻结教师模型
        self.lambda_kl = lambda_kl  # KL散度权重
        self.lambda_ce = lambda_ce  # 交叉熵权重
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.model = None  # 假设model会在外部设置

    def generate(self, inputs, targets, noise=1.0):
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

class BayesianGenerator:
    def __init__(self, x, fitness, alpha, density='gmm', h=0.1):
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
    def __init__(self, x, latent, fitness, alpha, density='gmm', h=0.1):
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
