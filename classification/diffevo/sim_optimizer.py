from .ddim import DDIMScheduler
from .generator import BayesianGenerator
from .generator import SIMGenerator
from .fitnessmapping import Identity
import torch
from tqdm import tqdm

from .gmm import GMM

def fitness_function(individuals, valid_queue, criterion, gen, population_instance):
    model.eval()
    teacher_model.eval()  # 确保教师模型处于评估模式
    fitness_values = []
    kl_loss_fn = KLDivLoss(reduction='batchmean')  # KL散度损失函数
    use_teacher_prob = 0.5  # 以50%的概率使用教师模型

    with torch.no_grad():
        for individual in individuals:
            model.update_alphas(individual)  # 更新当前个体的架构参数
            total_acc = 0.0
            total_kl = 0.0
            batch_count = 0

            for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)

                # 学生模型前向传播
                _, student_logits = model(inputs)

                # 以一定概率使用教师模型
                if random.random() < use_teacher_prob:
                    with torch.no_grad():
                        _, teacher_logits = teacher_model(inputs)
                else:
                    teacher_logits = student_logits

                # 计算准确率（原始适应度）
                student_probs = F.softmax(student_logits, dim=1)
                acc = (student_probs.argmax(dim=1) == targets).float().mean()
                total_acc += acc.item() * batch_size

                # 计算KL散度（蒸馏损失）
                student_log_probs = F.log_softmax(student_logits, dim=1)
                teacher_probs = F.softmax(teacher_logits, dim=1)
                kl_loss = kl_loss_fn(student_log_probs, teacher_probs)
                total_kl += kl_loss.item() * batch_size

                batch_count += batch_size

            # 计算平均指标
            avg_acc = total_acc / batch_count
            avg_kl = total_kl / batch_count

            # **关键调整：将KL散度作为惩罚项融入适应度**
            # 适应度 = 准确率 - λ_KL * KL散度（λ_KL可通过参数配置）
            lambda_kl = de.lambda_kl  # 从DiffEvo201实例中获取超参数
            adjusted_fitness = avg_acc - lambda_kl * avg_kl

            fitness_values.append(adjusted_fitness)

    return torch.tensor(fitness_values, device=device)

class DiffEvo:
    """Diffusion evolution algorithm for optimization.

        Args:
            - num_step: int, the number of steps to evolve the population.
            - alpha: str or torch.Tensor, the alpha schedule for the diffusion process.
            - density: str, the mode of the density function, only support 'uniform' and 'kde'.
            - sigma: str, the mode of the sigma, only support 'ddpm' and 'zero'.
            - sigma_scale: float, the scaling factor for the sigma.
            - sample_steps: list of int, the steps to evaluate the fitness.
            - scaling: float, the scaling factor for the population.
            - fitness_mapping: str, the mapping function from fitness to probability, only support 'identity' and 'energy'.
            - temperature: float or list of float, the temperature for the fitness mapping.
            - method: str, the method to estimate the density, only support 'bayesian' and 'nn'.
            - kde_bandwidth: float, the bandwidth for the KDE density estimator.
            - nn: nn.Module, the neural network for the density estimator.

        Methods:
            - step(gt, t): evolve the population by one step.
                outputs:
                    - gt: torch.Tensor, the evolved population.
                    - density: torch.Tensor, the estimated density of the evolved population.

            - optimize(fit_fn, initial_population, trace=False): optimize the population.
                outputs:
                    - population: torch.Tensor, the optimized population.
                    - population_trace: list of torch.Tensor, the population trace during the optimization.
                    - fitness_count: list of float, the fitness count during the optimization.

        Example:
            ```python
            optimizer = DiffEvo(num_step=100, sigma='ddpm', density='kde')
            sampled, trace, fitness = optimizer.optimize(fitness_function_2d, torch.randn(512, 2), trace=True)
            ```
        """
    def __init__(self,
                 teacher_model,
                 num_step: int = 1,
                 density='kde',
                 noise:float=1.0,
                 lambda_kl: float = 0.5,
                 scaling: float=1,
                 fitness_mapping=None,
                 kde_bandwidth=0.1):
        self.num_step = num_step
        self.lambda_kl = lambda_kl
        self.teacher_model = teacher_model


        if not density in ['uniform', 'kde', 'gmm']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')
        self.density = density
        self.kde_bandwidth = kde_bandwidth
        self.scaling = scaling
        self.noise = noise
        if fitness_mapping is None:
            self.fitness_mapping = Identity()
        else:
            self.fitness_mapping = fitness_mapping
        self.scheduler = DDIMScheduler(self.num_step)
        self.used_gmm = density == 'gmm'  # 添加属性以记录是否使用了GMM

    def get_density_method(self):
        """返回使用的密度估计算法"""
        return self.density

    #def optimize(self, fit_fn, initial_population, trace=False):
    def optimize(self, initial_population, train_data, trace=False):  # 参数顺序调整
        x = torch.tensor(initial_population, dtype=torch.float32)  # Convert initial_population to a torch.Tensor


        if x.ndim != 2:
            raise ValueError(f"initial_population should be a 2D tensor, but got shape {x.shape}")
        x = x.to(train_data[0].device)
        fitness_count = []
        population_trace = [x] if trace else []
        #if trace:
          #  population_trace = [x]

        for t, alpha in tqdm(self.scheduler):
            # 构建SIM生成器
            generator = SIMGenerator(
                x,
                fitness=self._compute_fitness(x,train_data),  # 适应度函数,去掉了model
                alpha=alpha,
                teacher_model=self.teacher_model,
                lambda_kl=self.lambda_kl
            )
            generator.model = self.model  # 设置生成器中的模型，改成了self.model

            # 单步生成新架构参数
            x_next = generator.generate(
                inputs=train_data[0],
                targets=train_data[1],
                noise=self.noise
            )

            # 跟踪记录
            if trace:
                trace_list.append(x_next.clone())
            x = x_next

        return x, trace_list if trace else x

    def _compute_fitness(self, arch_params, model, train_data):
        """基于验证集的适应度计算（FID/准确率等）"""
        self.model.copy_arch_parameters(arch_params)  # 加载架构参数，改成了self.model
        with torch.no_grad():
            logits = model(train_data[0])
            acc = (logits.argmax(dim=1) == train_data[1]).float().mean()
        return  fitness_function(arch_params, self.valid_queue, self.criterion, gen, self.population) #acc  # 以准确率作为适应度指标（可替换为FID等）


'''
            fitness = fit_fn(x * self.scaling)
            print(f"Fitness shape after fit_fn: {fitness.shape}")
            print(f"x shape before BayesianGenerator: {x.shape}")
            if self.density == 'uniform' or self.density == 'kde':
                generator = BayesianGenerator(x, self.fitness_mapping(fitness), alpha, density=self.density,
                                              h=self.kde_bandwidth)
            elif self.density == 'gmm':
                print("Using GMM for density estimation.")
                generator = BayesianGenerator(x, self.fitness_mapping(fitness), alpha, density='gmm')

            x = generator(noise=self.noise)
            if trace:
                population_trace.append(x)
            fitness_count.append(fitness)

        if trace:
            population_trace = torch.stack(population_trace) * self.scaling

        if trace:
            return x, population_trace, fitness_count
        else:
            return x
'''
