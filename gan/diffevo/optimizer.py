from .ddim import DDIMScheduler
from .generator import ConsistencyGenerator, BayesianEstimator
from .fitnessmapping import Identity
import torch
from tqdm import tqdm
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.fid_score import calculate_activation_statistics, calculate_frechet_distance
import tensorflow as tf
import numpy as np

class DiffEvo:
    def __init__(self,
                 teacher_model,
                 num_step: int = 1,
                 density='kde',
                 noise: float = 1.0,
                 lambda_kl: float = 0.5,
                 lambda_ce: float = 1.0,
                 lambda_consistency: float = 1.0,
                 consistency_type: str = 'mse',
                 perturb_scale: float = 0.1,
                 kde_bandwidth=0.1,
                 scaling: float = 1):
        self.num_step = num_step
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        self.teacher_model = teacher_model
        self.density = density
        self.kde_bandwidth = kde_bandwidth
        self.scaling = scaling
        self.noise = noise

        if not density in ['uniform', 'kde', 'gmm']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

        self.scheduler = DDIMScheduler(self.num_step)
        self.used_gmm = density == 'gmm'

    def get_density_method(self):
        """返回使用的密度估计算法"""
        return self.density

    def optimize(self, initial_population, train_data, model, fid_stat, trace=False):
        x = torch.tensor(initial_population, dtype=torch.float32)
        
        if x.ndim != 2:
            raise ValueError(f"initial_population should be a 2D tensor, but got shape {x.shape}")
        
        device = train_data[0].device
        x = x.to(device)
        fitness_count = []
        population_trace = [x] if trace else []

        for t, alpha in tqdm(self.scheduler, desc="Diffusion steps"):
            # 计算当前种群的适应度
            fitness = self._compute_fitness(x, model, train_data, fid_stat)
            fitness = torch.tensor(fitness, dtype=torch.float32, device=device)
            
            generator = ConsistencyGenerator(
                x,
                fitness=fitness,
                alpha=alpha,
                teacher_model=self.teacher_model,
                density=self.density,
                h=self.kde_bandwidth,
                lambda_kl=self.lambda_kl,
                lambda_ce=self.lambda_ce,
                lambda_consistency=self.lambda_consistency,
                consistency_type=self.consistency_type,
                perturb_scale=self.perturb_scale
            )
            generator.model = model

            x_next = generator.generate(
                inputs=train_data[0],
                targets=train_data[1],
                noise=self.noise
            )

            if trace:
                population_trace.append(x_next.clone())
            x = x_next

        if trace:
            return x, population_trace, fitness_count
        else:
            return x

    def _compute_fitness(self, arch_params, model, train_data, fid_stat):
        # 确保输入是CPU上的numpy数组
        if isinstance(arch_params, torch.Tensor):
            arch_params = arch_params.cpu().numpy()
        
        fitness_values = []
        
        for i in range(arch_params.shape[0]):
            # 复制模型参数
            model.copy_arch_parameters(arch_params[i])
            
            # 生成样本
            with torch.no_grad():
                z = torch.randn(100, model.z_dim).to(train_data[0].device)
                generated_images = model(z)
                
                # 转换图像格式
                generated_images = generated_images.mul(127.5).add(127.5).clamp(0, 255)
                generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            # 计算FID
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                mu_gen, sigma_gen = calculate_activation_statistics(generated_images, sess)
            
            # 加载真实数据统计
            fid_data = np.load(fid_stat)
            mu_real = fid_data['mu']
            sigma_real = fid_data['sigma']
            
            fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            
            # 计算IS
            is_mean, _ = get_inception_score(generated_images)
            
            # 组合适应度：IS越高越好，FID越低越好
            fitness = is_mean - fid
            fitness_values.append(fitness)
        
        return np.array(fitness_values)