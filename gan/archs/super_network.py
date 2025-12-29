from torch import nn
from archs.basic_blocks_search import Cell, DisCell, OptimizedDisBlock


from torch import nn
from archs.basic_blocks_search import Cell, DisCell, OptimizedDisBlock


from torch import nn
from archs.basic_blocks_search import Cell, DisCell, OptimizedDisBlock



import torch.nn.functional as F
'''
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        if args.dataset == 'cifar10':
            # for lower resolution (32 * 32) dataset CIFAR-10
            self.base_latent_dim = args.latent_dim // 3
        else:
            # for higher resolution (48 * 48) dataset STL-10
            self.base_latent_dim = args.latent_dim // 2
        self.l1 = nn.Linear(self.base_latent_dim,
                            (self.bottom_width ** 2) * args.gf_dim)
        self.l2 = nn.Linear(self.base_latent_dim, ((
                                                           self.bottom_width * 2) ** 2) * args.gf_dim)
        if args.dataset == 'cifar10':
            self.l3 = nn.Linear(self.base_latent_dim, ((
                                                               self.bottom_width * 4) ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim), nn.ReLU(), nn.Conv2d(
                args.gf_dim, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z, genotypes):
        h = self.l1(z[:, :self.base_latent_dim]) \
            .view(-1, self.ch, self.bottom_width, self.bottom_width)

        n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2]) \
            .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
        
        # 处理不同数据集的情况
        if self.args.dataset == 'cifar10':
            # CIFAR-10 使用 l3 层
            n2 = self.l3(z[:, self.base_latent_dim * 2:]) \
                .view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)
        else:
            # 对于 STL-10 等其他数据集，重新使用 l2 层的输出并上采样到匹配尺寸
            n2 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2]) \
                .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
            # 使用双线性插值将 n2 上采样到与 h2 相同的尺寸
            target_size = (self.bottom_width * 4, self.bottom_width * 4)
            n2 = F.interpolate(n2, size=target_size, mode='bilinear', align_corners=False)

        h1_skip_out, h1 = self.cell1(h, genotype=genotypes[0])
        h2_skip_out, h2 = self.cell2(h1 + n1, (h1_skip_out,), genotype=genotypes[1])
        _, h3 = self.cell3(h2 + n2, (h1_skip_out, h2_skip_out), genotype=genotypes[2])
        output = self.to_rgb(h3)

        return output
'''
# archs/super_network.py
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.latent_dim = args.latent_dim
        
        # 计算各阶段的特征图尺寸
        self.stage_sizes = [
            self.bottom_width,                 # 第一阶段
            self.bottom_width * 2,             # 第二阶段
            self.bottom_width * 4,             # 第三阶段
            self.args.img_size                 # 最终输出尺寸
        ]
        
        # 确保阶段尺寸不超过目标尺寸
        for i in range(1, len(self.stage_sizes)):
            if self.stage_sizes[i] > self.args.img_size:
                self.stage_sizes[i] = self.args.img_size
        
        # 计算潜在空间划分
        self.num_stages = len(self.stage_sizes) - 1  # 减去初始阶段
        self.latent_per_stage = self.latent_dim // self.num_stages
        
        # 创建线性层
        self.linear_layers = nn.ModuleList()
        for i in range(self.num_stages):
            self.linear_layers.append(
                nn.Linear(
                    self.latent_per_stage, 
                    (self.stage_sizes[i] ** 2) * self.ch
                )
            )
        
        # 创建Cell层
        self.cells = nn.ModuleList()
        for i in range(self.num_stages):
            mode = 'nearest' if i % 2 == 0 else 'bilinear'
            self.cells.append(
                Cell(self.ch, self.ch, mode, num_skip_in=i)
            )
        
        # 输出层
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, genotypes):
        # 初始特征
        h = self.linear_layers[0](z[:, :self.latent_per_stage]) \
            .view(-1, self.ch, self.stage_sizes[0], self.stage_sizes[0])
        
        skip_connections = []
        stage_inputs = []
        
        # 处理每个阶段
        for i in range(1, self.num_stages):
            # 获取当前阶段的潜在向量
            latent_start = i * self.latent_per_stage
            latent_end = (i + 1) * self.latent_per_stage if i < self.num_stages - 1 else self.latent_dim
            stage_z = z[:, latent_start:latent_end]
            
            # 生成阶段特征
            stage_feature = self.linear_layers[i](stage_z) \
                .view(-1, self.ch, self.stage_sizes[i], self.stage_sizes[i])
            
            # 调整尺寸以匹配当前阶段
            if h.size(2) != self.stage_sizes[i]:
                h = F.interpolate(h, size=(self.stage_sizes[i], self.stage_sizes[i]), 
                                 mode='bilinear', align_corners=False)
            
            # 合并特征
            stage_input = h + stage_feature
            
            # 通过Cell处理
            if i < self.num_stages - 1:
                skip_out, h = self.cells[i-1](stage_input, tuple(skip_connections), genotype=genotypes[i-1])
                skip_connections.append(skip_out)
            else:
                _, h = self.cells[i-1](stage_input, tuple(skip_connections), genotype=genotypes[i-1])
        
        # 最终转换为RGB图像
        output = self.to_rgb(h)
        
        # 确保输出尺寸正确
        if output.size(2) != self.args.img_size:
            output = F.interpolate(output, size=(self.args.img_size, self.args.img_size), 
                                  mode='bilinear', align_corners=False)
        
        return output


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block3 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block4 = DisCell(args, self.ch, self.ch, activation=activation)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x, genotypes):
        h = x
        h = self.block1(h)
        h = self.block2(h, genotypes[0])
        h = self.block3(h, genotypes[1])
        h = self.block4(h, genotypes[2])
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output