


from typing import Optional
import torch.distributed as dist
import torch
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor


# global batch frame norm/global mean real-time var
class GlobalNorm4(torch.nn.Module):

    def __init__(self, dims: int, dim2:int,worldsize:int,momentum: int =0.1, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.bias = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.momentum=momentum
        self.worldsize=worldsize
        self.mean_scale = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.register_buffer('global_mean', torch.zeros(1, 16, dim2, 1))

        self.reset_parameters()
        
    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x):
        B, F, N, D = x.shape  
        global_mean=x.mean(dim=[0,-1], keepdim=True)
        if self.training:
            with torch.no_grad():
                if dist.is_initialized():
                    dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
                    global_mean = global_mean / self.worldsize  
                self.global_mean.mul_(1-self.momentum).add_(self.momentum*global_mean)
        else:
            global_mean=self.global_mean


        out=x-global_mean*self.mean_scale
        var=torch.sum(out.pow(2),dim=[-1],keepdim=True)/(D-1)
        std=(var+self.eps).sqrt()
        return out * self.weight/std +self.bias
    


    def __repr__(self):
        return f'{self.__class__.__name__}({self.dims})'

