import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

class k3Config:
   d_model = 64

class RMSNorm(nn.Module):
      def __init__(self , dim , eps=1e-6):
          super().__init__()
          self.eps = eps 
          self.weight = nn.parameter(torch.ones(dim))
      def forward(self, x):
          norm = x.pow(2).mean(-1, keepdim = True)
          return x * torch.rsqrt(norm + self.eps) * self.weight
          
