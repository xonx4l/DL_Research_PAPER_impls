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

class SwiGLUExpert(nn.Module):
    def __init__(self , d_in , d_out, hidden_mult=2):
        super().__init__()
        h = d_in * hidden_mult
        self.W_gate = nn.Linear(d_in, h , bias = False)
        self.W_up = nn.Linear(d_in, h , bias = False)
        self.W_down = nn.Linear(h , d_out, bias = False)

    def forward(self , x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
          
