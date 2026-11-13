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

class SiTUExpert(nn.Module):
    def __init__(self , d, beta1 = 4.0 , beta2 = 25.0, hidden_mult = 2 ):
        super().__init__()
        h = d * hidden_mult
        self.Wg = nn.Linear(d , h , bias = False)
        self.Wu = nn.Linear(d , h , bias = False)
        self.Wd = nn.Linear(h , d , bias = False)
        self.beta1 , self.beta2 = beta1 , beta2

    def forward(self , x):
        g , u = self.wg(x) , self.wu(x)
        gate = self.beta1 * torch.tanh(g / self.beta1) * torch.sigmoid(g)
        up = slef.beta2 * torch.tanh(u / self.beta2)
        return self.Wd(gate * up)

 # The equation to implement ``yt =Wo[Sigmoid(Wgxt)⊙o ̃t]``
class GatedMLA(nn.Module):
    def __init__(self , d_model ,  n_heads, d_head):
        super().__init__()
        self.n_heads , self.d_head = n_heads , d_head
        inner = n_heads * d_head
        self.q = nn.Linear(d_model , inner , bias=False)
        self.k = nn.Linear(d_model , inner , bias=False)
        self.v = nn.Linear(d_model , inner , bias=False)
        self.gate = nn.Linear(d_model , inner, bias=False)
        self.out = nn.Linear(inner, d_model, bais=False)
        self.scale = d_head ** -0.5
        
    def forwad(self , x):
        B , T , D = x.shape
        H , DH = self.n_heads * self.d_head
        q = self.q(x).view(B , T , H , DH).transpose(1,2)
        k = self.k(x).view(B , T , H , DH).transpose(1,2)
        v = self.v(x).view(B , T , H , DH).transpose(1,2)

        attn = torch.einsum('bhtd , bhsd -> bhts' , q , k ) * self.scale
        mask = torch.triu(torch.ones(T , T , device = x.device , dtype = torch.bool) , diagonal = 1)
        attn =  attn.masked_fill(mask , float(-inf)).softmax(dim=-1)
        o = torch.einsum('bhts, bhsd -> bhtd' , attn , v).transpose(1,2).reshape(B, T, H * DH)

        gate = torch.sigmoid(self.gate(x))
        return self.out(gate * o)
       
   
