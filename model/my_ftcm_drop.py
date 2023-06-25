import torch
import torch.nn as nn
#from model.module.trans import Transformer as Transformer_s
#from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        args.d_hid = 512

        # dimension tranfer
        self.pose_emb = nn.utils.weight_norm(nn.Linear(34, args.d_hid, bias=True))
        self.gelu = nn.GELU()

        self.mlpmixer = MlpMixer(6, args.frames,args.d_hid)

        self.pose_lift = nn.utils.weight_norm(nn.Linear(args.d_hid, 3*17, bias=True))

        #self.sequence_pos_encoder = PositionalEncoding(args.d_hid, 0.1)

        self.tem_pool = nn.AdaptiveAvgPool1d(1)
        # self.spa_pool = nn.AdaptiveAvgPool1d(1)
        # self.tem_fc = nn.Linear(args.frames,args.frames)
        # self.ln = nn.LayerNorm(args.d_hid)
        # self.sig = nn.Sigmoid()



    def forward(self, x):

        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() 
        x = x.view(x.shape[0], x.shape[1], -1) 
        B,T,C = x.shape
        x = self.pose_emb(x)
        x = self.gelu(x)


        x = self.mlpmixer(x)


        x = x.transpose(1,2)
        x = self.tem_pool(x)
        x = x.transpose(1,2)
        x = self.pose_lift(x)

        x = x.permute(0, 2, 1).contiguous() 
        x = x.view(x.shape[0], self.num_joints_out, -1, x.shape[2]) 
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) 

        return x



class GlobalFilter(nn.Module):
    def __init__(self, dim, n=243):
        super().__init__()
      #  self.complex_weight = nn.Parameter(torch.randn(41, dim,2, dtype=torch.float32) * 0.02)
      #  self.w = w
      #  self.h = h
        self.n = n
        self.dim = dim//2
        # self.lfilter = nn.Parameter(torch.randn(self.dim, self.n//2+1,2, dtype=torch.float32) * 0.02)
        self.lfilter = nn.Parameter(torch.randn((self.n+1)//2, self.n//2+1,2, dtype=torch.float32) * 0.02)
      #  self.bias = nn.Parameter(torch.zeros((self.n+1)//2,2, dtype=torch.float32))
      #  self.learnedfilter = nn.Linear(self.n//2+1,self.n//2+1)

    def forward(self, x):
        B, N, C = x.shape


        #print(x.shape)
        x = x.to(torch.float32)
       # lfilter = self.learnedfilter(x.transpose(1,2)).transpose(1,2)
       # lfilter = torch.fft.rfft(self.lfilter,n=self.n, dim=1, norm='ortho')


        x = torch.fft.rfft(x,n=self.n, dim=1, norm='ortho')
       # cc = torch.view_as_complex(self.ccc)
        lfilter = torch.view_as_complex(self.lfilter)
        # x = x.transpose(1,2)
        #print(x.shape)
        #exit()
        x = torch.matmul(lfilter,x)
        # x = x.transpose(1,2)
        # x = x*lfilter.transpose(0,1)
       # x = self.learnedfilter(x.transpose(1,2)).transpose(1,2)
        x = torch.fft.irfft(x,n=self.n, dim=1, norm='ortho')
       # x = x*gating
        #print(x.shape)
        #exit()
        #x = x.reshape(B, N, C)

        return x




class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
      #  self.complex_weight = nn.Parameter(torch.randn(41, dim,2, dtype=torch.float32) * 0.02)
      #  self.w = w
      #  self.h = h
        self.dim = dim
        self.ln_1 = nn.Linear(self.dim,self.dim,bias=True)
        self.ln_2 = nn.Linear(self.dim,self.dim,bias=True)
        self.gelu  = nn.GELU()
        self.drop = nn.Dropout(0.1)
        #self.reweight = Mlp(512,128,512*2)


    def forward(self, x):
        B,T,C = x.shape

        res = self.ln_1(x)
        res = self.gelu(res) 
        res = self.drop(res) 
        res = self.ln_2(res)
        res = self.gelu(res) 
        # res = self.drop(res)

        return res








class Hrnet2(nn.Module):
    def __init__(self,dim):
        super().__init__()
      #  self.complex_weight = nn.Parameter(torch.randn(41, dim,2, dtype=torch.float32) * 0.02)
      #  self.w = w
      #  self.h = h
        self.dim = dim
        self.ln_1024 = nn.Linear(self.dim,self.dim,bias=True)
        self.ln_512 = nn.Linear(self.dim//2,self.dim//2,bias=True)
        self.ln_256 = nn.Linear(self.dim//4,self.dim//4,bias=True)
        self.ln_128 = nn.Linear(self.dim//8,self.dim//8,bias=True)
        self.gelu  = nn.GELU()
        self.drop = nn.Dropout(0.1)
        #self.reweight = Mlp(512,128,512*2)


    def forward(self, x):
        B,T,C = x.shape

        res_11,res_12 = torch.chunk(x,2,-1)
        res_12 = self.ln_512(res_12)
        res_12 = self.gelu(res_12)

        # res_21,res_22 = torch.chunk(res_12,2,-1)
        # res_22 = self.ln_256(res_22)
        # res_22 = self.gelu(res_22)

        # res_31,res_32 = torch.chunk(res_22,2,-1)
        # res_32 = self.ln_128(res_32)
        # res_32 = self.gelu(res_32)

        res = torch.cat((res_11,res_12),-1)
        # res = torch.cat((res,res_31),-1)
        # res = torch.cat((res,res_32),-1)  

        res = self.drop(res)
        res = self.ln_1024(res)
        res = self.gelu(res)
        # res = self.drop(res)  
        

        return res




class ECL(nn.Module):
    def __init__(self, dim, n=243):
        super().__init__()
        group = 2
        cdiv = 2
        self.conv1 = nn.Conv1d(dim, dim//cdiv, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.conv2 = nn.Conv1d(dim//cdiv, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.gelu = nn.GELU()
        self.sig = nn.Sigmoid()
        self.ln1 = nn.LayerNorm(dim//cdiv)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self,x):
        B,T,C = x.shape
        gating = self.conv1(x.transpose(1,2)).transpose(1,2)
        gating = self.ln1(gating)
        gating = self.gelu(gating)
        gating = self.conv2(gating.transpose(1,2)).transpose(1,2)
        gating = self.ln2(gating)
        gating = self.sig(gating)
        x = x*gating
        return x







class SpatialGatingUnit(nn.Module):
    def __init__(self,dim,len_sen):
        super().__init__()
        self.in_dim = dim
        self.len_sen =len_sen
        #self.proj=nn.Linear (351,243)
        #self.gelu  = nn.GELU()
        #self.lpm = LearnedPosMap1D(win_size=self.len_sen,gamma=4) 
        self.ecl =ECL(dim=self.in_dim//2,n=self.len_sen)
        self.gfp = GlobalFilter(dim=self.in_dim,n=self.len_sen)
        #self.ln1=nn.LayerNorm(dim//2)
        #self.ln2=nn.LayerNorm(dim//2)
    
    def forward(self,x):
        res,gate=torch.chunk(x,2,-1) #bs,n,d_ff
        #gate=self.ln1(gate) #bs,n,d_ff
        #res=self.ln2(res) #bs,n,d_ff
        ###Spatial Proj        #r
        res = self.gfp(res) #bs,n,d_ff
        #res = self.gelu(res)
        #res = res.transpose(1,2)

        gate= self.ecl(gate)

        return torch.cat((res,gate),-1)





class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) +x

class MlpMixer(nn.Module):
    def __init__(self, num_block, num_object, in_dim):
        super(MlpMixer, self).__init__()

        self.num_block = num_block
        self.token_mlp_dim = num_object
        self.channel_mlp_dim = in_dim
        self.d_ff = in_dim
        self.len_sen = num_object

        self.gmlp=nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d'%i,nn.LayerNorm(self.channel_mlp_dim)),
            ('fc1_%d'%i,nn.Linear(self.channel_mlp_dim,self.d_ff)),
            ('gelu_%d'%i,nn.GELU()),
            ('SG%d'%i,SpatialGatingUnit(self.d_ff,self.len_sen)),
            ('fc2_%d'%i,Hrnet2(self.d_ff)),
        ])))  for i in range(self.num_block)])

    def forward(self, input):
        # blocks layers
        out = nn.Sequential(*self.gmlp)(input)
        return out



if __name__ == "__main__":
    inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Model()
    output = net(inputs)
    print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs,))
    print(flops)
    print(params)
    """
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name,':',param.size())
    """