"""Small pixel-token transformer for bounded real-data attention experiments."""
import torch
from torch import nn
import torch.nn.functional as F


class DigitAttention(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Linear(1,128);self.position=nn.Parameter(torch.empty(1,64,128));nn.init.normal_(self.position,std=.02)
        self.norm1=nn.LayerNorm(128);self.qkv=nn.Linear(128,384);self.projection=nn.Linear(128,128)
        self.norm2=nn.LayerNorm(128);self.ff1=nn.Linear(128,256);self.ff2=nn.Linear(256,128);self.norm3=nn.LayerNorm(128);self.classifier=nn.Linear(128,10)

    def forward(self,x,attention=None,trace=False):
        embedded=self.embed(x[:,:64,None])+self.position;normalized=self.norm1(embedded);qkv=self.qkv(normalized).reshape(x.shape[0],64,3,2,64)
        q,k,v=(qkv[:,:,index].permute(0,2,1,3).contiguous() for index in range(3))
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.) if attention is None else attention(q,k,v)
        first=embedded+self.projection(attended.transpose(1,2).reshape(x.shape[0],64,128));second=self.norm2(first)
        hidden=F.relu(self.ff1(second));residual=first+self.ff2(hidden);pooled=self.norm3(residual).mean(dim=1);logits=self.classifier(pooled)
        return (logits,{"embedded":embedded,"normalized":normalized,"q":q,"k":k,"v":v,"attended":attended,"first":first,"second":second,"hidden":hidden,"residual":residual,"pooled":pooled}) if trace else logits
