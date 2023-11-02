##############################
# self attention layer
# author Xu Mingle
# time Feb 18, 2011
##############################
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1) # Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1) # Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)  # Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H) torch.Size([1, 8, 65536])
        g = self.g(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height) # B * C * (W * H)
        
        attention = torch.bmm(f.permute(0, 2, 1), g) # B * (W * H) * (W * H)
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention) # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H
        
        out = self.gamma * self_attetion + x # torch.Size([1, 64, 256, 256])
        return out
    
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # (b,H*W,in_channel//8)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # (b,in_channel//8,H*W)
        energy = torch.bmm(proj_query, proj_key) # (b,H*W,H*W)
        attention = self.softmax(energy) # (b,H*W,H*W)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # (b,in_channels,H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))# bmm (b,in_channels,H*W) * (b,H*W,H*W) = (b,in_channels,H*W) 
        out = out.view(m_batchsize, C, height, width) # (B,in_channels,H,W)

        out = self.gamma*out + x # (B,in_channels,H,W)
        return out 

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.attention = PAM_Module(in_dim=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        m = self.attention(x)
        attention_map = self.conv2(m)
        return attention_map
    
class SelfBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.attention = SelfAttention(in_dim=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        m = self.attention(x)
        attention_map = self.conv2(m)
        return attention_map



class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.q_proj = nn.Linear(in_dim, k_dim * n_heads, bias=False) # (b=1,)
        self.k_proj = nn.Linear(in_dim, k_dim * n_heads, bias=False)
        self.v_proj = nn.Linear(in_dim, v_dim * n_heads, bias=False)
        self.out_proj = nn.Linear(v_dim * n_heads, in_dim, bias=False)
        
        
    def forward(self, x):
        batch_size, _, h, w = x.size()
        
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1) # 将输入x展开成一维张量
        print('x1:',x.shape)
        
        q1 = self.q_proj(x)
        print('q1',q1.shape)
        
        q = self.q_proj(x).view(batch_size, self.n_heads, self.k_dim, h*w).permute(0, 1, 3, 2) # (1,8,64,H*W) -> (1,8,H*W,64)
        k = self.k_proj(x).view(batch_size, self.n_heads, self.k_dim, h*w) # (1,8,64,H*W)
        v = self.v_proj(x).view(batch_size, self.n_heads, self.v_dim, h*w) # (1,8,64,H*W)
        print('q k v',q.shape,k.shape,v.shape)
        attn_weights = torch.softmax(torch.matmul(q, k), dim=-1) / torch.sqrt(torch.tensor(self.k_dim).float())
        attn_output = torch.matmul(attn_weights, v).view(batch_size, self.n_heads * self.v_dim, h, w)
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention1(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class SelfAttentionBlock1(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock1, self).__init__()
        
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.attention = MultiHeadAttention(in_dim=in_channels, k_dim=64, v_dim=64, n_heads=8)
        self.attention = MultiHeadAttention1(n_head=8, d_model=256, d_k=256, d_v=256)
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # x = self.conv1(x) # x: torch.Size([1, 64, 256, 256])
        print('x:',x.shape)
       
       
        m = self.attention(x)
        # attention_map = self.conv2(m)
        return m
    
