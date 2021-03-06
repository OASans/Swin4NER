import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, x_mask, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    x_mask = x_mask.view(B, L // window_size, window_size)

    windows = x.contiguous().view(-1, window_size, C)
    x_mask = x_mask.view(-1, window_size)
    return windows, x_mask


def attention_partition(attn, window_size):
    temp = []
    B, L, L = attn.shape
    start, end = 0, window_size
    while end <= L:
        temp.append(attn[:, start:end, start:end])
        start += window_size
        end += window_size
    temp = torch.stack(temp, 1)
    return temp.view(-1, window_size, window_size)


def window_reverse(windows, window_size, seq_len):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (seq_len // window_size))## ????????????wind????????? 64 7 7  96
    x = windows.view(B, seq_len // window_size, window_size, -1)
    x = x.contiguous().view(B, seq_len, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape## x??????????????? 64 49 96 ??????????????????????????????B???64???N???49???C???96
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            mask = mask.unsqueeze(1).expand(attn.shape)
            attn = attn.masked_fill_(mask.bool(), value=torch.tensor(-float('inf')))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = torch.nan_to_num(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===============================Swin Tree Transformer Encoder===============================
class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8, use_cuda=True):
        super(GroupAttention, self).__init__()
        self.d_model = d_model // 2
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        #self.linear_output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda

    def forward(self, context, eos_mask, prior):
        batch_size,seq_len = context.size()[:2]
        eos_mask = eos_mask.unsqueeze(1)
        context =self.norm(context)

        if not self.use_cuda:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1))
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0))
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1))
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0))
        else:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1)).cuda()
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0)).cuda()
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1)).cuda()
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0)).cuda()

        #mask = eos_mask & (a+c) | b
        mask = eos_mask & (a+c)
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        
        if eos_mask.dtype == torch.float16:
            a = a.to(torch.float16)
            b = b.to(torch.float16)
            c = c.to(torch.float16)
            tri_matrix = tri_matrix.to(torch.float16)
            masking_value = -1e4
        else:
            masking_value = -1e9

        scores = scores.masked_fill(mask == 0, masking_value)

        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-9)
        neibor_attn = prior + (1. - prior)*neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a==0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int()-b)==0, 0)     
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b==0, 1e-9)
        
        return g_attn, neibor_attn


## 2.
class WindowTreeAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, group_prob=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape## x??????????????? 64 49 96 ??????????????????????????????B???64???N???49???C???96
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            mask = mask.unsqueeze(1).expand(attn.shape)
            attn = attn.masked_fill_(mask.bool(), value=torch.tensor(-float('inf')))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = torch.nan_to_num(attn)
        if group_prob is not None:
            attn = attn * group_prob.unsqueeze(1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTreeTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, seq_len, dim, shift_size, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowTreeAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def sub_forward(self, x, x_mask, group_prob=None):
        B, L, C = x.shape## ?????????B???1???L???seq_len??????3136???C???????????????96 
        shortcut = x
        x = self.norm1(x)

        # partition windows
        x_windows, x_mask = window_partition(x, x_mask, self.window_size) ## 64 7 7 96  # nW*B, window_size, window_size, C
        group_prob = attention_partition(group_prob, self.window_size)

        x_mask = x_mask.double().unsqueeze(2)
        mask_t = x_mask.permute(0, 2, 1)
        attn_mask = torch.matmul(x_mask, mask_t)
        attn_mask = 1 - attn_mask

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, group_prob=group_prob)## attn_windows 64 49 96??????trm?????????????????????49?????????toen??????96?????????  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, L)  # B H' W' C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    def shifted_forward(self, x, x_mask, shift_size, group_prob=None):
        B, L, C = x.shape## ?????????B???1???L???seq_len??????3136???C???????????????96 
        shortcut = x
        x = self.norm1(x)

        # shift
        x = torch.roll(x, -shift_size, dims=1)
        attn_mask = torch.cat((torch.ones(B, shift_size), 2 * torch.ones(B, L - shift_size)), 1).to(
                        x_mask.device)
        attn_mask = attn_mask * x_mask
        attn_mask = torch.roll(attn_mask, -shift_size, dims=1)
        group_prob = torch.roll(group_prob, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # partition windows
        x_windows, x_mask = window_partition(x, attn_mask, self.window_size)
        group_prob = attention_partition(group_prob, self.window_size)

        # attn mask
        x_mask = x_mask.masked_fill(x_mask == 0, float('inf'))
        attn_mask = x_mask.unsqueeze(1) - x_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, group_prob=group_prob)## attn_windows 64 49 96??????trm?????????????????????49?????????toen??????96?????????  # nW*B, window_size*window_size, C

        # reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, L)  # B H' W' C

        # reverse shift
        x = torch.roll(x, shift_size, dims=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        

    def forward(self, x, x_mask, group_prob):
        x = self.sub_forward(x, x_mask, group_prob)  

        if self.shift_size == 'auto':
            shift_size = self.window_size // 2
            x = self.shifted_forward(x, x_mask, shift_size, group_prob)
        elif self.shift_size == 'cycle':
            shift_size = 1
            while 0 < shift_size <= self.window_size // 2:
                x = self.shifted_forward(x, x_mask, shift_size, group_prob)
                shift_size += 1
        else:
            x = self.sub_forward(x, x_mask, group_prob)

        return x


class SwinTreeTransformerLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, seq_len, dim, shift_size, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_cuda=True):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTreeTransformerBlock(seq_len=seq_len, dim=dim, shift_size=shift_size,
                                 num_heads=num_heads, window_size=window_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        self.group_attn = GroupAttention(dim, use_cuda=use_cuda)

    def forward(self, x, x_mask, neibor_attn):
        group_prob, neibor_attn = self.group_attn(x, x_mask, neibor_attn)
        for blk in self.blocks:
            x = blk(x, x_mask, group_prob)
        return x, neibor_attn


class SwinTreeTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    """

    def __init__(self, config, depths=[1, 1, 3, 1], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = config.num_layers
        self.embed_dim = config.hidden_size
        self.num_features = config.hidden_size
        self.mlp_ratio = mlp_ratio
        self.seq_len = config.model_args.max_seq_length
        self.dropout = config.dropout
        self.shift_size = config.shift_size
        self.use_cuda = True if config.accelerator == 'gpu' else False

        window_size = [self.seq_len // 2 ** (self.num_layers - 1) * 2**i for i in range(self.num_layers)]

        # absolute position embedding
        # self.absolute_pos_embed = nn.Embedding(self.seq_len, self.embed_dim)
        self.pos_encoder = AbsolutePositionalEncoding(config.hidden_size, config.dropout)

        # self.pos_drop = nn.Dropout(p=self.dropout)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTreeTransformerLayer(seq_len=self.seq_len, dim=self.embed_dim,
                                shift_size=self.shift_size,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=self.dropout, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, use_cuda=self.use_cuda)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, x_mask=None):
        x = self.pos_encoder(x)
        # x = self.pos_drop(x)

        neibor_attn = 0.

        for layer in self.layers:
            x, neibor_attn = layer(x, x_mask, neibor_attn)

        x = self.norm(x)  # B L C
        return x