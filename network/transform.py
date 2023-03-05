import torch
import torch.nn as nn
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]

        # Linearly project the queries, keys, and values.
        qkv = self.qkv_proj(torch.cat([query, key, value], dim=-1))
        q, k, v = qkv.chunk(3, dim=-1)

        # Split queries, keys, and values into multiple heads.
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute the dot-product attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / (self.embed_dim ** 0.5)

        # Apply the attention mask, if given.
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply the softmax function to compute the attention weights.
        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # Compute the weighted sum of the values.
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate and linearly project the attention output.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                    self.embed_dim)  # [batch_size, seq_len, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(self.linear2(self.dropout(torch.relu(self.linear1(self.norm1(src2))))))
        src = self.norm2(src)
        return src


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=7, dim=10, depth=5, heads=4, mlp_dim=5, dropout=0):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # 将输入的图片分成patch
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer的encoder部分
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # MLP分类器
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # 提取patch，添加位置编码，添加CLS token
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)

        # 通过Transformer的encoder部分
        for transformer_layer in self.transformer:
            x = transformer_layer(x)

        # 取出CLS token的输出，进行分类
        x = x[:, 0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    vit = ViT(224, 16, 3, 4, 5)
    print(vit)



