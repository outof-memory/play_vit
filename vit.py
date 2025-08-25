from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim

        assert img_size%patch_size==0, f"img_size: {img_size} is not divisible by patch_size: {patch_size}"
        self.num_patches = (img_size//patch_size)**2
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, imgs):
        # imgs: B x C x H x W
        x = self.proj(imgs)
        return x.flatten(2).transpose(1, 2)

class InputEmbedings(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, dropout_prob=0.1):
        super().__init__()
        self.patch_emb = PatchEmbedding(img_size, patch_size, in_c, embed_dim)
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim)))
        self.pos_emb = nn.Parameter(torch.randn((1, self.patch_emb.num_patches + 1, embed_dim)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, imgs):
        # imgs: B x C x H x W
        patch_emb = self.patch_emb(imgs)
        cls_token = self.cls_token.expand(patch_emb.shape[0], -1, -1)
        x = torch.cat([cls_token, patch_emb], axis=1) + self.pos_emb
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attn_head_size, dropout_prob=0.1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_head_size = attn_head_size
        self.query = nn.Linear(hidden_size, attn_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attn_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attn_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attention_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.attn_head_size) 
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, v)
        return attention_output, attention_probs
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"hidden_size: {hidden_size} is not divisible by num_heads: {num_heads}"
        self.attn_head_size = hidden_size // num_heads
        self.attn_heads = nn.ModuleList([AttentionHead(hidden_size, self.attn_head_size, dropout_prob, bias) for i in range(num_heads)])
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        output = []
        attn_probs = []
        for attn_head in self.attn_heads:
            attn_output, attn_prob = attn_head(x)
            output.append(attn_output)
            attn_probs.append(attn_prob)
        outputs = torch.cat(output, axis=-1)
        outputs = self.output_proj(outputs)
        outputs = self.dropout(outputs)
        attn_probs = torch.stack(attn_probs, axis=1)
        return outputs, attn_probs

class FasterMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"hidden_size: {hidden_size} is not divisible by num_heads: {num_heads}"
        self.attn_head_size = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3*hidden_size)
        self.attn_dropout = nn.Dropout(dropout_prob)

        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.attn_head_size).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.attn_head_size).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.attn_head_size).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.attn_head_size)
        attn_probs = self.attn_dropout(F.softmax(attn_scores, axis=-1))
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.hidden_size)
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        return output, attn_probs

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=2, dropout_prob=0.1, bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.multi_head_attn = MultiHeadAttention(hidden_size, num_heads, dropout_prob, bias)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(mlp_ratio*hidden_size)),
            nn.GELU(),
            # nn.Dropout(dropout_prob),
            nn.Linear(int(mlp_ratio*hidden_size), hidden_size),
            nn.Dropout(dropout_prob))
        
    def forward(self, x):
        attn_output, attn_probs = self.multi_head_attn(self.norm1(x))
        x = attn_output + x
        mlp_output = self.ffn(self.norm2(x))
        output = mlp_output + x
        return output, attn_probs

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_ratio=2, dropout_prob=0.1, bias=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio, dropout_prob, bias) for _ in range(num_layers)])

    def forward(self, x):
        all_attn_probs = []
        for block in self.blocks:
            x, attn_probs = block(x)
            all_attn_probs.append(attn_probs)
        return x, all_attn_probs
    
class ViTForClassfication(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, 
                 mlp_ratio=4, 
                 image_size=384, 
                 patch_size=32,
                 dropout_prob=0.1, 
                 bias=True, 
                 num_classes=1000,
                 initializer_range=0.02):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.initializer_range = initializer_range

        self.input_emb = InputEmbedings(self.image_size, self.patch_size, 3, hidden_size)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, mlp_ratio, dropout_prob, bias)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)

    def forward(self, imgs):
        input_embeddings = self.input_emb(imgs)
        image_embeddings, all_attn_probs = self.encoder(input_embeddings)
        logits = self.fc(image_embeddings[:, 0])
        return logits, all_attn_probs

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, InputEmbedings):
            module.pos_emb.data = nn.init.trunc_normal_(
                module.pos_emb.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.pos_emb.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.cls_token.dtype)









if __name__ == '__main__':
    pass
