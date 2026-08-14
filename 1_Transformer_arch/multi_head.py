from attn_mask import causal_mask
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module): 
    """1.4 Multi-head Attention Mechanism with explicit shape tracing 

        Dimensions (before masking):
        x:          (B, T, d_model)                         -> Input of Batch size, Sequence of Tokens & d_model as input
        qkv:        (B, T, 3*d_model)                       -> When pushed through qkv, 3 copies of x are made, x is a matrix
        view:       (B, T, 3, n_head, d_head)               -> Then we push it through the 3 linear layers
        split:      (q, k, v each (B, T, n_head, d_head)    -> Then we get the individual q, k, v values which are n⨯head * d⨯head for each token and batch. So there will be 4 dimentional tensors which will be split along 3 dimensions
        swap:       (B, n_head, T, d_head)                  -> Finding the transpose of the Key Vector in the mathematical formula of the Attention mechanism.
        scores:     (B, n_head, T, T) = q * k^T / sqrt(d_head)
        weights:    (B, n_head, T, T) = softmax(scores)
        ctx:        (B, n_head, T, d_head) = weights * v    -> We get the weights matrix
        merge:      (B, T, n_head * d_head) = (B, T, d_model) -> We concatenate all the attention heads together, this will be along the third dimension i.e., which will bring us back to d_model

        This is the output of the every Multi-headed Self Attention Mechanism
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):

        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
    
    # Implementing a forward pass for a mult-headed self attention mechanism where we pass in a sentence
    # through all the math operations and layers and spit out the final prediction at the end
    
    def forward(self, x: torch.Tensor):         # (B, T, 3*C)
        
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)           # (B, T, 3, heads, dim)

        if self.trace_shapes:
            print ("q k v view:", qkv.shape)

        q, k, v = qkv.unbind(dim=2)                                 # Each: (B, T, heads, dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.trace_shapes:
            print("q: ", q.shape, "k: ", k.shape, "v: ", v.shape)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale         # (B, heads, T, T)
        mask = causal_mask(T, device=x.device)                      # Creating a Causal Mask
        attn = attn.masked_fill(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v)

        if self.trace_shapes:
            print("Weights: ", w.shape, "ctx: ", ctx.shape)

        out = ctx.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        if self.trace_shapes:
            print("out: ", out.shape)

        return out, w