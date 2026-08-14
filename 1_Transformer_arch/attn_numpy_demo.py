"""
    1.2 Self - Attention from first principles on a tiny example (NumPy only)
    We use T = 3 tokens, d_model = 4, d_k = 2, single-head.
    This script prints intermediate tensors so you can trace the math.

    Dimensions summary (single head)
    --------------------------------
    X:                          (B = 1, T = 3, d_model = 4)
    Wq/Wk/Wv                    (d_model = 4, d_k = 2)
    Q, K, V                     (1, 3, 2)
    Scores                      (1, 3, 3)           = Q @ K^T
    Weights                     (1, 3, 3)           = softmax over last dim
    Output                      (1, 3, 2)           = Weights @V
"""

import numpy as np

# Toy inputs (batch = 1, seq = 3, d_model = 4)

X = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
              [0.0, 0.1, 0.0, 0.1]], dtype=np.float32)

# Weights matrices (learned in the real models). We fix numbers for the determination
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)

Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)

Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

# Project to Q, K, V

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("Q shape:", Q.shape, "\nQ=\n", Q[0])
print("K shape:", K.shape, "\nK=\n", K[0])
print("V shape:", V.shape, "\nV=\n", V[0])

# Scaled dot products
scale = 1.0 / np.sqrt(Q.shape[-1])
attn_scores = (Q @ K.T) * scale

# Causal mask (upper triangle set to -inf so that softmax function value would be 0)
mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
attn_scores = np.where(mask, -1e9, attn_scores)

# Softmax over the last dim
weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print("Weights shape: ", weights.shape, "\nAttention Weights (Causal)=\n", weights[0])

# Weighted sum of V
out = weights @ V
print("Output shape: ", out.shape, "\nOutput=\n", out[0])