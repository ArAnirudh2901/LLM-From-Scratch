import torch
from multi_head import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid, save_matrix_heatmap
from pos_encoding import SinusoidalPositionalEncoding

def main():
    B, T, d_model, n_head = 1, 6, 16, 4
    x = torch.randn(B, T, d_model)
    
    pos_enc = SinusoidalPositionalEncoding(max_len=10, d_model=d_model)
    x_pos = pos_enc(x)
    save_matrix_heatmap(x_pos[0].detach().numpy(), "Input Embeddings + Positional Encoding", "input_pos_encoding.png", xlabel="d_model", ylabel="Token Pos")
    
    attn = MultiHeadSelfAttention(d_model=d_model, n_head=n_head, trace_shapes=False)
    _, weights = attn(x_pos)
    save_attention_heads_grid(weights.detach().numpy(), "multi_head_attention.png", title_prefix="Head")

if __name__ == "__main__":
    main()
