MLX port of [EAGLE-2](https://arxiv.org/abs/2406.16858) that builds trees instead of beams as drafts, but using a smaller model as drafter instead of EAGLE-1's custom drafter. 

Currently inference is very slow because [MLX matmul kernels for small batch sizes need further optimization](https://github.com/ml-explore/mlx/discussions/1593).

The RoPE kernel had to be modified to support tree mask attention, which caused additional slowdowns, but slow in comparison.