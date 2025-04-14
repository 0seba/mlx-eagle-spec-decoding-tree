MLX port of [EAGLE-2](https://arxiv.org/abs/2406.16858) that builds trees instead of beams as drafts, but using a smaller model as drafter instead of EAGLE-1's custom drafter. 

Using custom batched QMM kernel. In my tests FOR M1 ARCHITECTURE the speed of batch size 8 was similar to batch size 1 on small matrices and about 40% slower on big matrices (2048x8192), increasing batch size by `8` often leads to an almost linear increase in time.

I haven't performed a lot of tests on >=M3/A17, but the time should be the same for all matrix sizes and bigger batch sizes 8, 16, maybe 24 and 32, more than that it's likely it causes another wave in the kernel and the runtime duplicates.

The RoPE kernel had to be modified to support tree mask attention, which caused additional slowdowns, but slow in comparison.

**IMPORTANT: Current code only works with Llama3/Qwen2 architectures, both target and draft model have to be 4 bit quantized and uses greedy decoding, no sampling yet**

Testing on M1 Air 8GB with the following command `python mlx_eagle.py --prompt "How many positive whole-number divisors does 196 have?" --depth 3 --verify-num-tokens 7 --max-tokens 300 -N 300` I get around 22 tokens per second, with 4.5 tokens generated per single evaluation of the target model, compared to 18 tokens per second using `mlx_lm.generate` with `--num-draft-tokens 2` for the same target `Qwen2.5 7B` and draft models `Qwen2.5 0.5B`.

The parameters are:
* `--draft-model`
* `--target-model`
* `--prompt`
* `--depth`: Depth of the draft tree explored (can be `0`, in that case `--verify-num-tokens` has to be `7`)
* `--verify-num-tokens`: Number of nodes of the draft tree extracted to be verified by the target model (not the whole generated tree is verified). Has to be multiple of `8` minus `1` (`7, 15, 23, 31` recommended to try)
* `--max-tokens`: maximum amount of tokens to generate
* `--max-generation-steps`: Each step can generate more than 1 token
