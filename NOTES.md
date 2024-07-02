# Implementation of GPT2 (124M)
* https://github.com/openai/gpt-2 has open-weights but in tf
    * tensorflow -> PyTorch tensors
    * attrs in this repo mimic https://huggingface.co/openai-community/gpt2/ implementation https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py 
* parameters count within the paper incorrect: "117M"=124M
* ~1h $10 to train now on 

### GPT2 vs Original Transformer Paper
1. decoder only (no cross-attention)
2. LayerNorm placement
    * moved from after MHA/FF to before
    * moved from inside residual stream to inside the residual block (clean residual pathway)
    * added before final linear classifier

### IMPL
Refer to both GPT2 (less clear) and GPT3 (more clear) papers
* weight sharing: wte, lm_head
    * performance driven but also saves lots of parameters
* initialization: refer to openai gpt-2 source code
    * better to be a function of feature count (magic number of 0.2 ~= Xavier initialization scheme)
    * residual stream: standard deviation grows inside -> scale down by sqrt(n) for each layer (so 2x for each block)

### GPU: ie A100 80GB SXM
* Many workloads are memory bound!!! (ie 60% utilization of hardware is high) 
    - floating point important for training (to match normal distributions of activations/weights)
    - INT8 (uniform spacing) can only be used during inference
* Tensor Cores (on SMs): specialized processing units within GPU that can execute matrix multiplication (GEMM) and accumulation operations very efficiently
    - accessed through CUDA instructions like HMMA.1688.F16 (FP16 computation with FP32 accumulation), IMMA.8816.S8.S8.SAT (INT8 GEMM), WMMA
        - enable mixed-precision computations
    - ie A100: 4 third-gen per SM -> 432 per GPU
        - third-gen: 256 FP16/FP32 FMA operations per clock

## Step-by-Step:
* Setup inference
    1. define GPTConfig data, define GPT nn.Module model structure 
    2. define modules needed by transformer structure
    3. implement GPT core functionality: forward, generate
    4. implement GPT.from_pretrained to load hf model weights and sample
* Setup training
    1. create Data Loader and setup basic optimization skeleton: load next batch -> move to device -> optimizer.zero_grad() -> forward pass -> backward pass-> optimizer.step() to update
        - choose batch size to be largest (multiple of the GPU optimized units for dtype) without running out of RAM --- hardward-aware!
    2. add initialization scaling 
    3. optimize GPU utilization
        - reduced precision: FP32 -> TF32 -> BF16 inputs (same exponent just reduced mantissa whereas FP16 reduces range and need gradient scalers, additional source of state and complexity)
            - FP32 baseline: 16.3K tokens/s throughput, ~1s per iteration (B=16, T=1024)
            - TF32 `torch.set_float32_matmul_precision("high")`: 49K tokens/s throughput (not 8x b/c memory bound and still moving around float 32), 333ms
            - BF16 `with torch.autocast(device_type=device, dtype=torch.bfloat16)` for activations: 54.7K tokens/s throughput, 300ms
        - JIT compilation with `model = torch.compile(model)` (kernel fusion for ie GELU to reduce GPU reads/writes, optimized code to reduce Python overhead): 126K tokens/s, 129.5ms
        - Flash Attention `F.scaled_dot_product_attention`: 170K tokens/s, 97ms
        - fix "ugly" numbers: kernels are inefficient for leftovers that do not fill a tile so padding can result in speedup
            - ie increase vocab size from 50257 to 50304: 176K tokens/s, 93ms
    4. configure optimizer based on GPT3 paper: : 182K tokens/s, 90ms
        - hyperparameters, regularize with weight decay (identify parameters that should be weight decayed and configure weight decay, `p.requires_grad and p.dim() >= 2`)
        - learning rate scheduler
    5. impl high batch size algo (does not fit in GPU memory): **gradient accumulation** over serial microsteps (choice of B purely for GPU utilization)
        - number of microsteps: total_batch_size // (B * T)
        - each microstep will 
            1. load next batch, 
            2. apply forward pass to get loss
            3. *reduce loss contribution by number of microsteps* !!!
            4. apply backprop
    6. use multi-GPUs with `torch.nn.parallel.DistributedDataParallel`: 1.5M tokens/s
        - update configs for ddp RANK, WORLD_SIZE dependencies:
            - `DataLoaderLite`
                - init: `self.current_position = B * T * self.process_rank`
                - next_batch: `self.current_position += B * T * num_processes`
            - `model = DistributedDataParallel(model, device_ids=[<local rank>])`
            - num microsteps: `total_batch_size // (B * T * world_size)`
        - update optimization for explicit gradient-synchronization on DDP-wrapped model:
            - during grad accumulation, instead of `no_sync` context manager to disable gradient synchronization across DDP processes, toggle `require_backward_grad_sync` to False directly (naughty!)
        - guard print lines to only process for "master_process"
        - add cleanup with `torch.distributed.destroy_process_group()`
        - launch with `torchrun --standalone --nproc_per_node=8 train_gpt2.py`
* Setup periodic eval during training run
    - sample validation loss to check for overfitting
    - sample generation
    - multi-gpu eval of HellaSwag accuracy (play.ipynb shows plots of log stats)
        - if training data (ie HF fineweb) has been contaminated with the eval dataset, eval curve improvements may just be from the additional training
