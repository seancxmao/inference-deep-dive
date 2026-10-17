# vLLM on Apple silicon CPU

## Background

### Apple silicon & M3 Pro
Apple silicon is a series of system on a chip (SoC) and system in a package (SiP) processors designed by Apple Inc., mainly using the ARM architecture.

The M3 Pro is a more powerful version of the M3, with five or six performance cores, six efficiency cores, 14 to 18 GPU cores, 16 Neural Engine cores, up to 36 GB unified RAM with 150 GB/s memory bandwidth, and 48% more transistors. It is used in the 14- and 16-inch MacBook Pro and operates at a clock frequency of up to about 4.1 GHz. Apple claims the CPU performance is 30 percent faster than the M1 Pro and the GPU is 40 percent faster than the M1 Pro.

### vLLM Apple silicon Support

vLLM官网如是说：

* vLLM has experimental support for macOS with Apple Silicon. For now, users must build from source to natively run on macOS. 
* Currently, there are no pre-built Apple silicon CPU wheels.
* Currently the CPU implementation for macOS supports FP32 and FP16 datatypes.
* For GPU-accelerated inference on Apple Silicon using Metal, check out vllm-metal, a community-maintained hardware plugin that uses MLX as the compute backend.

vLLM 官方提供两种从源码安装的方式：

* Python-only Build (without compilation)
* Full Build (with compilation)

对于Linux x86_64平台，Python-only Build通常可以正常工作；而对于 macOS Apple Silicon（M1/M2/M3/M4）平台，官方文档明确指出：Currently, there are no pre-built Apple silicon CPU wheels. 因此，Apple Silicon 用户目前只能采用 Full Build 方案。

Python-only Build的本质是，核心原生扩展（native extensions）已经由官方预先编译并打包为wheel文件，安装过程只需要处理Python层代码，而不需要重新编译底层运行时组件。Python-only Build的主要价值就是提高开发迭代效率，避免每次修改Python代码都重新编译整个Native Runtime，因为Full Build的成本很高。

## Build from source

Create a new Python environment. 默认环境名称是`.venv`，这里指定`vllm-dev`，并且放在`venvs`目录统一管理。

```
cd venvs
uv venv vllm-dev --python 3.12 --seed --managed-python
source vllm-dev/bin/activate
```

在projects目录统一管理项目：

```
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

```
Using Python 3.12.13 environment at: /.../venvs/vllm-dev
Resolved 149 packages in 25.27s
      Built vllm @ file:///.../projects/vllm
Prepared 1 package in 1m 09s
Installed 1 package in 2ms
 + vllm==0.32.0rc1.dev323+gce49c47cb.cpu (from file:///.../projects/vllm)
```

## Serve

### 使用quickstart示例启动

```
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

报错，启动失败。关键的日志如下：

```
(APIServer pid=41960) INFO 10-17 16:39:25 [api_utils.py:418]
...
(APIServer pid=41960) INFO 10-17 16:39:25 [api_utils.py:357] non-default args: {'model_tag': 'Qwen/Qwen2.5-1.5B-Instruct', 'model': 'Qwen/Qwen2.5-1.5B-Instruct'}
...
(EngineCore pid=42045) INFO 10-17 16:39:33 [core.py:133] Initializing a V1 LLM engine (v0.32.0rc1.dev323+gce49c47cb) with config: model='Qwen/Qwen2.5-1.5B-Instruct', ...
...
(VllmWorker-0 pid=42078) ERROR 10-17 16:39:37 [multiproc_executor.py:953] WorkerProc failed to start.
...
(VllmWorker-0 pid=42078) ERROR 10-17 16:39:37 [multiproc_executor.py:953] ValueError: Available memory on node 0 (6.77/18.0 GiB) on startup is less than desired CPU memory utilization (0.92, 16.56 GiB). On the CPU backend, the `--gpu-memory-utilization` flag controls the fraction of CPU memory reserved (despite its name). To resolve: decrease `--gpu-memory-utilization` (e.g. `--gpu-memory-utilization 0.5`) or reduce CPU memory used by other processes.
...
(EngineCore pid=42045) ERROR 10-17 16:39:37 [core.py:1543] EngineCore failed to start.
...
(EngineCore pid=42045) ERROR 10-17 16:39:37 [core.py:1543] Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
```

启动过程涉及几个进程：APIServer、EngineCore和Worker。Worker进程启动失败，导致整个启动失败。gpu-memory-utilization默认值是0.92，所以`18.0*0.92=16.56`，而Available memory只有6.77了，所以启动失败。虽然参数gpu-memory-utilization名字里是gpu，但是在CPU backend的情况，指的是预留的CPU memory。应当降低预留内存，使其小于Available memory（比如当前是6.77）。

### 降低CPU Memory

预留CPU内存为18 * 0.2 = 3.6

```
vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.2
```

还是报错，启动失败。但是报错的情况不同了，关键的日志如下：

```
(APIServer pid=46698) INFO 10-17 16:47:44 [api_utils.py:418]
...
(APIServer pid=46698) INFO 10-17 16:47:44 [api_utils.py:357] non-default args: {'model_tag': 'Qwen/Qwen2.5-1.5B-Instruct', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'gpu_memory_utilization': 0.2}
...
(EngineCore pid=46813) INFO 10-17 16:47:55 [core.py:133] Initializing a V1 LLM engine (v0.32.0rc1.dev323+gce49c47cb) with config: model='Qwen/Qwen2.5-1.5B-Instruct', ...
(VllmWorker-0 pid=46846) WARNING 10-17 16:47:58 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
(Worker pid=46846) INFO 10-17 16:47:58 [parallel_state.py:1911] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm_dist_37f8fc6842e247d4a879194c2a84e3a5 backend=gloo
...
(Worker pid=46846) INFO 10-17 16:47:58 [cpu_model_runner.py:68] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
...
(Worker pid=46846) INFO 10-17 16:47:58 [base_loader.py:82] Initializing model took 0.102745 seconds
...
(Worker pid=46846) INFO 10-17 16:48:07 [default_loader.py:501] Loading weights took 2.38 seconds
...
(Worker pid=46846) INFO 10-17 16:48:07 [cpu_model_runner.py:87] Warming up model for the compilation...
...
(Worker pid=46846) INFO 10-17 16:48:43 [cpu_model_runner.py:91] Warming up done.
(Worker pid=46846) ERROR 10-17 16:48:43 [multiproc_executor.py:1063] WorkerProc hit an exception.
...
(Worker pid=46846) ERROR 10-17 16:48:43 [multiproc_executor.py:1063] ValueError: The worker already uses 4.05 GiB (model weights and runtime), more than the 3.6 GiB of CPU memory reserved for it on node 0, leaving none for the KV cache. On the CPU backend, the `--gpu-memory-utilization` flag controls the fraction of CPU memory reserved (despite its name). To resolve: increase `--gpu-memory-utilization` to more than 0.23, or set the KV cache size explicitly with `--kv-cache-memory-bytes` or VLLM_CPU_KVCACHE_SPACE.
...
(EngineCore pid=46813) ERROR 10-17 16:48:43 [core.py:1543] EngineCore failed to start.
...
(EngineCore pid=46813) ERROR 10-17 16:48:43 [core.py:1543] RuntimeError: Worker failed with error 'The worker already uses 4.05 GiB (model weights and runtime), more than the 3.6 GiB of CPU memory reserved for it on node 0, leaving none for the KV cache. On the CPU backend, the `--gpu-memory-utilization` flag controls the fraction of CPU memory reserved (despite its name). To resolve: increase `--gpu-memory-utilization` to more than 0.23, or set the KV cache size explicitly with `--kv-cache-memory-bytes` or VLLM_CPU_KVCACHE_SPACE.', please check the stack trace above for the root cause
```

之前的错误发生在启动前的内存检查。这里Worker进程已经启动，加载模型权重也已经成功，开始kv cache allocation，但是发现内存不够，所以失败了。从日志看，预留的CPU内存（3.6），但是worker已经使用了4.05 GiB (model weights and runtime)，没有空余给kv cache了。

vLLM CPU模式下整个系统内存组成：

* 非vLLM
    * 操作系统
    * 其他程序
* vLLM
    * 模型权重
    * KV Cache
    * 运行时开销

Qwen2.5-1.5B的参数量1.54B，如果是默认FP16/BF16，内存占用1.54B * 2 bytes=3.08 GiB，再加上其他一些开销，实际模型加载后通常会更多，以至于超过预留的3.6了。有一个解决办法是预留更多的内存。

### 提高CPU Memory

预留更多的内存，启动成功。

```
vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.3
```

memory: 18 * 0.3 = 5.4

获得完整的日志，值得细读：

```
W1017 17:01:03.045000 54311 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:01:03.045000 54311 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:01:03.045000 54311 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
W1017 17:01:06.536000 54311 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]        █     █     █▄   ▄█
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.32.0rc1.dev323+gce49c47cb
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-1.5B-Instruct
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:418]
(APIServer pid=54311) INFO 10-17 17:01:06 [api_utils.py:357] non-default args: {'model_tag': 'Qwen/Qwen2.5-1.5B-Instruct', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'gpu_memory_utilization': 0.3}
(APIServer pid=54311) INFO 10-17 17:01:08 [model.py:733] Resolved architecture: Qwen2ForCausalLM
(APIServer pid=54311) INFO 10-17 17:01:08 [model.py:2152] Using max model len 32768
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
(APIServer pid=54311) INFO 10-17 17:01:11 [kernel.py:476] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native'])
(APIServer pid=54311) WARNING 10-17 17:01:11 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
Fetching 9 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 4243.34it/s]
Download complete:                                                                                                                                                                            |  0.00B
Reconstruction complete: |                                                                                                                                                           |  0.00B /  0.00B
W1017 17:01:12.869000 54407 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:01:12.869000 54407 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:01:12.870000 54407 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
(EngineCore pid=54407) INFO 10-17 17:01:15 [core.py:133] Initializing a V1 LLM engine (v0.32.0rc1.dev323+gce49c47cb) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=main, tokenizer_revision=main, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, quantization_config=None, enforce_eager=False, aux_output_config=AuxOutputConfig(enable_return_routed_experts=False, max_bytes=None), kv_cache_dtype=auto, device_config=cpu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, custom_histogram_buckets=None, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.DYNAMO_TRACE_ONCE: 2>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none', '+gelu', '+gelu_tanh', '+gelu_and_mul'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'cudagraph_decoder_replay': True, 'decoder_replay_cudagraph_capture_sizes': [], 'decoder_replay_trim_threshold': 768, 'compile_sizes': None, 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'dce': True, 'size_asserts': False, 'nan_asserts': False, 'epilogue_fusion': True, 'cpp.dynamic_threads': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native']), enable_flashinfer_autotune=True, enable_rocm_segmented_attn_autotune=False, enable_cutedsl_warmup=True, enable_jit_warmup=True, monokernel=False, moe_backend='auto', sparse_indexer_topk_backend='auto', linear_backend='auto', linear_backend_per_quant=None)
(EngineCore pid=54407) INFO 10-17 17:01:15 [multiproc_executor.py:153] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=192.168.3.57 (local), world_size=1, local_world_size=1
(EngineCore pid=54407) INFO 10-17 17:01:15 [ompmultiprocessing.py:185] OpenMP thread binding info:
(EngineCore pid=54407) INFO 10-17 17:01:15 [ompmultiprocessing.py:185] 	VLLM_CPU_OMP_THREADS_BIND='auto', auto_setup=True, skip_setup=False
(EngineCore pid=54407) INFO 10-17 17:01:15 [ompmultiprocessing.py:185] 	local_world_size=1, reserve_cpu_num=1
(EngineCore pid=54407) INFO 10-17 17:01:15 [ompmultiprocessing.py:185] 	local_rank=0, core ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(EngineCore pid=54407) INFO 10-17 17:01:15 [ompmultiprocessing.py:185] 	reserved_cpus=[10]
W1017 17:01:15.835000 54440 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:01:15.836000 54440 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:01:15.836000 54440 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm/vllm/triton_utils/dispatcher.py:138: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if hasattr(value, "__dict__"):
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm/vllm/triton_utils/dispatcher.py:139: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  wrapper = lookup(vars(value).get("kernel"))
W1017 17:01:18.673000 54440 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
(VllmWorker-0 pid=54440) WARNING 10-17 17:01:18 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
(Worker pid=54440) INFO 10-17 17:01:18 [parallel_state.py:1911] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm_dist_9d446c485bb54a0e83c692efff6b231f backend=gloo
(Worker pid=54440) INFO 10-17 17:01:18 [parallel_state.py:2355] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, ETP rank 0, EP rank N/A, EPLB rank N/A
(Worker pid=54440) INFO 10-17 17:01:18 [cpu_model_runner.py:68] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
(Worker pid=54440) WARNING 10-17 17:01:19 [compilation.py:1355] Op 'gelu' not present in model, enabling with '+gelu' has no effect
(Worker pid=54440) WARNING 10-17 17:01:19 [compilation.py:1355] Op 'gelu_tanh' not present in model, enabling with '+gelu_tanh' has no effect
(Worker pid=54440) WARNING 10-17 17:01:19 [compilation.py:1355] Op 'gelu_and_mul' not present in model, enabling with '+gelu_and_mul' has no effect
(Worker pid=54440) INFO 10-17 17:01:19 [base_loader.py:82] Initializing model took 0.113318 seconds
(Worker pid=54440) INFO 10-17 17:01:20 [weight_utils.py:1112] Filesystem type for checkpoints: unknown. Checkpoint size: 2.88 GiB. Available RAM: 5.79 GiB.
(Worker pid=54440) INFO 10-17 17:01:20 [weight_utils.py:1135] Auto-prefetch is disabled because the filesystem (unknown) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.98s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.98s/it]

(Worker pid=54440) INFO 10-17 17:01:22 [default_loader.py:501] Loading weights took 2.01 seconds
(Worker pid=54440) WARNING 10-17 17:01:22 [torch_utils.py:283] OMP_NUM_THREADS=10 is set; leaving Torch threads at 10 for serving. Multi-threaded torch CPU ops during serving can degrade performance through spin-wait contention and cgroup CPU-quota throttling.
(EngineCore pid=54407) INFO 10-17 17:01:22 [torch_utils.py:295] Reducing Torch threads from 5 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=54407) INFO 10-17 17:01:22 [utils.py:348] Using LBHNC KV cache layout.
(Worker pid=54440) INFO 10-17 17:01:22 [cpu_model_runner.py:87] Warming up model for the compilation...
(Worker pid=54440) WARNING 10-17 17:01:23 [decorators.py:340] [backbone] Compiling model again due to a load failure from /Users/maochenxiao/.cache/vllm/torch_compile_cache/torch_aot_compile/da1b47f48daf9bfcc6f993fb7dcab902d84935b402c3741caadb62a3f0bbb3ef/rank_0_0/model, reason: 'function' object has no attribute 'finalize_loading'
(Worker pid=54440) INFO 10-17 17:01:34 [decorators.py:752] [backbone] saved AOT compiled function to /Users/maochenxiao/.cache/vllm/torch_compile_cache/torch_aot_compile/da1b47f48daf9bfcc6f993fb7dcab902d84935b402c3741caadb62a3f0bbb3ef/rank_0_0/model
(Worker pid=54440) INFO 10-17 17:01:58 [monitor.py:91] [backbone] Initial profiling/warmup run took 24.10 s
(Worker pid=54440) INFO 10-17 17:01:59 [cpu_model_runner.py:91] Warming up done.
(Worker pid=54440) INFO 10-17 17:01:59 [cpu_worker.py:274] Auto set (1.35/18.0) GiB for KV cache on node 0, with 5.4 GiB requested memory for the worker. 4.05 GiB memory was consumed by non-kv usages.
(EngineCore pid=54407) INFO 10-17 17:01:59 [kv_cache_utils.py:1101] CPU KV cache size: 50,304 tokens, Maximum concurrency for 32,768 tokens per request: 1.54x
(EngineCore pid=54407) INFO 10-17 17:01:59 [core.py:402] init engine (profile, create kv cache, warmup model) took 36.90 s
W1017 17:02:00.682000 54407 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
Fetching 9 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 5465.29it/s]
Reconstruction complete: |                                                                                                                                                           |  0.00B /  0.00B
Download complete:                                                                                                                                                                            |  0.00B
(EngineCore pid=54407) WARNING 10-17 17:02:00 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
(EngineCore pid=54407) WARNING 10-17 17:02:03 [vllm.py:1771] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=54407) INFO 10-17 17:02:03 [kernel.py:476] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native'])
(APIServer pid=54311) INFO 10-17 17:02:03 [entry.py:136] Supported tasks: ['generate']
(APIServer pid=54311) INFO 10-17 17:02:03 [factories.py:89] Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.
(APIServer pid=54311) INFO 10-17 17:02:03 [hf.py:647] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=54311) WARNING 10-17 17:02:03 [model.py:1893] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=54311) INFO 10-17 17:02:03 [entry.py:140] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:186] Available routes are:
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /docs, Methods: GET, HEAD
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /redoc, Methods: GET, HEAD
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /load, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /version, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /kv_event_sources, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /health, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /metrics, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /tokenize, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /detokenize, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/models, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/systemone, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/decisions, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /ping, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /ping, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /invocations, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/chat/completions, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/responses, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/completions, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/messages, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /generative_scoring, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/messages/render, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/completions/render, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/responses/render, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/chat/completions/derender, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /v1/completions/derender, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /inference/v1/generate, Methods: POST
(APIServer pid=54311) INFO 10-17 17:02:03 [launcher.py:195] Route: /inference/v1/abort_requests, Methods: POST
INFO:     Started server process [54311]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

最终，三个进程APIServer、EngineCore、Worker，都启动成功。

### 降低模型规模

除了多预留内存，还有一个方案是降低内存的使用，比如使用更小的模型。使用Qwen/Qwen3-0.6B：

```
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.2
```

报错，启动失败。关键日志如下：

```
(EngineCore pid=58336) ERROR 10-17 17:08:27 [core.py:1543] ValueError: To serve at least one request with the model's max seq len (40960), (4.38 GiB KV cache is needed, which is larger than the available KV cache memory (1.69 GiB). Based on the available memory, the estimated maximum model length is 15744. Try increasing `gpu_memory_utilization` (which also controls CPU memory on the CPU backend) or decreasing `max_model_len` when initializing the engine. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.
```

为什么0.6B的模型，需要4.38GiB的KV cache呢？

对于标准Transformer：

* 如果context length是40960，意味着要为40960个token缓存K和V
* K和V意味着token数量的2倍
* K和V分别是用1024长的向量表示（Qwen3-0.6B使用类似的数量级）
* 向量中的每个数值如果是BF16，则占用2个字节
* 并且，每个attention层都要缓存。（Qwen3-0.6B有28个transformer层，也即28个attention层）

所以KV cache的计算公式是：

```
context_length      // 40960
* K_and_V           // 2: K and V
* embedding_size    // 1024: vector length of K or V
* dtype_size        // 2: BF16, how many bytes for each number in the vector
* number_of_layers  // 28
```

(40960 * 2 * 1024 * 2 * 28) / (1024 * 1024 * 1024) = 4.38 GiB

Qwen3并不是传统MHA，而是GQA(Grouped Query Attention)：

```
KV Cache
=
context_length  // 40960 
* K_and_V       // 2
* num_kv_heads  // 8
* head_dim      // 128
* dtype_size    // 2
* num_layers    // 28
```

(40960 * 2 * 8 * 128 * 2 * 28) / (1024 * 1024 * 1024) = 4.38 GiB

### 降低上下文长度

使用报错信息中推荐的长度15744.

```
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.2 --max_model_len 15744
```

启动成功。保留一份完整日志，供细读：

```
W1017 17:15:03.197000 62438 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:15:03.198000 62438 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:15:03.198000 62438 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
W1017 17:15:06.724000 62438 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]        █     █     █▄   ▄█
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.32.0rc1.dev323+gce49c47cb
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]   █▄█▀ █     █     █     █  model   Qwen/Qwen3-0.6B
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:418]
(APIServer pid=62438) INFO 10-17 17:15:06 [api_utils.py:357] non-default args: {'model_tag': 'Qwen/Qwen3-0.6B', 'max_model_len': 15744, 'gpu_memory_utilization': 0.2}
(APIServer pid=62438) INFO 10-17 17:15:09 [model.py:733] Resolved architecture: Qwen3ForCausalLM
(APIServer pid=62438) INFO 10-17 17:15:09 [model.py:2152] Using max model len 15744
(APIServer pid=62438) INFO 10-17 17:15:12 [kernel.py:476] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native'])
(APIServer pid=62438) WARNING 10-17 17:15:12 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
Fetching 9 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 3079.02it/s]
Download complete:                                                                                                                                                                            |  0.00B
Reconstruction complete: |                                                                                                                                                           |  0.00B /  0.00B
W1017 17:15:14.116000 62543 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:15:14.116000 62543 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:15:14.116000 62543 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
(EngineCore pid=62543) INFO 10-17 17:15:16 [core.py:133] Initializing a V1 LLM engine (v0.32.0rc1.dev323+gce49c47cb) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=main, tokenizer_revision=main, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=15744, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, quantization_config=None, enforce_eager=False, aux_output_config=AuxOutputConfig(enable_return_routed_experts=False, max_bytes=None), kv_cache_dtype=auto, device_config=cpu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, per_request_spec_decode_metrics='none', kv_cache_metrics=False, kv_cache_metrics_sample=0.01, custom_histogram_buckets=None, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.DYNAMO_TRACE_ONCE: 2>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none', '+gelu', '+gelu_tanh', '+gelu_and_mul'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'cudagraph_decoder_replay': True, 'decoder_replay_cudagraph_capture_sizes': [], 'decoder_replay_trim_threshold': 768, 'compile_sizes': None, 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'dce': True, 'size_asserts': False, 'nan_asserts': False, 'epilogue_fusion': True, 'cpp.dynamic_threads': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native']), enable_flashinfer_autotune=True, enable_rocm_segmented_attn_autotune=False, enable_cutedsl_warmup=True, enable_jit_warmup=True, monokernel=False, moe_backend='auto', sparse_indexer_topk_backend='auto', linear_backend='auto', linear_backend_per_quant=None)
(EngineCore pid=62543) INFO 10-17 17:15:16 [multiproc_executor.py:153] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=192.168.3.57 (local), world_size=1, local_world_size=1
(EngineCore pid=62543) INFO 10-17 17:15:16 [ompmultiprocessing.py:185] OpenMP thread binding info:
(EngineCore pid=62543) INFO 10-17 17:15:16 [ompmultiprocessing.py:185] 	VLLM_CPU_OMP_THREADS_BIND='auto', auto_setup=True, skip_setup=False
(EngineCore pid=62543) INFO 10-17 17:15:16 [ompmultiprocessing.py:185] 	local_world_size=1, reserve_cpu_num=1
(EngineCore pid=62543) INFO 10-17 17:15:16 [ompmultiprocessing.py:185] 	local_rank=0, core ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(EngineCore pid=62543) INFO 10-17 17:15:16 [ompmultiprocessing.py:185] 	reserved_cpus=[10]
W1017 17:15:17.036000 62576 torch/_opaque_base.py:6] torch._opaque_base is deprecated, use torch._custom_class_base instead
W1017 17:15:17.036000 62576 torch/_library/opaque_object.py:288] register_opaque_type is deprecated, use register_custom_class instead
W1017 17:15:17.036000 62576 torch/_library/opaque_object.py:211] typ='value' is deprecated, use typ='constant' instead
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm/vllm/triton_utils/dispatcher.py:138: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if hasattr(value, "__dict__"):
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm/vllm/triton_utils/dispatcher.py:139: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  wrapper = lookup(vars(value).get("kernel"))
W1017 17:15:19.825000 62576 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
(VllmWorker-0 pid=62576) WARNING 10-17 17:15:19 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
(Worker pid=62576) INFO 10-17 17:15:19 [parallel_state.py:1911] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm_dist_a2935531865846158ec67a5b3c21612a backend=gloo
(Worker pid=62576) INFO 10-17 17:15:20 [parallel_state.py:2355] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, ETP rank 0, EP rank N/A, EPLB rank N/A
(Worker pid=62576) INFO 10-17 17:15:20 [cpu_model_runner.py:68] Starting to load model Qwen/Qwen3-0.6B...
(Worker pid=62576) WARNING 10-17 17:15:20 [compilation.py:1355] Op 'gelu' not present in model, enabling with '+gelu' has no effect
(Worker pid=62576) WARNING 10-17 17:15:20 [compilation.py:1355] Op 'gelu_tanh' not present in model, enabling with '+gelu_tanh' has no effect
(Worker pid=62576) WARNING 10-17 17:15:20 [compilation.py:1355] Op 'gelu_and_mul' not present in model, enabling with '+gelu_and_mul' has no effect
(Worker pid=62576) INFO 10-17 17:15:20 [base_loader.py:82] Initializing model took 0.100302 seconds
(Worker pid=62576) INFO 10-17 17:15:22 [weight_utils.py:1112] Filesystem type for checkpoints: unknown. Checkpoint size: 1.40 GiB. Available RAM: 6.31 GiB.
(Worker pid=62576) INFO 10-17 17:15:22 [weight_utils.py:1135] Auto-prefetch is disabled because the filesystem (unknown) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.14it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.14it/s]

(Worker pid=62576) INFO 10-17 17:15:22 [default_loader.py:501] Loading weights took 0.25 seconds
(Worker pid=62576) WARNING 10-17 17:15:22 [torch_utils.py:283] OMP_NUM_THREADS=10 is set; leaving Torch threads at 10 for serving. Multi-threaded torch CPU ops during serving can degrade performance through spin-wait contention and cgroup CPU-quota throttling.
(EngineCore pid=62543) INFO 10-17 17:15:22 [torch_utils.py:295] Reducing Torch threads from 5 to 1 for serving. Set OMP_NUM_THREADS in the external environment to override.
(EngineCore pid=62543) INFO 10-17 17:15:22 [utils.py:348] Using LBHNC KV cache layout.
(Worker pid=62576) INFO 10-17 17:15:22 [cpu_model_runner.py:87] Warming up model for the compilation...
(Worker pid=62576) WARNING 10-17 17:15:23 [decorators.py:340] [backbone] Compiling model again due to a load failure from /Users/maochenxiao/.cache/vllm/torch_compile_cache/torch_aot_compile/29ba82301d9a3006ef49a0f955cadc7bf2acbcc97f1fae93889615cafc81e89f/rank_0_0/model, reason: 'function' object has no attribute 'finalize_loading'
(Worker pid=62576) INFO 10-17 17:15:36 [decorators.py:752] [backbone] saved AOT compiled function to /Users/maochenxiao/.cache/vllm/torch_compile_cache/torch_aot_compile/29ba82301d9a3006ef49a0f955cadc7bf2acbcc97f1fae93889615cafc81e89f/rank_0_0/model
(Worker pid=62576) INFO 10-17 17:15:44 [monitor.py:91] [backbone] Initial profiling/warmup run took 8.74 s
(Worker pid=62576) INFO 10-17 17:15:45 [cpu_model_runner.py:91] Warming up done.
(Worker pid=62576) INFO 10-17 17:15:45 [cpu_worker.py:274] Auto set (1.71/18.0) GiB for KV cache on node 0, with 3.6 GiB requested memory for the worker. 1.89 GiB memory was consumed by non-kv usages.
(EngineCore pid=62543) INFO 10-17 17:15:45 [kv_cache_utils.py:1101] CPU KV cache size: 16,000 tokens, Maximum concurrency for 15,744 tokens per request: 1.02x
(EngineCore pid=62543) INFO 10-17 17:15:45 [core.py:402] init engine (profile, create kv cache, warmup model) took 23.25 s
W1017 17:15:46.748000 62543 torch/_library/triton.py:53] Triton not available, find_triton_kernels = []
Fetching 9 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 5360.51it/s]
Reconstruction complete: |                                                                                                                                                           |  0.00B /  0.00B
Download complete:                                                                                                                                                                            |  0.00B
(EngineCore pid=62543) WARNING 10-17 17:15:47 [vllm.py:726] Model Runner V2 requires Triton; using the V1 model runner instead.
(EngineCore pid=62543) WARNING 10-17 17:15:47 [vllm.py:1771] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=62543) INFO 10-17 17:15:47 [kernel.py:476] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native'])
(APIServer pid=62438) INFO 10-17 17:15:47 [entry.py:136] Supported tasks: ['generate']
(APIServer pid=62438) INFO 10-17 17:15:47 [factories.py:89] Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.
(APIServer pid=62438) INFO 10-17 17:15:47 [hf.py:647] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=62438) WARNING 10-17 17:15:47 [model.py:1893] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=62438) INFO 10-17 17:15:47 [entry.py:140] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:186] Available routes are:
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /docs, Methods: GET, HEAD
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /redoc, Methods: GET, HEAD
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /load, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /version, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /kv_event_sources, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /health, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /metrics, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /tokenize, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /detokenize, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/models, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/systemone, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/decisions, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /ping, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /ping, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /invocations, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/chat/completions, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/responses, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/completions, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/messages, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /generative_scoring, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/messages/render, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/completions/render, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/responses/render, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/chat/completions/derender, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /v1/completions/derender, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /inference/v1/generate, Methods: POST
(APIServer pid=62438) INFO 10-17 17:15:47 [launcher.py:195] Route: /inference/v1/abort_requests, Methods: POST
INFO:     Started server process [62438]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

如果用更小的值，自然也能成功，比如4096:

```
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.2 --max_model_len 4096
```

### 无参数启动

最后发现，不指定参数启动（`vllm serve`）时，用的就是`Qwen/Qwen3-0.6B`，相当于

```
vllm serve Qwen/Qwen3-0.6B
```

这和quickstart示例（`vllm serve Qwen/Qwen2.5-1.5B-Instruct`）一样，会报错。解决方法上面已经给出。

## 其他问题

因为在多个地方安装vllm，可能导致混用而出错。比如，先用brew安装了vllm-metal，再从源码安装vllm，启动成功，但是报错：

```
vllm on  main via △ via 🐍 v3.12.13 (vllm-dev)
❯ vllm serve Qwen/Qwen2.5-1.5B-Instruct
...
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031] Error saving model info cache.
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031] Traceback (most recent call last):
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]   File "/opt/homebrew/Cellar/vllm-metal/0.31.0/libexec/lib/python3.12/site-packages/vllm/model_executor/models/registry.py", line 1023, in _save_modelinfo_to_cache
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]     "modelinfo": asdict(mi),
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]                  ^^^^^^^^^^
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]   File "/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/lib/python3.12/dataclasses.py", line 1329, in asdict
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]     return _asdict_inner(obj, dict_factory)
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]   File "/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/lib/python3.12/dataclasses.py", line 1339, in _asdict_inner
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]     f.name: _asdict_inner(getattr(obj, f.name), dict)
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031]                           ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=4497) ERROR 10-17 10:57:50 [registry.py:1031] AttributeError: '_ModelInfo' object has no attribute 'supports_mamba_prefix_caching'
...
INFO:     Started server process [4497]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

本来应该用`vllm-dev`的Python包，却使用了`/opt/homebrew/Cellar/vllm-metal`的Python包。可通过以下命令诊断问题：

```
❯ echo "$VIRTUAL_ENV"

❯ which vllm
/opt/homebrew/bin/vllm

❯ which -a vllm
/.../venvs/vllm-dev/bin/vllm
/opt/homebrew/bin/vllm

❯ head -n 1 "$(which vllm)"
#!/opt/homebrew/Cellar/vllm-metal/0.31.0/libexec/bin/python3.12

❯ echo "$PATH"
/.../venvs/vllm-dev/bin:...:/opt/homebrew/bin:...

# 确认实际导入的Python模块
python -c 'import sys, vllm; print(sys.executable); print(vllm.__file__)'
```

明明PATH中的顺序是对的，却使用了错误的命令，怀疑是缓存的问题，通过rehash解决。rehash的作用是刷新命令查找缓存，让zsh重新扫描PATH中的可执行命令。

```
❯ rehash
```

## Summary

启动过程中碰到3类问题：

* 系统可用内存不够，小于vLLM预留内存 -> 增大系统可用内存，或者降低vLLM预留内存
* vLLM拿到了预留内存，但是模型权重占用太多，KV cache不够 -> 调大vLLM预留内存，或者使用更小的模型
* 使用小的模型，但是上下文很长，权重加载成功，KV cache不够 -> 调小上下文长度

几点收获：

* 这三类问题都是内存相关的，可以粗略一窥vLLM的内存格局。
* 就内存而言，Training时代，参数量决定成本；Inference时代，KV Cache决定成本。
* 启动成功后，能够看到本地部署的进程模型，APIServer、EngineCore、Worker。
* 错误日志的可读性在改进。

## References

* https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
* https://en.wikipedia.org/wiki/Apple_silicon
* https://huggingface.co/Qwen/Qwen2.5-1.5B
* https://huggingface.co/Qwen/Qwen3-0.6B
* gpu_memory_utilization defaults to 0.92 in the source code. https://github.com/vllm-project/vllm/blob/releases/v0.32.0/vllm/config/cache.py#L102
* BF16, bfloat16, brain floating point. https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
