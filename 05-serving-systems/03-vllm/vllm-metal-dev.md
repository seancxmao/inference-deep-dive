# Build vLLM Metal

## Background

On an Apple Silicon Mac, install full Xcode with macOS SDK 26.2 or newer. Select Xcode as the active developer directory. Initial source setup builds native Metal artifacts, including for Python-only contributions.

## Build

### 总体步骤

```
git clone https://github.com/YOUR_USERNAME/vllm-metal.git
cd vllm-metal
git remote add upstream https://github.com/vllm-project/vllm-metal.git
git switch -c my-change
./install.sh
source .venv-vllm-metal/bin/activate
uv pip install -e ".[dev]"
```

### `./install.sh`

`./install.sh` creates the local environment, installs the matching vLLM core and editable plugin, and builds the native artifacts. It downloads the Metal toolchain if needed. Restart the server after editing Python files.

关键点：

* 在项目更目录创建虚拟环境`.venv-vllm-metal`
* Installing vLLM core：vllm==0.31.0+cpu (from https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl)
* Build and install vllm-metal==0.31.0
* Ensuring Metal toolchain
* Building native Metal artifacts: Building native paged-attention extension

保留完整日志，精读。

```
❯ ./install.sh
=== Creating virtual environment ===
Using CPython 3.12.13
Creating virtual environment with seed packages at: .venv-vllm-metal
 + pip==26.2.1
Activate with: source .venv-vllm-metal/bin/activate

=== Installing vLLM core ===
Wheel: vLLM 0.31.0 (prebuilt macOS arm64)
Using Python 3.12.13 environment at: .venv-vllm-metal
Resolved 149 packages in 2.35s
Installed 149 packages in 648ms
 + agent-detector==2.0.0
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.5
 + aiosignal==1.4.0
 + annotated-doc==0.0.5
 + annotated-types==0.8.0
 + anthropic==1.15.0
 + anyio==4.15.1
 + apache-tvm-ffi==0.1.14.post1
 + astor==0.8.1
 + attrs==26.1.0
 + blake3==1.0.11
 + cachetools==7.2.3
 + cbor2==6.1.5
 + certifi==2026.7.22
 + cffi==2.1.1
 + charset-normalizer==3.5.2
 + click==8.5.0
 + cloudpickle==3.1.2
 + compressed-tensors==0.17.0
 + cryptography==50.0.2
 + depyf==0.20.0
 + detect-installer==0.2.1
 + dill==0.4.1
 + dnspython==2.9.0
 + docstring-parser==0.18.0
 + einops==0.8.2
 + email-validator==2.3.0
 + fastapi==0.136.3
 + fastapi-cli==0.0.32
 + fastapi-cloud-cli==0.26.0
 + fastar==0.12.0
 + filelock==4.2.1
 + frozenlist==1.8.0
 + fsspec==2026.9.0
 + googleapis-common-protos==1.75.5
 + grpcio==1.84.0
 + h11==0.16.0
 + hf-xet==1.7.0
 + httpcore==1.0.9
 + httpcore2==2.13.1
 + httptools==0.9.0
 + httpx==0.28.1
 + httpx2==2.13.1
 + huggingface-hub==1.33.0
 + idna==3.20
 + ijson==3.6.0
 + interegular==0.3.3
 + jinja2==3.1.6
 + jiter==0.17.0
 + jmespath==1.1.0
 + jsonschema==4.26.0
 + jsonschema-specifications==2025.9.1
 + lark==1.2.2
 + llguidance==1.7.6
 + llvmlite==0.47.0
 + lm-format-enforcer==0.11.3
 + loguru==0.7.3
 + markdown-it-py==4.2.0
 + markupsafe==3.0.4
 + mcp==2.3.0
 + mcp-types==2.3.0
 + mdurl==0.1.2
 + mistral-common==1.12.0
 + model-hosting-container-standards==0.1.16
 + mpmath==1.3.0
 + msgspec==0.22.0
 + multidict==7.1.0
 + networkx==3.7
 + ninja==1.13.2
 + numba==0.65.0
 + numpy==2.3.5
 + openai==3.35.0
 + opencv-python-headless==5.0.0.93
 + opentelemetry-api==1.45.1
 + opentelemetry-exporter-http-transport==0.66b1
 + opentelemetry-exporter-otlp==1.45.1
 + opentelemetry-exporter-otlp-common==0.66b1
 + opentelemetry-exporter-otlp-proto-common==1.45.1
 + opentelemetry-exporter-otlp-proto-grpc==1.45.1
 + opentelemetry-exporter-otlp-proto-http==1.45.1
 + opentelemetry-proto==1.45.1
 + opentelemetry-sdk==1.45.1
 + opentelemetry-semantic-conventions==0.66b1
 + opentelemetry-semantic-conventions-ai==0.5.1
 + oss-harmony==0.0.11
 + outlines-core==0.2.14
 + packaging==26.3
 + partial-json-parser==0.2.1.1.post7
 + pillow==13.0.0
 + prometheus-client==0.26.0
 + prometheus-fastapi-instrumentator==8.1.0
 + propcache==0.5.4
 + protobuf==7.36.2
 + psutil==7.2.2
 + py-cpuinfo==9.0.0
 + pybase64==1.5.1
 + pycountry==26.2.16
 + pycparser==3.11
 + pydantic==2.14.1
 + pydantic-core==2.50.1
 + pydantic-extra-types==2.11.1
 + pydantic-settings==2.16.0
 + pygments==2.21.0
 + pyjwt==2.15.1
 + python-dotenv==1.2.4
 + python-json-logger==4.2.0
 + python-multipart==0.0.32
 + pyyaml==6.0.3
 + pyzmq==27.2.0
 + referencing==0.37.0
 + regex==2026.9.29
 + requests==2.34.2
 + rich==15.0.0
 + rich-toolkit==0.20.6
 + rignore==0.8.1
 + rpds-py==2026.9.1
 + safetensors==0.8.0
 + sentencepiece==0.2.2
 + sentry-sdk==2.72.0
 + setproctitle==1.3.8
 + setuptools==77.0.3
 + shellingham==1.5.4
 + six==1.17.0
 + sniffio==1.3.1
 + sse-starlette==3.5.0
 + starlette==1.8.0
 + supervisor==4.3.0
 + sympy==1.14.0
 + tiktoken==0.14.0
 + tokenizers==0.23.3
 + torch==2.13.0
 + torchaudio==2.11.0
 + torchcodec==0.17.0
 + torchvision==0.28.0
 + tqdm==4.70.1
 + transformers==5.17.0
 + truststore==0.10.5
 + typer==0.27.3
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
 + urllib3==2.8.0
 + uvicorn==0.54.0
 + uvloop==0.23.0
 + vllm==0.31.0+cpu (from https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl)
 + watchfiles==1.3.0
 + websockets==17.2
 + xgrammar==0.2.7
 + yarl==1.25.1
✓ Installed vLLM core
Using Python 3.12.13 environment at: .venv-vllm-metal
Resolved 82 packages in 992ms
      Built vllm-metal @ file:///Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal
Prepared 4 packages in 4.06s
Installed 12 packages in 130ms
 + accelerate==1.15.0
 + miniaudio==1.71
 + mlx==0.32.2
 + mlx-audio==0.5.9
 + mlx-lm==0.32.0 (from git+https://github.com/ml-explore/mlx-lm@9e6acca691e64d6d8bb808c328fcdea459099cca)
 + mlx-metal==0.32.2
 + mlx-vlm==0.7.6
 + nanobind==2.10.2
 + opencv-python==5.0.0.93
 + scipy==1.18.1
 + sounddevice==0.5.6
 + vllm-metal==0.31.0 (from file:///Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal)
=== Ensuring Metal toolchain ===
✓ Metal toolchain present
=== Building native Metal artifacts ===
INFO:__main__:Building native paged-attention extension ...
INFO:__main__:  clang++ -std=c++17 -shared -fPIC -O2 -mmacosx-version-min=15.0 -fvisibility=default -I/Users/maochenxiao/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/include/python3.12 -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/nanobind/include -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/nanobind/src -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/nanobind/ext/robin_map/include -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/mlx/include -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/mlx/include/metal_cpp -L/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/mlx/lib -lmlx -framework Metal -framework Foundation -framework IOKit -framework CoreFoundation -D_METAL_ -DACCELERATE_NEW_LAPACK -DVLLM_METAL_PARTITION_SIZE=512 -DVLLM_METAL_PA_WINDOW_ROWS=2 -DVLLM_METAL_PA_WINDOW_MAX_HEAD=256 -undefined dynamic_lookup /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/nanobind/src/nb_combined.cpp /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_ops.cpp /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/mlx_patch.cpp -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/_paged_ops.cpython-312-darwin.so
INFO:__main__:Built /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/_paged_ops.cpython-312-darwin.so
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/_paged_ops.cpython-312-darwin.so
INFO:__main__:Building Metal shader libraries ...
INFO:__main__:  xcrun -sdk macosx metal -fno-fast-math -mmacosx-version-min=15.0 /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/tmpmpvfizrk.metal -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_attention_v2_kern.metallib
INFO:__main__:  xcrun -sdk macosx metal -fno-fast-math -mmacosx-version-min=15.0 /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/tmpfhcqo6g7.metal -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/gdn_kern.metallib
INFO:__main__:  xcrun -sdk macosx metal -fno-fast-math -mmacosx-version-min=15.0 /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/tmp4e2vfi36.metal -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_mla_kern.metallib
INFO:__main__:  xcrun -sdk macosx metal -fno-fast-math -mmacosx-version-min=15.0 /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/tmpb6pj9roq.metal -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/rope_kern.metallib
INFO:__main__:  xcrun -sdk macosx metal -fno-fast-math -mmacosx-version-min=26.2 /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/tmpbp5r4o_v.metal -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_attention_nax_kern.metallib
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_attention_v2_kern.metallib
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/gdn_kern.metallib
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_mla_kern.metallib
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/rope_kern.metallib
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/metal/paged_attention_nax_kern.metallib
INFO:root:building '_mps_ops' extension
INFO:root:creating /var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm-metal-mps-build-rej9ank2/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend
INFO:root:cc -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -O3 -arch arm64 -mmacosx-version-min=11.0 -Wno-nullability-completeness -Wno-expansion-to-defined -Wno-undef-prefix -fPIC -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/torch/include -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/include -I/Users/maochenxiao/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/include/python3.12 -c /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/mps_ops.mm -o /var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm-metal-mps-build-rej9ank2/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/mps_ops.o -O3 -mmacosx-version-min=15.0 -DVLLM_METAL_PARTITION_SIZE=512 -DVLLM_METAL_PA_WINDOW_ROWS=2 -DVLLM_METAL_PA_WINDOW_MAX_HEAD=256 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_mps_ops -std=c++20
In file included from /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/mps_ops.mm:7:
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/torch/include/ATen/native/mps/OperationUtils.h:845:13: warning: unused
      function 'is_dense_broadcastable' [-Wunused-function]
  845 | static bool is_dense_broadcastable(const Tensor& from, const Tensor& into) {
      |             ^~~~~~~~~~~~~~~~~~~~~~
1 warning generated.
INFO:root:c++ -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -O3 -arch arm64 -mmacosx-version-min=11.0 -Wno-nullability-completeness -Wno-expansion-to-defined -Wno-undef-prefix -fPIC -bundle -undefined dynamic_lookup -arch arm64 -mmacosx-version-min=11.0 -LModules/_hacl -Wl,-headerpad,40 /var/folders/7g/nmg6_hj17hn31qmn93dt0r400000gn/T/vllm-metal-mps-build-rej9ank2/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/mps_ops.o -L/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/torch/lib -L/Users/maochenxiao/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -o /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/_mps_ops.cpython-312-darwin.so -mmacosx-version-min=15.0 -framework Metal -framework Foundation -framework IOKit -framework CoreFoundation
ld: warning: search path 'Modules/_hacl' not found
/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/vllm_metal/pytorch_backend/_mps_ops.cpython-312-darwin.so

✓ Installation complete!

To use vllm, activate the virtual environment:
  source /Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/bin/activate

Or add the venv to your PATH:
  export PATH="/Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal/.venv-vllm-metal/bin:$PATH"
```

### 安装dev依赖

`uv pip install -e ".[dev]"`是安装dev依赖：

```
❯ uv pip install -e ".[dev]"
Using Python 3.12.13 environment at: .venv-vllm-metal
Resolved 93 packages in 2.49s
      Built vllm-metal @ file:///Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal
Prepared 7 packages in 1m 11s
Uninstalled 1 package in 5ms
Installed 12 packages in 32ms
 + ast-serialize==0.13.1
 + gguf==0.19.0
 + iniconfig==2.3.1
 + librt==0.16.0
 + mypy==2.4.0
 + mypy-extensions==1.1.0
 + pathspec==1.1.1
 + pluggy==1.7.0
 + pytest==9.1.1
 + pytest-asyncio==1.4.0
 + ruff==0.16.0
 ~ vllm-metal==0.31.0 (from file:///Users/maochenxiao/workspace/aiengineer/projects/community/vllm-metal)
```

## References

* https://github.com/seancxmao/vllm-metal/blob/main/docs/CONTRIBUTING.md
