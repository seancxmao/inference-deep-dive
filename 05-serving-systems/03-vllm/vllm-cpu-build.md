# Build vLLM on Apple silion CPU

## Build概览

### 主要的build脚本

```
source .../venvs/vllm-dev/bin/activate
cd vllm
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

需要看详细信息，可以加`-v`：

`pip install -v -e .`

从开发者视角看，最重要的是4个层次：

```text
pip install -e .
    ↓
pyproject.toml
    ↓
setup.py
    ↓
cmake/
    ↓
csrc/*
```

其中：

* `pyproject.toml`：Python打包入口，告诉pip如何构建项目。
* `setup.py`：vLLM的核心构建脚本，决定编译哪些Extension、启用哪些平台（CUDA/CPU/Metal）。
* `cmake/`：存放CMake配置和辅助脚本，负责组织底层C++编译。
* `csrc/`：真正被编译的源码。

### 总体流程

当执行`pip install -e .`时，流程可以理解为：

```text
pyproject.toml
    ↓
启动构建环境
    ↓
setup.py
    ↓
检查平台(CUDA/ROCm/CPU/Metal)
    ↓
调用CMake
    ↓
编译csrc/*
    ↓
生成vllm/_C.abi3.so
    ↓
editable install
```

最终结果有两部分：

* Python代码不复制，直接指向当前源码目录（editable mode）。
* C++代码被编译成`vllm/_C.abi3.so`，供Python通过`torch.ops.xxx(...)`调用
  
根据以上原理，可以判断是否需要重新执行`pip install -e .`：

* 只修改Python，无需重新build
* 修改`csrc/*`，必须重新build

## References

* https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#build-wheel-from-source
