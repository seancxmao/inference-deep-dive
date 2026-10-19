# Build vLLM on Apple silion CPU

## Background


## Build脚本

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

## 总体流程

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


## 平台

`platform_machine` 和 `platform_system` 都是 Python 环境标记（environment markers）中使用的属性，用于描述当前运行环境，但它们描述的维度不同。

| 属性                 | 含义         | macOS Apple Silicon 示例               |
| ------------------ | ---------- | ------------------------------------ |
| `platform_machine` | CPU / 机器架构 | `arm64` 或 `aarch64`（具体取决于 Python 构建） |
| `platform_system`  | 操作系统名称     | `Darwin`                             |

在 Python 包依赖配置中，还经常看到 `sys_platform`。

| 环境标记               | macOS                | Linux                  | Windows             |
| ------------------ | -------------------- | ---------------------- | ------------------- |
| `platform_machine` | `arm64` / `x86_64` 等 | `x86_64` / `aarch64` 等 | `AMD64` / `ARM64` 等 |
| `platform_system`  | `Darwin`             | `Linux`                | `Windows`           |
| `sys_platform`     | `darwin`             | `linux`                | `win32`             |

在 `uv` 依赖解析中如何使用？假设 `pyproject.toml` 中有：

```
dependencies = [
    "some-package; platform_machine == 'aarch64'",
    "another-package; platform_system == 'Darwin'",
]
```

含义分别是：

- 只有在机器架构为 `aarch64` 时，才需要 `some-package`。
- 只有在操作系统为 Darwin（macOS）时，才需要 `another-package`。

这些条件可以组合：

```
"some-package; platform_system == 'Darwin' and platform_machine == 'arm64'"
```

如何查看当前 Python 环境的实际值？直接运行：

```
python -c "import platform, sys; print('machine:', platform.machine()); print('system:', platform.system()); print('sys.platform:', sys.platform)"
```

核心区别： `platform_machine` 回答“什么架构”，`platform_system` 回答“什么操作系统”，`sys_platform` 则提供 Python 所使用的平台标识。

## 版本

直接clone个人fork，build后的版本不对，类似`0.1.dev23312+g38127f629`。规避的办法是，rebase到upstream/main，版本才正确，类似`0.32.1rc1.dev32+ged871c91c`。

vLLM使用`setuptools-scm`动态生成版本，怀疑跟`setuptools-scm`有关。setuptools-scm是Python生态中的一个基于版本控制系统自动生成项目版本号的工具。它通常从Git仓库的标签、提交历史和工作区状态推导版本号，避免开发者每次提交代码都手动修改版本字符串。可以用`setuptools-scm`查看实际识别的版本。实验发现，rebase前后的build版本号跟`setuptools-scm`查看结果一致，所以应该是`setuptools-scm`的问题。

```
uv run --no-project --with setuptools-scm python -m setuptools_scm
```

相关Git操作，记录在这里备用：

```
git rev-parse --short HEAD

# 当前提交可达的版本标签
git describe --tags --long --always

# 查看所有版本标签
git tag --list 'v*' --sort=-version:refname | head -20
```

## References

* https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#build-wheel-from-source
