# vLLM Debug

## Background


## Python Debug

### pdb

`demo.py`：

```python
def add(a, b):
    result = a + b
    return result

x = 10
y = 20
z = add(x, y)
print(z)
```

用`pdb`启动`demo.py`

```
python -m pdb demo.py
```

此时`pdb`和`demo.py`运行在同一个Python进程中：

```
❯ ps -eo pid,ppid,stat,command | grep -i '[p]ython'
23484  2044 S+   python -m pdb demo.py
```

用于调试常用命令如下：

|命令   |含义  |
|------|------|
|n      |next，执行下一行，不进入被调用函数 |
|s	    |step，执行下一步，必要时进入函数 |
|c	    |continue，继续执行直到下一个断点或程序结束 |
|b 2	|在第 2 行设置断点 |
|p result	|打印变量 result 的值 |
|l	    |查看当前代码附近的内容 |
|w	    |查看调用栈|
|q	    |退出调试|

### debugpy

```
python -m debugpy --listen 5678 --wait-for-client demo.py
```

此时会启动2个Python进程：

* 被调试程序，跟debugpy的debug server在同一个进程
* debugpy的debug adapter进程

```
❯ ps -eo pid,ppid,stat,command | grep -i '[p]ython'
31694  2044 S+   python -m debugpy --listen 5678 --wait-for-client demo.py
31700     1 S    /.../.venv/bin/python /.../.venv/lib/python3.12/site-packages/debugpy/adapter --for-server 55645 --host 127.0.0.1 --port 5678 --server-access-token b8dfe2ac24baeb01b5d0b272f47a6a7149e3e107a35a970eff9cc3f6a6c836bc
```

用VS Code打开demo.py所在项目，Run -> Add Configuration，在代码上打断点，点击 Run and Debug：

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        }
    ]
}
```

此时，VS Code作为debug client。

VS Code <-> debug adapter <-> 被调试程序，三者进行交互完成调试。debugpy官方文档的术语：

* IDE - VSCode or other DAP client.
* Debug adapter - debugpy adapter that mediates between IDE and server.
* Debug server - pydevd with debugpy wrapper; hosted inside the debuggee process, one for each.
* Debuggee process - the process that is being debugged.

可以从三个层次理解：

* Python运行时：负责执行Python代码，并提供跟踪执行、检查栈帧等能力。
* 调试后端：例如debugpy，负责断点管理、变量检查、单步执行以及调试通信协议。
* 调试前端：例如VS Code的调试界面，负责展示变量、调用栈、断点等，并把你的操作传给调试后端。

## Debug vLLM with VS Code

TODO

## References

* https://github.com/microsoft/debugpy/wiki/Ramping-up-on-debugpy
* https://github.com/microsoft/debugpy/blob/main/doc/Subprocess%20debugging.md
