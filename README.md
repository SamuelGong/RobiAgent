# RobiAgent

## 1. 准备工作

### 1.1 Python 环境配置

当前版本的 RobiAgent 首先是基于 LeRobot 构建的，
因此你需要先准备对应的 conda 环境（假设其名称为 `lerobot`；
可以参考[这里](https://github.com/EmbodiedFX/LeRobot-SO101/blob/main/README-CH.md#%E4%B8%89%E5%9C%A8-macbook-%E5%87%86%E5%A4%87-lerobot-%E6%89%80%E9%9C%80%E7%9A%84-python-%E7%8E%AF%E5%A2%83)的现有说明），然后执行：

```bash
conda activate lerobot
````

接下来，需要安装 RobiAgent 本身：

```bash
pip install -e .
```

最后执行：

```bash
cp config.yml_template config.yml
```

并对`config.yml`作必要更改，其中关键字段的含义参考[此文档](doc/config_intro.md)。

### 1.2 IPC 方案

多进程是 RobiAgent 的一个重要特性，其实现基于 Redis。
因此，你还需要在系统层面安装并运行 Redis 服务器。

```bash
bash install_redis_server.sh  # 应该可在多种操作系统上运行
nohup redis-server &
```

## 1.3 物理环境准备

通过以下命令找出每个机械臂对应的 USB 端口：

```bash
lerobot-find-port
```

记住端口号。然后，使用如下命令校准每个机械臂：

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<你确认的端口> --robot.id=<你指定的 ID>
```

不要忘记相应修改 `config.yml`。

# 2. 演示示例

先试试这个简单示例！

```bash
ra demo
```