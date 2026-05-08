# RobiAgent

<img width="1115" height="550" alt="Screenshot 2026-04-09 at 10 53 22" src="https://github.com/user-attachments/assets/d086bd10-d2e2-4019-a9bb-da9d2cd3457f" />

# 1. 准备工作

## 1.1 Python 环境配置

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

## 1.2 IPC 方案

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

# 2. 演示示例

如果是第一次运行，需要先创建必要的配置文件`config.yml`和`.env`：

```bash
cp config.yml_template config.yml
cp .env_template .env
```

并对它们作必要更改，其中字段含义分别参考 [config_intro.md](doc/config_intro.md) 和 [env_intro.md](doc/env_intro.md)。

随后，试试这个简单示例！

```bash
ra demo
```
