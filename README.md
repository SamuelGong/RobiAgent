# RobiAgent

## 1. Preparation

### 1.1 Python Environment Setup

The current version of RobiAgent is first built on top of LeRobot,
and thus you need to prepare the corresponding conda environment (suppose that it is called `lerobot`; 
one may follow an existing instruction [here](https://github.com/EmbodiedFX/LeRobot-SO101/blob/main/README-CH.md#%E4%B8%89%E5%9C%A8-macbook-%E5%87%86%E5%A4%87-lerobot-%E6%89%80%E9%9C%80%E7%9A%84-python-%E7%8E%AF%E5%A2%83)), and then

```bash
conda activate lerobot
```

Next, one need to install RobiAgent itself:

```bash
pip install -e .
```

### 1.2 IPC Solution

Multiprocessing is an important feature for RobiAgent, and the implementation is based on Redis.
Therefore, one also need to install and run Redis server at the system level.

```bash
bash install_redis_server.sh  # should work at various operating systems
nohup redis-server &
```

## 1.3 Physical Environment Preparation

Find out the USB port for each arm via

```bash
lerobot-find-port
```

Remember the port number. Then, calibrate each arm using commands like:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<Your confirmed port> --robot.id=<Your given ID>
```

Finally, change the `config.yml` to suit your need, important fields include:
- `physical.arm.left.id`, `physical.arm.left.port`, `physical.arm.right.id`, `physical.arm.right.port`.

# 2. Demonstrating Examples

Try this toy example first!

```bash
ra demo
```



