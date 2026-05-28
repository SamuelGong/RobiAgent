# RobiAgent

<p align="center">
    <a href="https://github.com/SamuelGong/RobiAgent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SamuelGong/RobiAgent?color=yellow" alt="License"></a>
</p>

<p align="center">
    <strong>简体中文</strong> | <a href="./README-EN.md">English</a>
</p>

RobiAgent 是一个**具身智能体**，能够让本体在真实物理环境中完成复杂、长程、需要多模块协同的任务。

作为**任务编排与执行框架**，RobiAgent 可以从高层目标出发，
* 将任务拆解为一系列可执行技能，
* 分析技能之间的依赖关系，
* 调度不同物理资源，
* 并在执行过程中通过感知反馈、状态共享和闭环调整来推进任务完成。

为了便于开发者快速体验端到端能力，当前版本的 RobiAgent 也提供了**一组开箱即用的感知与动作模块**，包括 LeRobot 机械臂控制、相机输入、语音输出和屏幕交互等。

> 但是，RobiAgent 并不绑定于特定硬件或交互形式；开发者可以按需接入新的传感器、执行器或外部工具，将其封装为可调度技能，并纳入统一的具身任务流程。

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

再通过以下命令找到 RGB 相机（深度相机不用）的 UID（如果需要）：

```bash
swift utils/list_cams.swift
```

记住UID。

# 2. 演示示例：展厅手机导购

该 demo 展示了一段面向**线下导购场景**的手机演示流程。
当顾客进入或唤醒系统后，
* RobiAgent 会协调双机械臂与语音模块，先通过人脸追踪，由一只机械臂将手机送到**更适合顾客观看**的位置和角度，同时通过**语音介绍**产品卖点；
* 在需要进一步展示功能时，另一只机械臂会**在手机屏幕上进行点击、滑动等操作**，完成更直观的功能演示。

整个过程体现的是 RobiAgent 将顾客感知、语音讲解、机械臂运动控制与终端交互串联起来的能力。

https://github.com/user-attachments/assets/ec7ef897-21cb-43f1-afcc-c0d3baec4243

> 视频有声音（语音播报），建议取消静音欣赏。

**试试看！**

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
