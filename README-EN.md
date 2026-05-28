# RobiAgent

<p align="center">
    <a href="https://github.com/SamuelGong/RobiAgent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SamuelGong/RobiAgent?color=yellow" alt="License"></a>
</p>

<p align="center">
    <a href="./README.md">简体中文</a> | <strong>English</strong>
</p>

RobiAgent is an **embodied agent** designed to help physical bodies complete complex, long-horizon tasks in real-world environments where multiple modules need to work together.

As a **task orchestration and execution framework**, RobiAgent can start from a high-level goal and:
* decompose the task into a series of executable skills,
* analyze dependencies among those skills,
* schedule different physical resources,
* and drive task completion through perceptual feedback, shared state, and closed-loop adjustment during execution.

To help developers quickly experience an end-to-end workflow, the current version of RobiAgent also ships with **a set of ready-to-use perception and action modules**, including LeRobot arm control, camera input, speech output, and screen interaction.

> RobiAgent, however, is not tied to any specific hardware setup or interaction pattern. Developers can plug in new sensors, actuators, or external tools as needed, wrap them as schedulable skills, and bring them into a unified embodied task workflow.

<img width="1190" height="578" alt="Screenshot 2026-05-28 at 13 25 05" src="https://github.com/user-attachments/assets/974a997d-6463-4362-af9d-e5b933afbd2b" />

# 1. Preparation

## 1.1 Python Environment

The current version of RobiAgent is built on top of LeRobot.
Therefore, you need to prepare a corresponding conda environment first — assuming it is named `lerobot`.
You can refer to [this guide](https://github.com/EmbodiedFX/LeRobot-SO101/blob/main/README-CH.md#%E4%B8%89%E5%9C%A8-macbook-%E5%87%86%E5%A4%87-lerobot-%E6%89%80%E9%9C%80%E7%9A%84-python-%E7%8E%AF%E5%A2%83) for an existing setup reference. Then run:

```bash
conda activate lerobot
```

Next, install RobiAgent itself:

```bash
pip install -e .
```

## 1.2 IPC Setup

Multi-process execution is an important feature of RobiAgent, and its implementation is based on Redis.
Therefore, you also need to install and run a Redis server at the system level.

```bash
bash install_redis_server.sh  # should work on multiple operating systems
nohup redis-server &
```

## 1.3 Physical Environment Setup

Use the following command to find the USB port corresponding to each robotic arm:

```bash
lerobot-find-port
```

Keep the port numbers. Then calibrate each arm with:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<the port you confirmed> --robot.id=<the ID you assigned>
```

If needed, use the following command to find the UID of the RGB camera — the depth camera is not required here:

```bash
swift utils/list_cams.swift
```

Keep the UID for later use.

# 2. Demo Example: Smartphone Sales Assistant in a Showroom

This demo presents a smartphone demonstration flow for an **offline retail sales scenario**.
When a customer enters the space or wakes up the system:
* RobiAgent coordinates two robotic arms and the speech module. With face tracking, one arm first moves the phone to a position and angle that are **more comfortable for the customer to view**, while the system introduces product highlights through **speech output**;
* When further feature demonstration is needed, the other arm can **tap and swipe on the phone screen** to provide a more direct functional walkthrough.

The overall process demonstrates RobiAgent's ability to connect customer perception, voice explanation, robotic-arm motion control, and terminal interaction into one coherent workflow.

https://github.com/user-attachments/assets/ec7ef897-21cb-43f1-afcc-c0d3baec4243

> The video includes audio narration. For the best experience, please unmute it.

**Try it out!**

If this is your first run, create the required configuration files `config.yml` and `.env` first:

```bash
cp config.yml_template config.yml
cp .env_template .env
```

Then update them as needed. For the meaning of each field, refer to [config_intro.md](doc/config_intro.md) and [env_intro.md](doc/env_intro.md).

Finally, try this simple example:

```bash
ra demo
```
