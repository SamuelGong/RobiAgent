# 工具脚本合集

1. MacOS 上看摄像头本身所支持的配置：

```bash
swift list_cams.swift  # 查看摄像头的 UID
swift cam_formats.swift <摄像头的UID>
```

2. 确定某个SO101机械臂姿态对应的关节量，并写出到文件

```bash
vim record_target_pose_yaml.py  # 先修改 ROBOT_ID, PORT 和 OUTPUT_PATH
python record_target_pose_yaml.py
```