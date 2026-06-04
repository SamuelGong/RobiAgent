## 基本资料
这里是LinkerHand灵巧手给的基本资料：
1. 上位机调试工具：https://docs.linkerhub.work/vertex/zh-cn/   
	‼️Attention: windows或者linux， 而且这个上位机软件是 CAN通讯协议的，我们的灵巧人买的是RS485协议，这个上位机软件不可用。他们给了一个rs485的上位机调试工具，但是只有windows的，不过不用担心，这里不需要太过于担心这个问题。我们可以不使用这个工具，这里的上位机上位机调试工具不是那么重要，它其实就是一个页面，可以直观的控制和观察灵巧手的活动，可以让我们对灵巧手的操控和形态有一个比较直观和清晰的认识。
2. 开发SDK（推荐Python）：https://github.com/orgs/linker-bot/repositories 优先用Python SDK开发，上手快、适配全系列手型，有问题随时沟通。
3. 在开发者中心,您可以找到: https://document.linkeros.cn/developer/1
• 使用手册: 详细指导如何使用不同型号的LinkerHand产品,包括L10、L20以及人形机器人与灵巧手的配套操作说明。
• SDK开发工具: 完整的开发工具包及接口文档,助力快速集成和创新应用。
• 技术协议:CAN通信协议详解,为开发者提供标准化的技术支持。
• Demo演示:典型抓取应用和最小系统的软件演示,让您更直观地了解灵巧手的强大能力。
• 产品模型:包括连接件和3D模型等资源,助力产品设计与开发。

## 灵巧手使用
目前在ubuntu 和 macos上成功跑起来了：
#### 正确连接线：
电源适配器、灵巧手、还有一个rs485<->USB的接口。电源上电之后，灵巧手会开始进行自检，也就是他的六个关节会进行弯曲，这个是自动的过程，我们不用管。USB接口插到电脑上。
#### 软件环境准备：
<https://document.linkeros.cn/developer/69>； 具体我的实操如下：
1. 下载灵巧手的sdk： <https://github.com/linker-bot/linkerhand-python-sdk>
2. 构建一个新的虚拟环境： 
  ``` bash
  conda create -n linkerhand python=3.12
  conda activate linkerhand
  ```
3.  安装对应的依赖：
  ```bash 
	$ pip install -r requirements.txt
	# Install system-level related drivers
	$ pip install minimalmodbus --break-system-packages
	$ pip install pyserial --break-system-packages
	$ pip install pymodbus --break-system-packages
	# View the USB-RS485 port number
	$ ls /dev
	# You should see a port similar to ttyUSB0. Grant permissions to the port:
	$ sudo chmod 777 /dev/ttyUSB0 (if your os is macos, it maybe /dev/tty.usbserial-1130)
	# GUI control example
  ```
4. 修改配置文件：
```bash 
vim LinkerHand/config/setting.yaml
```
```
EFT_HAND:
	EXISTS: FALSE # 我们的是右手
RIGHT_HAND:
	EXISTS: TRUE # 是否存在右手
	TOUCH: TRUE # 是否有压力传感器
	CAN: "can0" # 这个不重要，因为下面的配置了rs485协议之后，这个字段就无效了
	MODBUS: "/dev/tty.usbserial-1130" # 通讯协议是否为485 默认None 如果启动485，则是设备端口 /dev/ttyUSB* CAN配置失效 当前只支持O6/L6.后续版本正在努力增加中
	JOINT: O6 # 右手型号 O6/L6/L7/L10/L20/G20/L21/L25/
```
5. 运行example：
```bash 
python example/gui_control/gui_control.py
```
会得到一个控制界面，控制界面上有一些预设的动作，可以初步感受灵巧手的运动方式
![[Pasted image 20260529112051.png]]

‼️ Attention: 现在给定的SDK中还不支持获取触觉信号，获取触觉信号要自己通过RS485 modbus的寄存器来获取，寄存器的信息在：<https://linkerbot.feishu.cn/sheets/Xe7qsazIYhTMeWtx44UcozghnOf?sheet=2QxEeN>

我实现了一个读取每个指头的案例在 thirdparty/linkerhand/touch/touch.py
