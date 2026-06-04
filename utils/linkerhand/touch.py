import time
from threading import Lock
from typing import List

from pymodbus.client import ModbusSerialClient


class ModbusDriver:
    FRAME_GAP = 0.030  # 30 ms

    """
    ModbusDriver is a class that wraps the ModbusSerialClient class and provides a context manager for the ModbusSerialClient class.
    It also provides a method to read the pressure data from the hand.
    """

    def __init__(self, hand_id=0x27, modbus_port="/dev/ttyUSB0", baudrate=115200):
        """
        hand_id: 0x27 is right hand; 0x28 is left hand;
        modbus_port: the port of the modbus, default is /dev/ttyUSB0
        baudrate: the baudrate of the modbus, default is 115200, don't support for other baudrate now
        """
        self._id = hand_id
        self._lock = Lock()
        self._last_ts = 0.0
        self._last_pressure_finger_id = 0
        self.cli = ModbusSerialClient(
            port=modbus_port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.15,
            handle_local_echo=False,
        )

    def connect(self):
        if not self.cli.connect():
            raise ConnectionError(f"Failed to connect to {self.cli.port}")

    def disconnect(self):
        if not self.cli.disconnect():
            raise ConnectionError(f"Failed to disconnect from {self.cli.port}")

    def __enter__(self):
        """进入 with 块时自动连接串口"""
        if not self.cli.connect():
            raise ConnectionError(f"Failed to connect to {self._client.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时自动关闭串口"""
        self.cli.close()

    def _bus_free(self):
        """保证距离上一帧 ≥ 30 ms"""
        with self._lock:
            elapse = time.perf_counter() - self._last_ts
            if elapse < self.FRAME_GAP:
                time.sleep(self.FRAME_GAP - elapse)

    def _send_with_gap(self, send_func):
        """在锁保护下确保帧间隔，执行发送函数，更新时间戳"""
        with self._lock:
            elapsed = time.perf_counter() - self._last_ts
            if elapsed < self.FRAME_GAP:
                time.sleep(self.FRAME_GAP - elapsed)
            result = send_func()
            self._last_ts = time.perf_counter()
            return result

    def _read_reg(self, address: int, count: int):
        def _read_func():

            rsp = self.cli.read_input_registers(
                address=address, count=count, slave=self._id
            )

            if rsp.isError():
                raise RuntimeError(
                    f"Modbus Read Failed (Addr={address}, Count={count}): {rsp}"
                )
            # 确保返回的值是 Python 原生整数
            return rsp.registers

        return self._send_with_gap(_read_func)

    def _write_reg(self, address: int, values: List[int]):
        """执行 Modbus 批量写入操作 (功能码 16), 带总线仲裁。"""

        def _write_func():
            rsp = self.cli.write_registers(
                address=address, values=values, slave=self._id
            )
            if rsp.isError():
                raise RuntimeError(
                    f"Modbus Write Failed (Addr={address}, Values={values}): {rsp}"
                )
            return rsp.registers

        return self._send_with_gap(_write_func)

    def read_pressure(self, finger_id: int):
        if finger_id < 0 or finger_id > 6:
            raise ValueError("finger_id must be 0-5")
        if (
            self._last_pressure_finger_id == 0
            or self._last_pressure_finger_id != finger_id
        ):
            self._write_reg(18, [finger_id])
            self._last_pressure_finger_id = finger_id
        if self._read_reg(45, 1)[0] != finger_id:
            raise RuntimeError("finger_id mismatch")
        # 获取压力数据的基本信息（size等)
        spec = self._read_reg(46, 1)[0]
        rows = (spec >> 4) & 0x0F
        cols = spec & 0x0F
        total_cells = rows * cols
        # 读取所有压力数据
        data = self._read_reg(47, total_cells)
        pressure_matrix = [data[i * cols : (i + 1) * cols] for i in range(rows)]
        return pressure_matrix


def main():
    with ModbusDriver(modbus_port="/dev/tty.usbserial-110") as modbus_driver:
        # 获取指定手指的压力数据, 1-大拇指 - 5-小拇指
        data = modbus_driver.read_pressure(5)
        print(data)


if __name__ == "__main__":
    main()
