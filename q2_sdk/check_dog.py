"""Read-only connection test for the Unitree Go2: prints IMU, battery and one joint angle.

Usage (inside the unitree_sdk2_python virtual environment):
    python check_dog.py eth0        # eth0 = the network interface connected to the dog
"""
import sys, time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

count = 0
def handler(msg: LowState_):
    global count
    count += 1
    if count % 100 == 0:  # the dog sends ~500 msgs/sec; print a few per second
        print(f"msgs: {count} | IMU roll/pitch/yaw: {[round(x, 3) for x in msg.imu_state.rpy]}"
              f" | battery: {msg.power_v:.1f} V | FR hip angle: {msg.motor_state[0].q:.3f}")

if len(sys.argv) < 2:
    sys.exit("Usage: python check_dog.py <network interface, e.g. eth0>")
ChannelFactoryInitialize(0, sys.argv[1])
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)
print("Listening to the dog... (Ctrl+C to stop)")
while True:
    time.sleep(1)
