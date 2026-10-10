# Question 2 – Unitree Go2 Python SDK

Setup and results are written up in [`../TECHNICAL_NOTE.md`](../TECHNICAL_NOTE.md#question-2--getting-the-dog-environment-ready).

## Install (WSL Ubuntu)

```bash
mkdir -p ~/eng6010 && cd ~/eng6010
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv venv --python 3.10 .venv          # cyclonedds 0.10.2 only ships prebuilt wheels for Python <= 3.10
source .venv/bin/activate
uv pip install -e .
```

## Examples I ran

| # | Command (from `unitree_sdk2_python/`) | Moves robot? | What it shows |
|---|---|---|---|
| 1 | `python example/helloworld/publisher.py` + `python example/helloworld/subscriber.py` | No robot needed | DDS publish/subscribe works |
| 2 | `python example/go2/front_camera/capture_image.py eth0` | No | Saves `img.jpg` from the front camera |
| 3 | `python example/go2/high_level/go2_sport_client.py eth0` | **Yes** | Stand up (1), walk forward (3), stop (6), lie down (2) |
| – | `python check_dog.py eth0` (this folder) | No | Streams IMU, battery and joint data from `rt/lowstate` |

## Fix for `example/wireless_controller/wireless_controller.py`

By default this example imports the humanoid (G1/H1-2) message type, so it receives nothing from a Go2. Swap the imports:

```python
# Go2 / Go2-W / B2 / B2-W / H1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# G1 / H1-2  (comment these out for the Go2)
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
```

## Networking checklist (WSL)

1. Windows Ethernet adapter: `192.168.123.99` / `255.255.255.0`
2. `%USERPROFILE%\.wslconfig` contains `[wsl2]` and `networkingMode=mirrored`, then run `wsl --shutdown`
3. Admin PowerShell: `Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow`
4. In Ubuntu: `ip -4 addr` → find the interface with `192.168.123.99` (mine is `eth0`); `ping 192.168.123.161`
