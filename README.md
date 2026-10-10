# ENG 6010 – AI in Robotics: First Test

**Author:** Matthew Crist
**Robot:** Unitree Go2 X

This repo holds my writing, notes, and code for the ENG 6010 First Test.

| Question | Topic | Where to look | Status |
|---|---|---|---|
| 1 | Review paper: color space, Gaussian pyramid, optical flow, GrabCut, supervised learning | [`q1_review_paper/review_paper.pdf`](q1_review_paper/review_paper.pdf) | Done |
| 2 | Getting the dog environment ready: Linux, SDK, three examples | [`TECHNICAL_NOTE.md` §2](TECHNICAL_NOTE.md#question-2--getting-the-dog-environment-ready), [`q2_sdk/`](q2_sdk/) | Done (screenshots being added) |
| 3 | Getting familiar with the dog apps | [`TECHNICAL_NOTE.md` §3](TECHNICAL_NOTE.md#question-3--getting-familiar-with-dog-apps), [`q3_demos/`](q3_demos/) | Review done, demo in progress |
| 4 | Dog data and simulation | [`TECHNICAL_NOTE.md` §4](TECHNICAL_NOTE.md#question-4--dog-data-and-simulation), [`q4_vision/`](q4_vision/) | Data plan + all five methods done; rl_gym simulation in progress |

**Questions 2–4 are written up as one technical note: [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).**

## Repo layout

```
.
├── TECHNICAL_NOTE.md      # technical note for Questions 2-4
├── images/                # screenshots and photos used in the note
├── q1_review_paper/       # Question 1 review paper
├── q2_sdk/                # my scripts for the Unitree Python SDK (check_dog.py)
├── q3_demos/              # notes for the Go2 demo apps
└── q4_vision/             # synthetic Go2 data + the five vision/learning methods
    ├── 00_generate_synthetic_data.py ... 05_supervised_learning.py
    ├── data/              # generated synthetic data
    └── results/           # figures + metrics.json
```

## My setup

- Windows PC → **WSL2 Ubuntu** (mirrored networking), VS Code with the WSL extension
- [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python) in a **Python 3.10** virtual environment made with `uv`
- Go2 X connected by Ethernet: PC `192.168.123.99`, robot `192.168.123.161`, Linux interface `eth0`

Quick start for the dog examples:

```bash
cd ~/eng6010/unitree_sdk2_python && source .venv/bin/activate
python ~/eng6010/eng6010-ai-robotics/q2_sdk/check_dog.py eth0
```

Quick start for Question 4 (see [`q4_vision/README.md`](q4_vision/README.md)):

```bash
cd q4_vision && ./run_all.sh ~/eng6010/unitree_mujoco/unitree_robots/go2/go2.xml
```

## Tools

VS Code, Claude Code, Python, MuJoCo, OpenCV, scikit-learn.
