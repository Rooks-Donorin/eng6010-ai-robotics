#!/usr/bin/env bash
# Runs the whole Question 4 vision pipeline: generate synthetic data, then the five methods.
# Usage:  ./run_all.sh [path/to/unitree_mujoco/unitree_robots/go2/go2.xml]
set -e
cd "$(dirname "$0")"
GO2_XML="${1:-$HOME/eng6010/unitree_mujoco/unitree_robots/go2/go2.xml}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"   # off-screen software rendering, works without a GPU

rm -f results/metrics.json
python 00_generate_synthetic_data.py --go2-xml "$GO2_XML"
python 01_color_space.py
python 02_gaussian_pyramid.py
python 03_optical_flow.py
python 04_grabcut.py
python 05_supervised_learning.py
echo "All done - figures and results/metrics.json are in results/"
