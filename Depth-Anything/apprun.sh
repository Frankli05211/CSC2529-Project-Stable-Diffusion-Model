#!/bin/bash

env_dir="/w/339/frankli/CSC2529-Project/csc2529-project-venv/"

# Install necessary dependencies
# pip install -r requirements.txt

# Let home directory to be in virtual environment so that the model template will not fill up /h/ machine
export HF_HOME="$env_dir"

python3 run.py --encoder vitb --img-path assets/examples --outdir depth_vis
