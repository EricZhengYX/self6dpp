#!/usr/bin/env bash

this_dir=$(dirname "$0")
weight="/home/eric/datasets/s6dpp/GS6D_ape_e_0083455_oldV.pth"
cls="ape"
cd $this_dir/../

echo ""
echo "******** w/o depth w/o weak ********"
config_file="/home/eric/py_ws/self6dpp/configs/self6dpp/new_cfg_depth_weak/ssLMO/${cls}/${cls}_oo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** w depth w/o weak ********"
config_file="/home/eric/py_ws/self6dpp/configs/self6dpp/new_cfg_depth_weak/ssLMO/${cls}/${cls}_wo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** w/o depth w weak ********"
config_file="/home/eric/py_ws/self6dpp/configs/self6dpp/new_cfg_depth_weak/ssLMO/${cls}/${cls}_ow.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** w depth w weak ********"
config_file="/home/eric/py_ws/self6dpp/configs/self6dpp/new_cfg_depth_weak/ssLMO/${cls}/${cls}_ww.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight
