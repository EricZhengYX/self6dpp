#!/usr/bin/env bash

this_dir=$(dirname "$0")
cls1="ape"
cls2="benchvise"
cd $this_dir/../

echo ""
echo "******** LM1: w/o depth w/o weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls1}/${cls1}_oo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM1: w depth w/o weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls1}/${cls1}_wo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM1: w/o depth w weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls1}/${cls1}_ow.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM1: w depth w weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls1}/${cls1}_ww.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight


echo ""
echo "******** LM2: w/o depth w/o weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls2}/${cls2}_oo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM2: w depth w/o weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls2}/${cls2}_wo.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM2: w/o depth w weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls2}/${cls2}_ow.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight

echo ""
echo "******** LM2: w depth w weak ********"
config_file="${this_dir}/../configs/self6dpp/new_cfg_depth_weak/ssLM/${cls2}/${cls2}_ww.py"
python ./core/self6dpp/main_self6dpp.py --config-file $config_file --num-gpus 1 #--eval-only --opts MODEL.WEIGHTS=$weight
