#!/bin/bash

data_name=$1 # # arithmetic
data_version=$2 # 97

CRTDIR=$(pwd)
root_data_dir=${CRTDIR}/data/${data_name}/${data_version}

cd tools

train_data_file=${root_data_dir}/${data_name}_train_mwp_format.json
if [ -f ${train_data_file} ];then
    echo "[1] tarin_data: already exists ... "
else
    echo "[1] tarin_data: creating  ... "
    python prepare_mwp.py -work_mode gold -data_dir ${root_data_dir}/${data_name}_train.json
fi

valid_data_file=${root_data_dir}/${data_name}_valid_mwp_format.json
if [ -f ${valid_data_file} ];then
    echo "[1] valid_data: already exists ... "
else
    echo "[1] valid_data: creating  ... "
    python prepare_mwp.py -work_mode gold -data_dir ${root_data_dir}/${data_name}_valid.json
fi

test_data_file=${root_data_dir}/${data_name}_test_mwp_format.json
if [ -f ${test_data_file} ];then
    echo "[1] test_data: already exists ... "
else
    echo "[1] test_data: creating  ... "
    python prepare_mwp.py -work_mode gold -data_dir ${root_data_dir}/${data_name}_test.json
fi
