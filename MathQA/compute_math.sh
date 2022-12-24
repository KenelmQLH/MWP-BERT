#!/bin/bash

pred_data_dir=$1
data_name=$2 # arithmetic
data_version=$3 # 97


CRTDIR=$(pwd)
gold_data_file=/data/qlh/Math-Plan/data/linear_expression/${data_name}/${data_version}/${data_name}_test_mwp_format.json

cd tools

if [ -f ${gold_data_file} ];then
    echo "[1] gold_ques already exists ... "
else
    echo "[1] creating gold_ques ... "
    python prepare_mwp.py -data_dir ${gold_data_file} -work_mode gold
fi


if [ -f ${pred_data_dir}/infer_test_mwp_format.json ];then
    echo "[2] pred_ques already exists ... "
else
    echo "[2] creating pred_ques ... "
    python prepare_mwp.py -data_dir ${pred_data_dir}/infer_test_mwp.json -work_mode pred
fi

cd ..


echo "[3] compute equation_acc for gold_ques and pred_ques ... "
python test.py --test_file ${pred_data_dir}/infer_test_mwp_format.json --data_name $2
# --only_test_pred
