# coding: utf-8
# data preprocess: bert tokenize ; create data/xxx_token_xxx.json
import os
import sys
CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_FILE_DIR))
from EasyData.FileHandler import read_json, write_json


from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import json
import argparse

# --------------------------------------------------------------------- #
# 构造标签（题目正样本）
# --------------------------------------------------------------------- #

def recover_text_from_num(segmented_text, memery_num_to_idx):
    # 
    memery_idx_to_num = {v: k for k, v in memery_num_to_idx.items()}

    for i, token in enumerate(segmented_text):
        if token.startswith("num_"):
            num_idx = int(token.replace("num_", ""))
            segmented_text[i] = memery_idx_to_num.get(num_idx, "unk")

def recover_prediction_for_mwp_bert(items):
    mwp_data = []
    # 恢复原始题目格式
    for item in items:
        segmented_text = copy.deepcopy(item.get("question")).split()
        memery_num_to_idx = item["memery_num_to_idx"]
        recover_text_from_num(segmented_text, memery_num_to_idx)
        original_text = " ".join(segmented_text)

        """ NOTO: 暂时只能处理线性表达式， and format as 'x=.....' """
        assert len(item["equations"]) == 1
        equation = item["equations"][0]
        mwp_data.append({
            "original_text": original_text,
            "equation": equation.replace(" ", ""),
            "ans": str(item["answer"][0])
        })
    return mwp_data

def handle_gold(items):
    mwp_data = recover_prediction_for_mwp_bert(items)
    return mwp_data


def add_args(parser):
    parser.add_argument('-data_dir', type=str, default="/data/qlh/Math-Plan/output/planning1211_graph_5/arithmetic/version_0/checkpoints/infer_valid_mwp.json")
    parser.add_argument('-work_mode',
                        type=str,
                        choices=['gold', 'pred'],
                        default="pred")


parser = argparse.ArgumentParser(description='[Get args for wrok data]')
add_args(parser)
work_opt = parser.parse_args()

# work_opt.data_dir = "/data/qlh/Math-Plan/data/linear_expression/arithmetic/1/arithmetic_valid.json"
# work_opt.data_dir = "/data/qlh/Math-Plan/data/linear_expression/arithmetic/1/arithmetic_train.json"
# work_opt.work_mode = "gold"

# work_opt.data_dir = "/data/qlh/Math-Plan/output/planning1212_g7/arithmetic/version_0/checkpoints/infer_valid_mwp.json"
# work_opt.work_mode = "pred"


LANGUAGE = "english"
src_data_path = work_opt.data_dir
if work_opt.work_mode == "gold":
    gold_data_path = src_data_path.replace(".json", "_mwp.json")
    tgt_data_path = gold_data_path.replace(".json", "_format.json")
else:
    tgt_data_path = src_data_path.replace(".json", "_format.json")

work_mode = work_opt.work_mode


def main():
    # data = load_raw_data(src_data_path)
    data = read_json(src_data_path)
    for item in data:
        item["equation"] = item["equation"].replace(" ", "")

    if work_mode == "gold":
        data = handle_gold(data)

        write_json(data, gold_data_path)

    TRANSFER_METHOD = transfer_num
    # TRANSFER_METHOD = transfer_english_num
    pairs, generate_nums, copy_nums = TRANSFER_METHOD(data)

    if LANGUAGE == "english":
        bert_tokenizer = BertTokenizer.from_pretrained(os.path.dirname(CUR_FILE_DIR) + '/pretrained_models/mwp-bert-en') # ../pretrained_models/
        bert_tokenizer.add_special_tokens({"additional_special_tokens":["[num]"]})
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased') # ../pretrained_models/
        # bert_tokenizer.add_tokens(['[num]'])
        bert_tokenizer.add_special_tokens({"additional_special_tokens":["[num]"]})
    count = 0
    new_items = []
    for item in pairs:
        old_token = item[0]
        sent = ""
        for token in old_token:
            if (token == 'NUM'):
                sent += " " + "[num]"
            else:
                sent += " " + token
        sent = "[CLS]" + sent + "[SEP]"
        new_token = bert_tokenizer.tokenize(sent)
        new_num_pos = [] # use bert to tokenize，numpos changed
        for i, token in enumerate(new_token):
            if (token == '[num]' or token == '[NUM]'):
                new_num_pos.append(i)
        if len(new_num_pos) != len(item[2]):
            print("new num error")
            print("old:", old_token)
            print("new:", new_token)
            count += 1
        new_item = {
            "tokens": new_token,
            "expression": item[1],
            "nums": item[2],
            "num_pos": new_num_pos
        }
        new_items.append(new_item)
    print(count)
    print(123)
    
    json.dump(
        {"pairs": new_items, "generate_nums": generate_nums, "copy_nums": copy_nums},
        open(tgt_data_path, "w"),
        indent=4,
        ensure_ascii=False
    )

if __name__ == "__main__":
    main()