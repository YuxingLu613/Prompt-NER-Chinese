import csv
import json
import os
import pickle
from re import I
import time
from datetime import timedelta,datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_file(file_path: str, separater: str = None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    :param filepath:
    :param sep:
    :return:
    """
    with open(file_path,"r",encoding="utf-8-sig") as f:
        lines=f.readlines()
        if separater:
            return [line.strip().split(separater) for line in lines]
        else:
            return lines


def load_csv(file_path: str, is_tsv: bool = False):
    """
    加载csv文件为OrderDict()列表
    :param filepath:
    :param is_tsv:
    :return:
    """
    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(file_path, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)

def save_csv(data: pd.DataFrame, file_path: str,sep: str=","):
    """
    将DataFrame数据保存到对应file_path下
    :param data:
    :para file_path:
    :return:
    """
    data.to_csv(file_path,sep=sep)


def load_json(filepath: str):
    """
    加载json文件
    :param filepath:
    :return:
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip(), encoding="utf-8") for line in f.readlines()]


def save_json(list_data, filepath):
    """
    保存json文件
    :param list_data:
    :param filepath:
    :return:
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for data in list_data:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(json_str))
        f.flush()


def load_pkl(filepath):
    """
    加载pkl文件
    :param filepath:
    :return:
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, filepath):
    """
    保存pkl文件，数据序列化
    :param data:
    :param filepath:
    :return:
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def calculate_distance(vector1, vector2,mode="cos"):
    """
    计算两个向量的余弦相似度
    :param vector1: 向量1
    :param vector2: 向量2
    :param mode: "cos"余弦相似度，"euc"欧氏距离
    :return:
    """
    if mode=="cos":
        distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))  # 余弦夹角
    elif mode=="euc":
        distance = np.sqrt(np.sum(np.square(vector1 - vector2)))  # 欧式距离
    return distance


def split_data(data_set, ratio):
    """
    数据集切分
    :param data_set:
    :param ratio:分割比例
    :return:
    """
    split_data1,split_data2=train_test_split(data_set,ratio)
    return split_data1,split_data2


def format_data(t: datetime):
    """
    时间格式化，time.strftime("%Y-%m-%d %H:%M:%S")
    :param t:
    :return:
    """
    return t.strftime("%Y-%m-%d %H:%M:%S")


def get_used_dif(start_time):
    """
    获取已使用时间
    :param start_time: time.time()
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def scan_filepath(path):
    """
    递归返回指定目录下的所有文件
    :param ph:
    :return:
    """
    path_list = []
    for p in os.listdir(path):
        fp = os.path.join(path, p)
        if os.path.isfile(fp):
            path_list.append(fp)
        elif os.path.isdir(fp):
            path_list.extend(scan_filepath(fp))
    return path_list
