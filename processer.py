import enum
import torch
from utils import load_file, save_pkl, load_pkl
from config import Config
from logger import logger as logging
import os
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset

config = Config()


def dataset_format(file_path, separater=" "):
    label_map = {label: i for i, label in enumerate(config.label_list)}
    sentence_list = []
    label_list = []
    sentence = ""
    labels = []
    file = load_file(file_path, separater=separater)
    for unit in file[:]:
        if len(unit) <= 1:
            if len(sentence) >= config.min_sequence_length:
                sentence_list.append(sentence[:58])
                sentence = ""
                label_list.append(labels[:58])
                labels = []
            continue
        word, label = unit
        if word == "，" or word == "。":
            if len(sentence) >= config.min_sequence_length:
                sentence_list.append(sentence[:58])
                sentence = ""
                label_list.append(labels[:58])
                labels = []
        else:
            sentence += word
            labels.append(label_map[label])
    return sentence_list, label_list


class Template():
    def __init__(self, text, label) -> None:
        self.text = text
        self.label = label


def create_template(all_texts, all_labels):
    dataset = []
    for item in zip(all_texts, all_labels):
        texts, labels = item
        for word, label in zip(list(texts), labels):
            template = Template(text=texts+"，"+word+"是", label=label)
            dataset.append(template)
    return dataset


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, attention_mask, masked_index, label_id, ori_label):
        """
        :param input_ids:       单词在词典中的编码
        :param attention_mask:  指定 对哪些词 进行self-Attention操作
        :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
        :param label_id:        标签的id
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id
        self.ori_label = ori_label
        self.masked_index = masked_index


def convert_examples_to_features(dataset, tokenizer):
    features = []
    label_map = {label: i for i, label in enumerate(config.label_list)}
    max_seq_length = config.max_seq_length
    for example in tqdm(dataset):

        label = [0]*len(config.label_list)
        label[example.label] = 1

        # for i, word in enumerate(example_text):
        #     token = tokenizer.tokenize(word)
        #     tokens.extend(token)
        #     ori_tokens.append(word)

        if len(example.text) >= max_seq_length - 8:
            # -2的原因是因为序列需要加一个句首和句尾标志
            example.text = example.text[0:(max_seq_length - 9)]

        example.text = ["[CLS"]+list(example.text) + ['[MASK]']+["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(example.text)
        attention_mask = [1]*len(example.text)
        masked_index = example.text.index("[MASK]")

        if len(input_ids) < max_seq_length:
            input_ids.extend([0]*(max_seq_length-len(input_ids)))
            attention_mask.extend([0]*(max_seq_length-len(attention_mask)))

        token_type_ids = [0]*max_seq_length

        assert(len(input_ids) == len(attention_mask) ==
                len(token_type_ids) == max_seq_length)

        features.append(InputFeatures(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        label_id=label,
                                        ori_label=example.label,
                                        masked_index=masked_index))

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)
    all_ori_labels = torch.tensor(
        [f.ori_label for f in features], dtype=torch.long)
    all_masked_index = torch.tensor(
        [f.masked_index for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_token_type_ids,
                            all_attention_mask, all_label_ids, all_ori_labels, all_masked_index)
    return features, data


def get_labels(config: Config):
    """
    读取训练数据获取标签
    :param config:
    :return:
    """
    label_map = {label: i for i, label in enumerate(config.label_list)}
    label_pkl_path = "label_list.pkl"
    save_pkl(label_map, label_pkl_path)
    if os.path.exists(label_pkl_path):
        logging.info(f"loading labels info from {label_pkl_path}")
        labels = load_pkl(label_pkl_path)
        print(labels)
    else:
        logging.info(
            f"loading labels info from train file and dump in {config.output_path}")
        tokens_list = load_file(config.train_file, separater=config.sep)
        labels = set([tokens[1] for tokens in tokens_list if len(tokens) == 2])

    if len(labels) == 0:
        ValueError("loading labels error, labels type not found in data file: {}".format(
            config.output_path))
    else:
        save_pkl(labels, label_pkl_path)

    return labels


def get_label2id_id2label(output_path, label_list):
    """
    获取label2id、id2label的映射
    :param output_path:
    :param label_list:
    :return:
    """
    label2id_path = "label2id.pkl"
    if os.path.exists(label2id_path):
        label2id = load_pkl(label2id_path)
    else:
        label2id = {l: i for i, l in enumerate(label_list)}
        save_pkl(label2id, label2id_path)

    id2label = {value: key for key, value in label2id.items()}
    return label2id, id2label
