from prompt_model import Prompt_Based_NER
import torch
from torch import nn
from config import Config
import os
from processer import get_label2id_id2label
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader,SequentialSampler,TensorDataset
from prompt_model import Prompt_Based_NER

def build_model():
    config=Config()
    # model_path=os.path.join(config.base_path,"output","20220306205222","best_model.pth")
    model=Prompt_Based_NER(config)
    # net_dict = model.state_dict()
    # best_model = torch.load(model_path)
    # for k, v in best_model.items():
    #     name = k[7:] # remove `module.`
    #     net_dict[name] = v
    # model.load_state_dict(net_dict)
    # model.to(device)
    # torch.save(model.state_dict(),"best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))

    return model

def predict(input,model):
    config=Config()

    use_gpu = False
    device = torch.device('cuda' if use_gpu else config.device)
    config.device = device

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    
    label2id, id2label = get_label2id_id2label(config.output_path, label_list=config.label_list)
    
    def get_input_sequence(input):
        input_split=input.split("。")
        input_list=[]
        max_len=config.max_seq_length-6
        for input in input_split:
            if input:
                input+="。"
            while len(input)>0:
                input_list.append(input[:max_len])
                input=input[max_len:]
        return input_list

    class Template():
        def __init__(self,text) -> None:
            self.text=text

    def create_template(input_list):
        dataset=[]
        for text_list in input_list:
            for word in list(text_list):
                template=Template(text=text_list+"，"+word+"是")
                dataset.append(template)
        return dataset

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self, input_ids, token_type_ids, attention_mask, masked_index):
            """
            :param input_ids:       单词在词典中的编码
            :param attention_mask:  指定 对哪些词 进行self-Attention操作
            :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
            :param label_id:        标签的id
            """
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_mask = attention_mask
            self.masked_index = masked_index

    def convert_examples_to_features(dataset,tokenizer):
        features=[]
        max_seq_length = config.max_seq_length
        for example in tqdm(dataset):
            if len(example.text) >= max_seq_length - 3:
                # -2的原因是因为序列需要加一个句首和句尾标志
                example.text = example.text[0:(max_seq_length - 3)]
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
                                            masked_index=masked_index))

        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        all_masked_index = torch.tensor(
            [f.masked_index for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_token_type_ids,
                                all_attention_mask, all_masked_index)

        return features, data

    input_list=get_input_sequence(input)
    dataset=create_template(input_list)

    predict_features, predict_dataset=convert_examples_to_features(dataset,tokenizer)
    predict_dataloader=DataLoader(predict_dataset,batch_size=config.eval_batch_size,sampler=SequentialSampler(predict_dataset))

    model.eval()

    pred_labels = []

    for b_i, (input_ids, token_type_ids, attention_mask,masked_index) in enumerate(tqdm(predict_dataloader, desc="Predicting")):
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        token_type_ids = token_type_ids.to(config.device)
        masked_index=masked_index.to(config.device)
        with torch.no_grad():
            output = model.forward(input_ids, token_type_ids, attention_mask,masked_index)                
            logits=torch.argmax(output,dim=-1)

        for l in logits:
            pred_labels.append([id2label[l.item()]])

    output=[]

    for word,label in zip(list("".join(input_list)),pred_labels):
        output.append((word,label[0]))
    print(output)
    return output


if __name__=="__main__":
    model=build_model()
    input="我得了肠胃炎，现在要去做穿肠手术。我昨天吃了二甲双胍，今天准备去拍CT。三个月前我被切除了胃和肾脏，现在觉得特别空虚，希望能做一个白细胞检查。"
    for word,label in predict(input,model):
        print(word,label)


