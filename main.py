from transformers import BertForMaskedLM, BertTokenizer,AdamW, get_linear_schedule_with_warmup
from processer import *
import os
import numpy as np
import pandas as pd
from logger import logger as logging
from config import Config
from prompt_model import Prompt_Based_NER,evaluate
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from tqdm import trange,tqdm
import torch
import random
from utils import save_pkl,load_pkl
from torch.optim import Adam
from torch import nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

config=Config()

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

CURRENT_PATH=os.getcwd()
TRAIN_PATH=os.path.join(CURRENT_PATH,'data/train.txt')
EVAL_PATH=os.path.join(CURRENT_PATH,"data/eval.txt")
TEST_PATH=os.path.join(CURRENT_PATH,"data/test.txt")

use_gpu = torch.cuda.is_available() and config.use_gpu
device = torch.device('cuda' if use_gpu else config.device)
config.device = device
n_gpu = torch.cuda.device_count()
logging.info(f"available device: {device}，count_gpu: {n_gpu}")

tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
logging.info(f"=================Tokenizer {config.model_name_or_path} Load Successfully=================")

label_list = get_labels(config=config)
config.label_list = label_list
num_labels = len(label_list)
logging.info(f"loading labels successful! the size is {num_labels}, label is: {','.join(list(label_list))}")

label2id, id2label = get_label2id_id2label(config.output_path, label_list=label_list)
logging.info("loading label2id and id2label dictionary successful!")

if config.do_train:
    logging.info(f"=================Start Loading Train Dataset=================")
    train_text,train_label=dataset_format(TRAIN_PATH,separater=" ")
    assert(len(train_text)==len(train_label))
    train_length_list=[len(i) for i in train_text]
    logging.info(f"Discription of Train Dataset: {pd.DataFrame(train_length_list).describe()}")

    logging.info(f"=================Start Creating Train Templates=================")
    train_dataset=create_template(train_text,train_label)

    logging.info(f"=================Preview 3 Examples=================")
    for i in range(5):
        print(train_dataset[i].text+"[MASK]",train_dataset[i].label)

    logging.info(f"=================Convert Train Examples=================")
    # if os.path.exists("train_dataset.pkl"):
    #     train_dataset=load_pkl("train_dataset.pkl")
    # else:
    train_features,train_dataset=convert_examples_to_features(train_dataset,tokenizer)
    save_pkl(train_dataset,"train_dataset.pkl")
    train_dataloader=DataLoader(train_dataset,batch_size=config.train_batch_size,sampler=RandomSampler(train_dataset))

if config.do_eval:
    logging.info(f"=================Start Loading Eval Dataset=================")
    eval_text,eval_label=dataset_format(EVAL_PATH,separater=" ")
    assert(len(eval_text)==len(eval_label))
    
    logging.info(f"=================Start Creating Eval Templates=================")
    eval_dataset=create_template(eval_text,eval_label)

    logging.info(f"=================Convert Eval Examples=================")
    # if os.path.exists("eval_dataset.pkl"):
    #     eval_dataset=load_pkl("eval_dataset.pkl")
    # else:
    eval_features,eval_dataset=convert_examples_to_features(eval_dataset,tokenizer)
    save_pkl(eval_dataset,"eval_dataset.pkl")
    eval_dataloader=DataLoader(eval_dataset,batch_size=config.eval_batch_size,sampler=SequentialSampler(eval_dataset))

if config.do_test:
    logging.info(f"=================Start Loading Test Dataset=================")
    test_text,test_label=dataset_format(TEST_PATH,separater=" ")
    assert(len(test_text)==len(test_label))

    logging.info(f"=================Start Creating Test Templates=================")
    test_dataset=create_template(test_text,test_label)

    logging.info(f"=================Convert Test Examples=================")
    if os.path.exists("test_dataset.pkl"):
        test_dataset=load_pkl("test_dataset.pkl")
    else:
        test_features,test_dataset=convert_examples_to_features(test_dataset,tokenizer)
        save_pkl(test_dataset,"test_dataset.pkl")
    test_dataloader=DataLoader(test_dataset,batch_size=config.eval_batch_size,sampler=SequentialSampler(test_dataset))

model = Prompt_Based_NER(config).to(device)

if use_gpu and n_gpu > 1:
    model = torch.nn.DataParallel(model)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
]
optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)

t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=t_total)
CrossEntropyloss=nn.CrossEntropyLoss()

logging.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

logging.info("====================== Running training ======================")
logging.info(
    f"Num Examples:  {len(train_dataset)}, Num Batch Step: {len(train_dataloader)}, "
    f"Num Epochs: {config.num_train_epochs}, Num scheduler steps: {t_total}")

model.train()
global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
for ep in trange(int(config.num_train_epochs), desc="Epoch"):
    logging.info(f"#######[Epoch: {ep}/{int(config.num_train_epochs)}]#######")
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="DataLoader")):
        if step%100==0:
            logging.info(f"####[Step: {step}/{len(train_dataloader)}]####")

        batch = tuple(t.to(device) for t in batch)
        input_ids, token_type_ids, attention_mask, label_ids,ori_tokens,mask_index = batch

        outputs = model(input_ids, token_type_ids, attention_mask,mask_index)
        loss = CrossEntropyloss(outputs,label_ids)

        if use_gpu and n_gpu > 1:
            # mean() to average on multi-gpu.
            loss = loss.mean()

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        # 反向传播
        loss.backward()
        tr_loss += loss.item()

        # 优化器_模型参数的总更新次数，和上面的t_total对应
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # 更新参数
            optimizer.step()
            scheduler.step()
            # 梯度清零
            model.zero_grad()
            global_step += 1

        if global_step%config.logging_steps == 0:
            tr_loss_avg = tr_loss  / config.logging_steps
            tr_loss=0
            logging.info(f'tr_loss_avg= {tr_loss_avg}')

    if config.do_eval:
        logging.info("====================== Running Eval ======================")
        all_ori_tokens_eval = [token for token in list("".join(eval_text))]
        overall, by_type = evaluate(config, eval_dataloader, model, id2label, all_ori_tokens_eval)

        f1_score = overall.fscore

        # save the best performs model
        if f1_score > best_f1:
            logging.info(f"******** the best f1 is {f1_score}, save model !!! ********")
            best_f1 = f1_score
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, 'module') else model
        
            torch.save(model.state_dict(),config.output_path+"/best_model.pth")

            net_dict = model.state_dict()
            best_model = torch.load(config.output_path+"/best_model.pth")
            for k, v in best_model.items():
                name = k[7:] # remove `module.`
                net_dict[name] = v
            
            torch.save(net_dict,"best_model.pth")
            # model_to_save.save_pretrained(config.output_path)
            tokenizer.save_pretrained(config.output_path)


            # Good practice: save your training arguments together with the trained model
            torch.save(config, os.path.join(config.output_path, 'training_config.bin'))
            torch.save(model, os.path.join(config.output_path, 'ner_model.ckpt'))
            logging.info("training_args.bin and ner_model.ckpt save successful!")
logging.info("NER Prompt model training successful!!!")
logging.info(f"Best F1 is {best_f1}!")


if config.do_test:

    all_ori_tokens_test = [token for token in list("".join(test_text))]
    overall, by_type = evaluate(config, test_dataloader, model, id2label, all_ori_tokens_test)

    logging.info("====================== Running test ======================")
    logging.info(f"Num Examples:  {len(test_dataset)}, Batch size: {config.eval_batch_size}")
    f1_score = overall.fscore
    logging.info(f"**********Test F1 is {f1_score}")

    model.eval()

    pred_labels = []
    ori_labels=[]

    for b_i, (input_ids, token_type_ids, attention_mask, label_ids,ori_tokens,masked_index) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        token_type_ids = token_type_ids.to(config.device)
        label_ids = label_ids.to(config.device)
        masked_index=masked_index.to(config.device)
        with torch.no_grad():
            output = model.forward(input_ids, token_type_ids, attention_mask,masked_index)                
            logits=torch.argmax(output,dim=-1)

        for l in logits:
            pred_labels.append([id2label[l.item()]])

        for l in label_ids:
            ori_labels.append([id2label[l.tolist().index(1)]])

        for l in logits:
            pred_label = []
            for idx in l[1:]:
                pred_label.append(id2label[idx])
            pred_labels.append(pred_label)

    assert len(pred_labels) == len(ori_tokens) == len(ori_labels)

    with open(os.path.join(config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
        for ori_tokens, ori_labels, prel in zip(ori_tokens, ori_labels, pred_labels):
            for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                if ot in ["[CLS]", "[SEP]"]: 
                    continue
                else:
                    f.write(f"{ot} {ol} {pl}\n")
            f.write("\n")


