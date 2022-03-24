from re import sub
from select import select
from transformers import BertForMaskedLM,AutoConfig,AutoModel,AutoModelForMaskedLM,BertForTokenClassification
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import conlleval

class Prompt_Based_NER(nn.Module):

    def __init__(self, config):
        super(Prompt_Based_NER, self).__init__()
        self.config=config
        self.plm_config=AutoConfig.from_pretrained(self.config.model_name_or_path)
        # self.plm_model=BertForMaskedLM.from_pretrained("bert-base-chinese")
        self.plm_model=BertForMaskedLM.from_pretrained(self.config.model_name_or_path)
        self.dropout=nn.Dropout(self.plm_config.hidden_dropout_prob)
        self.linear1=nn.Linear(in_features=self.plm_config.vocab_size,out_features=256)
        self.linear2=nn.Linear(in_features=256,out_features=len(self.config.label_list))
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,masked_index=None):
        masked_index=masked_index.to(self.config.device)
        output=self.plm_model(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)[0]
        selected_output=None
        for index, sub_output in enumerate(output):
            if selected_output is None:
                selected_output=sub_output[masked_index[index]].unsqueeze(0)
            else:
                selected_output=torch.cat((selected_output,sub_output[masked_index[index]].unsqueeze(0)),dim=0)
        output=selected_output
        output=self.linear1(output)
        output=self.dropout(output)
        output=self.linear2(output)
        return output

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        output=self.plm_model(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask).logits
        output=self.hidden2tag(output)
        output=torch.argmax(output,dim=-1)
        return output

def evaluate(config, dataloader, model, id2label, all_ori_tokens):
        ori_labels, pred_labels = [], []
        model.eval()
        
        for b_i, (input_ids, token_type_ids, attention_mask, label_ids,ori_tokens,masked_index) in enumerate(tqdm(dataloader, desc="Evaluating")):
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

        eval_list = []
        for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
            for ot, ol, pl in zip(ori_tokens, oril, prel):
                if ot in ["[CLS]", "[SEP]"]:
                    continue
                eval_list.append(f"{ot} {ol} {pl}\n")
            eval_list.append("\n")
        
        # eval the model
        counts = conlleval.evaluate(eval_list)
        conlleval.report(counts)

        # namedtuple('Metrics', 'tp fp fn prec rec fscore')
        overall, by_type = conlleval.metrics(counts)
        return overall, by_type