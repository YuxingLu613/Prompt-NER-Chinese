import datetime
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd()))
            self._init_train_config()

    def __new__(cls, *args, **kwargs):
        """
        单例类
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.label_list = ["O","B-手术","I-手术","B-药物","I-药物","B-实验室检验","I-实验室检验","B-影像检查","I-影像检查","B-解剖部位","I-解剖部位","B-疾病和诊断","I-疾病和诊断"]
        self.use_gpu = True
        self.device = "cpu"
        self.sep = " "


        # 输入数据集、输出目录
        self.train_file = os.path.join(self.base_path, 'data', 'train.txt')
        self.eval_file = os.path.join(self.base_path, 'data', 'eval.txt')
        self.test_file = os.path.join(self.base_path, 'data', 'test.txt')
        self.log_path = os.path.join(self.base_path, 'output', "logs")
        self.log_name="Experiment_log.log"
        self.output_path = os.path.join(self.base_path, 'output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        model_list=["bert-base-chinese","hfl/chinese-bert-wwm-ext","hfl/chinese-roberta-wwm-ext","voidful/albert_chinese_base","hfl/chinese-electra-180g-base-discriminator","nghuyong/ernie-1.0"]

        # Pretrained model name or path if not the same as model_name
        # self.model_name_or_path = "/home/luyx/PLM/Continue_Pretrain/training_output"
        self.model_name_or_path = model_list[0]

        # 以下是模型训练参数
        self.do_train = True
        self.do_eval = True
        self.do_test = True
        self.clean = True
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 768
        self.max_seq_length = 64
        self.train_batch_size = 3072
        self.eval_batch_size = 3072
        self.num_train_epochs = 30
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-6
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 100
        self.train_eval_split=0.8
        self.min_sequence_length=10