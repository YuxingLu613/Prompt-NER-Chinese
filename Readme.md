# Prompt-NER on Chinese EHR data

### 简介

该仓库使用[BERT](https://github.com/google-research/bert)作为预训练模型使用Prompt预训练方法进行命名实体识别任务。



### 文件夹介绍：

```
.
├── .DS_Store
├── __init__.py
├── app.py 	# 接口文件
├── best_model.pth	# 模型文件（需要自己训练）
├── config.py	# 配置文件
├── conlleval.py	# 评价指标
├── data	# 数据集
│   ├── eval.txt # 处理好的验证集
│   ├── test.txt	# 处理好的测试集
│   └── train.txt	# 处理好的训练集
├── logger.py	# 日志文件
├── main.py	# 主文件
├── output	# 输出
│   └── logs
│       └── Experiment_log.log
├── predict.py	# 预测程序
├── processer.py	# 数据预处理文件
├── prompt_model.py	# 模型结构
├── test_predict.py	# 接口测试文件
└── utils.py	# 方法函数
```



### Requirements:

python

torch

sklearn

pandas

transformers



### 直接使用方法：

使用python运行main.py，获得模型文件。

使用python运行app.py，并且在test.py中修改input数据，获得返回的结果。



### 结果：

| 训练方法       | F1     |
| -------------- | ------ |
| 常规预训练方法 | 0.7617 |
| Prompt训练方法 | 0.8189 |