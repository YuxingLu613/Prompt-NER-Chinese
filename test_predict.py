import requests
url = "http://127.0.0.1:5000/predict_ner"
data = {
    'input':'我得了肠胃炎，现在要去做穿肠手术。我昨天吃了二甲双胍，今天准备去拍CT。三个月前我被切除了胃和肾脏，现在觉得特别空虚，希望能做一个白细胞检查。',
    }
response = requests.post(url,data=data)

print(response.text)