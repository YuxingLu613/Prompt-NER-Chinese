from multiprocessing.spawn import import_main_path
from flask import Flask,jsonify,request
from pip import main
from predict import build_model, predict


app=Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict_ner',methods=["POST"])
def predict_ner():
    global model
    input=request.form.get("input")
    output=predict(input,model)
    return jsonify(output=output)

if __name__=="__main__":
    model=build_model()
    app.run(host="127.0.0.1")