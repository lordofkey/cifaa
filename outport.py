#coding: utf-8
from flask import Flask
from flask import request
import trainmodel
import json
import modelmanage
import predict
app = Flask(__name__)


pre = predict.predictmedel()
tmodel = trainmodel.trainmodel(2)

def checkpar(batchsize, epochnum, classes):
    if(batchsize <= 2):
        raise Exception("assert： batchsize>2")
    if batchsize >= 30:
        raise Exception("assert： batchsize<30")
    if epochnum <= 10:
        raise Exception("assert: epochnum>10")
    if len(classes) < 2:
        raise Exception("assert: 至少一个类别")

def getstatue():
    pass

def deletemodel(modelname):
    pass

def updatemodel(modelname):
    pass

def recog(filepath):
    pass

@app.route('/trainPicModel', methods=["POST"])
def begin_train():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "训练任务开始"
    batchsize = int(request.form['BatchSize'])
    epochnum = int(request.form['EpochNum'])
    classes = request.form['Classes']
    classj = json.loads(classes)
    try:
        checkpar(batchsize, epochnum, classj)
        tmodel.starttrain()
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    return json.dumps(result, ensure_ascii=False)

@app.route('/stopPicModelTrain', methods=["POST", "GET"])
def stoptrain():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功，停止训练"
    try:
        tmodel.stptrain()
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    return json.dumps(result, ensure_ascii=False)

@app.route('/queryPicSystemStatus', methods=["POST"])
def querystatus():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功"
    statue = getstatue()
    return json.dumps(result, ensure_ascii=False)

@app.route('/getAllPicModelInfo', methods=["GET", "POST"])
def getAllPicModelInfo():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功"
    try:
        modelinfo = modelmanage.getmodels()
        result["info"] = modelinfo
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    print result
    return json.dumps(result, ensure_ascii=False)

@app.route('/deletePicModel', methods=["POST"])
def deletePicModel():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功"
    modelname = request.form["modelname"]
    try:
        deletemodel(modelname)
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    return json.dumps(result, ensure_ascii=False)

@app.route('/updatePicModel', methods=["POST"])
def undatePicModel():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功"
    modelname = request.form["modelname"]
    try:
        pre.loadmodel(modelname)
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    return json.dumps(result, ensure_ascii=False)

@app.route('/recognizePic', methods=["POST", "GET"])
def recognizePic():
    result = {}
    result["ResultCode"] = "success"
    result["ResultMessage"] = "操作成功"
    filepath = request.form["PicFilePath"]
    try:
        result["resultinfo"] = str(pre.predict())
    except Exception as e:
        result["ResultCode"] = "failed"
        result["ResultMessage"] = e.message
    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run()