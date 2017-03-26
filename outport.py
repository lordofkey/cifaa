# coding: utf-8

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
    if batchsize <= 2:
        raise Exception("assert： batchsize>2")
    if batchsize >= 30:
        raise Exception("assert： batchsize<30")
    if epochnum <= 10:
        raise Exception("assert: epochnum>10")
    if len(classes) < 2:
        raise Exception("assert: 至少一个类别")


@app.route('/trainPicModel', methods=["POST"])
def begin_train():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"训练任务开始"
    batchsize = int(request.form['BatchSize'])
    epochnum = int(request.form['EpochNum'])
    classes = request.form['Classes']
    classj = json.loads(classes)
    try:
        checkpar(batchsize, epochnum, classj)
        tmodel.starttrain(batchsize, epochnum, classes)
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/stopPicModelTrain', methods=["POST", "GET"])
def stoptrain():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功，停止训练"
    try:
        tmodel.stptrain()
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/queryPicSystemStatus', methods=["POST"])
def querystatus():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
#    statue = getstatue()
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/getAllPicModelInfo', methods=["GET", "POST"])
def getAllPicModelInfo():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
    try:
        modelinfo = modelmanage.getmodels()
        result[u"info"] = modelinfo
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    print result
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/deletePicModel', methods=["POST"])
def deletePicModel():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
    modelname = request.form["modelname"]
    try:
        modelmanage.deletemodel(modelname)
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/updatePicModel', methods=["POST"])
def undatePicModel():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
    modelname = request.form["modelname"]
    try:
        pre.loadmodel(modelname)
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


@app.route('/recognizePic', methods=["POST", "GET"])
def recognizePic():
    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
    filepath = request.form["PicFilePath"]
    try:
        result["resultinfo"] = str(pre.predict())
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message.decode("utf-8")
    return json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    app.run()
