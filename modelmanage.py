#coding:utf-8
import os
import json
import uuid
import shutil

MODELPATH = "./models/"

def getmodels():
    modelnames = os.listdir(MODELPATH)
    pack = {}
    for name in modelnames:
        if os.path.isfile(MODELPATH+name+"/ok.txt"):
            fp = open(MODELPATH+name+"/setting.json")
            setm = json.load(fp, encoding='utf-8')
            pack[name] = setm
    return pack

def addmodel(src, trainresult):
    path = MODELPATH+uuid.uuid1().get_hex()
    os.mkdir(path)
    flist = os.listdir(src)
    for ff in flist:
        shutil.move(src+ff, path)
    pf = open(path+"/setting.json", "w")
    print trainresult
    json.dump(trainresult, pf, ensure_ascii=False)
    pf.close()
    pf = open(path+"/ok.txt", "w")
    pf.close()

def deletemodel(modelname):
    try:
        shutil.rmtree(MODELPATH+modelname)
    except:
        raise Exception("no model named "+ modelname)

if __name__=="__main__":
    trainresult = {"last": 334.345, "time": "sf2342", "loss": 34.591805,
                   "accurency": 2.23}
    addmodel("./temp", trainresult)

    result = {}
    result[u"ResultCode"] = u"success"
    result[u"ResultMessage"] = u"操作成功"
    try:
        modelinfo = getmodels()
        result[u"info"] = modelinfo
    except Exception as e:
        result[u"ResultCode"] = u"failed"
        result[u"ResultMessage"] = e.message
    print result
    print json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)