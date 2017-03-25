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
    json.dump(trainresult, pf, ensure_ascii=False)
    pf = open(path+"/ok.txt", "w")

def deletemodel(modelname):
    try:
        shutil.rmtree(MODELPATH+modelname)
    except:
        raise Exception("no model named "+ modelname)

if __name__=="__main__":
    print getmodels()
    addmodel()