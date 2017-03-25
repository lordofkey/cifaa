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
            pack[name] = json.load(fp, encoding='utf-8')
    return str(pack)

def addmodel(src, trainresult):
    path = MODELPATH+uuid.uuid1().get_hex()
    os.mkdir(path)
    flist = os.listdir(src)
    for ff in flist:
        shutil.move(src+ff, path)
    pf = open(path+"/setting.json", "w")
    json.dump(trainresult, pf, ensure_ascii=False)
    pf = open(path+"/ok.txt", "w")

if __name__=="__main__":
    print getmodels()
    addmodel()