curl -d"BatchSize=22&EpochNum=40&Classes={\"0\":\"asdfjw\", \"1\":\"wfe\"}" http://127.0.0.1:5000/trainPicModel

curl -d"BatchSize=44&EpochNum=1&Classes={\"0\":\"asdfjw\", \"1\":\"wfe\"}" http://127.0.0.1:5000/trainPicModel

curl http://127.0.0.1:5000/stopPicModelTrain

curl http://127.0.0.1:5000/getAllPicModelInfo

curl -d"PicFilePath=asdf" http://127.0.0.1:5000/recognizePic

curl -d"modelname=79609ace10ff11e785a28cbebe005dd7" http://127.0.0.1:5000/updatePicModel
