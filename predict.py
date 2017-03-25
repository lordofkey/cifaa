#coding: utf-8
import tensorflow as tf
import Queue
import data_input
import time
import threading

class predictmedel:
    def __init__(self):
        self.isreloadmodel = False
        self.inputQ = Queue.Queue(100)
        self.sess = None
        self.thread = threading.Thread(target=self.threadfunc)
        self.thread.setDaemon(True)
        self.thread.start()

    def threadfunc(self):
        while(True):
            if(self.isreloadmodel):
                self.loadmodelt()
            if self.sess == None:
                time.sleep(1)
                continue
            try:
                inputd = self.inputQ.get(block=False)
            except:
                continue
            ex, ey = inputd["data"]
            predictions = self.sess.run(self.accur, feed_dict={self.x_data: ex, self.y_data: ey, self.prob: 1.0})
            inputd["Q"].put(predictions, block=False)

    def loadmodelt(self):                       #线程内函数
        ckpt = tf.train.get_checkpoint_state("./models/"+self.modelname)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        self.x_data = tf.get_collection("x_")[0]
        self.y_data = tf.get_collection("y_")[0]
        self.accur = tf.get_collection("acc")[0]
        self.prob = tf.get_collection("prob")[0]
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        self.sess = tf.Session()
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.isreloadmodel = False

    def loadmodel(self, name):
        self.modelname = name
        self.isreloadmodel = True

    def predict(self):
        predictmess = {}
        predictmess["data"] = data_input.alldata_input()
        predictmess["Q"] = Queue.Queue(1)
        self.inputQ.put(predictmess)
        try:
            result = predictmess["Q"].get(block=True, timeout= 2)
        except:
            raise Exception("系统忙，或无模型加载")
        return result

if __name__ == "__main__":
    tt = predictmedel()
    tt.loadmodel("79609ace10ff11e785a28cbebe005dd7")
    time.sleep(15)
    print tt.predict()
    print tt.predict()
    print tt.predict()
    print tt.predict()










