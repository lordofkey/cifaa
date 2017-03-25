# coding: utf-8

import tensorflow as tf
import threading
import numpy as np
import time
import data_input
import Queue
import modelmanage
import os

class trainmodel:
    def __init__(self, clsnum):
        self.clsnum = clsnum
        self.modeldir = "./temp/"
        self.inputq = Queue.Queue(10)  #进样队列
        self.istraining = False
        self.thread = None

    def starttrain(self):
        if(self.istraining):
            raise Exception("正在进行训练！")
        self.isrunning = True  # 是否继续工作
        self.istraining = True  # 任务占用中
        self.thread = threading.Thread(target=self.threadfunc)
        self.thread.setDaemon(True)
        self.thread.start()

    def stptrain(self):
        if self.istraining == False:
            raise Exception("无正在运行的任务")
        self.isrunning = False
        if self.thread is None:
            raise Exception("没有可停止的训练任务")
        self.thread.join()

    def threadinput(self):
        datagen = False
        while(self.isrunning):
            if not datagen:
                data = data_input.data_input()
                datagen = True
            try:
                self.inputq.put(data, False, 0.001)
                datagen = False
            except Queue.Full:
                continue

    def interface(self, clsnum):
        self.x = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name="x_data")
        self.y = tf.placeholder(tf.int64, shape=(None), name="y_data")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        '''
        curlay = self.conv(self.x, 64, 3)
        curlay = self.conv(curlay, 64, 3)
        curlay = self.pool(curlay, 2, 2)

        curlay = self.conv(curlay, 128, 3)
        curlay = self.conv(curlay, 128, 3)
        curlay = self.pool(curlay, 2, 2)

        curlay = self.conv(curlay, 256, 3)
        curlay = self.conv(curlay, 256, 3)
        curlay = self.conv(curlay, 256, 3)
        curlay = self.pool(curlay, 2, 2)

        curlay = self.conv(curlay, 512, 3)
        curlay = self.conv(curlay, 512, 3)
        curlay = self.conv(curlay, 512, 3)
        curlay = self.pool(curlay, 2, 2)

        curlay = self.conv(curlay, 512, 3)
        curlay = self.conv(curlay, 512, 3)
        curlay = self.conv(curlay, 512, 3)
        curlay = self.pool(curlay, 2, 2)

        curlay = self.fullconnect(curlay, 4096, "relu")
        curlay = tf.nn.dropout(curlay, keep_prob=self.keep_prob)

        curlay = self.fullconnect(curlay, 1024, "relu")
        curlay = tf.nn.dropout(curlay, keep_prob=self.keep_prob)
        '''

        curlay = self.conv(self.x, 96, 11, 4)
        curlay = self.pool(curlay, 3, 2)
#        curlay = tf.nn.moments(curlay,[0], keep_dims=True)
        curlay = self.conv(curlay, 256, 5)
        curlay = self.pool(curlay, 3, 2)
#        curlay = tf.nn.moments(curlay,[0], keep_dims=True)
        curlay = self.conv(curlay, 384, 3)
        curlay = self.pool(curlay, 3, 3)
        curlay = self.fullconnect(curlay, 512, "relu")
        curlay = tf.nn.dropout(curlay, self.keep_prob)
        curlay = self.fullconnect(curlay, 512, "relu")
        curlay = tf.nn.dropout(curlay, self.keep_prob)
        curlay = self.fullconnect(curlay, clsnum)
        return curlay

    def conv(self,lastlayer, fenum, kersize, stride = 1):
        lastfnum = lastlayer.get_shape()[3].value
        shape = [kersize, kersize, lastfnum, fenum]
        kerl = tf.Variable(tf.truncated_normal(shape, stddev=0.001))
        bias = tf.Variable(tf.zeros([fenum]))
        conv = tf.nn.conv2d(lastlayer, kerl, strides=[1, stride, stride, 1], padding='SAME')+bias
        rell = tf.nn.relu(conv)
        return rell

    def pool(self, lastlayer, kersize, stride):
        return tf.nn.max_pool(lastlayer, ksize=[1, kersize, kersize, 1], strides=[1, stride, stride, 1], padding="SAME")

    def fullconnect(self, lastlayer, outputn, activation=""):
        input_shape = lastlayer.get_shape().as_list()
        n_inputs = int(np.prod(input_shape[1:]))
        resh = tf.reshape(lastlayer, [-1, n_inputs])
        kerl = tf.Variable(tf.truncated_normal(shape=[n_inputs, outputn], stddev=0.1))
        bias = tf.Variable(tf.zeros([outputn]))
        out = tf.matmul(resh, kerl)+bias
        if(activation=="relu"):
            out = tf.nn.relu(out)
        return out

    def cleartmpdir(self):
        filelist = os.listdir(self.modeldir)
        for ff in filelist:
            os.remove(self.modeldir+ff)
        return

    def threadfunc(self):
        gra = tf.Graph()
        self.cleartmpdir()
        with gra.as_default():
            y = self.interface(self.clsnum)    #构建网络
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=y))
            global_step = tf.Variable(0,trainable=False)
            pred = tf.arg_max(y, 1, name="pred")
            preper = tf.nn.softmax(y, name="per")
            optimazer = tf.train.AdadeltaOptimizer().minimize(loss=loss, global_step=global_step)
            correct_prediction = tf.equal(pred, self.y)
            accur = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accur")
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            ex, ey = data_input.alldata_input()
            inputthreads = []
            for i in range(16):
                tm = threading.Thread(target=self.threadinput)
                tm.setDaemon(True)
                tm.start()
                inputthreads.append(tm)


            stime = time.time()
            with tf.Session() as sess:
                sess.run(init_op)
                for i in range(2000):
                    if not (self.isrunning):
                        break
                    try:
                        x_data, y_data = self.inputq.get(block=False, timeout=0.001)
                    except Queue.Empty:
                        continue
                    sess.run(optimazer, feed_dict={self.x: x_data, self.y: y_data, self.keep_prob: 0.5})
                    if i%10 == 0:
                        ll = sess.run([loss, accur], feed_dict={self.x: x_data, self.y: y_data, self.keep_prob:1.0})
                        print ll
                gra.add_to_collection(name="x_", value=self.x)
                gra.add_to_collection(name="y_", value=self.y)
                gra.add_to_collection(name="acc", value=accur)
                gra.add_to_collection(name="prob", value=self.keep_prob)
                saver.save(sess, self.modeldir+"tmp", global_step=global_step)
            traintime = (time.time() - stime)/3600
            trainresult = {"last": str(traintime), "time": time.asctime(time.localtime()), "loss": str(ll[0]), "accurency": str(ll[1])}
            modelmanage.addmodel(self.modeldir, trainresult)
            self.isrunning = False
            for i in range(16):
                inputthreads[i].join()
            self.istraining = False
            print "succeed saved"

if __name__ == "__main__":
    a = trainmodel(2)
    a.starttrain()
    time.sleep(30)
    a.stptrain()
    del a