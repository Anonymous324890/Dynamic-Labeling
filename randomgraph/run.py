#-*-coding:utf-8-*-
import sys
# sys.path.append('..')
import pickle
from Model import *
from resolve_data import *
import os
import tensorflow as tf
import numpy as np
import os
import math
import queue as Q
from copy import deepcopy
#from tqdm import tqdm

#os.envireni
#
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def create_model(session, g, d):
    if(os.path.exists("./save1")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("./save1/"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer(), feed_dict={})
        print("create a new model")

def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session, "save" + str(number) + "/model.cpkt")

def g_train(sess, model, batch):
    # batch_data 
    # 9-gram: NL, NL-charlist, Tree, Father, Grandfather, tree-path, predicted rules, target, functions/classes

    #batch = batch_data
    _, loss, accuracy = sess.run([model.optim, model.loss, model.accuracy], feed_dict={
                                                  model.input_list:batch[1],
                                                  model.input_mask:batch[2],
                                                  model.input_D:batch[3],
                                                  model.input_mat:batch[4],
                                                  model.input_y:batch[5],
                                                  model.keep_prob: 0.5,
                                                  model.is_train: True
                                                  })
    return loss, accuracy

def g_eval(sess, model, batch, trees):
    fprob = None
    retpre = []
    for tree in range(trees):
        l = []
        for t in batch[0]:
            mat = np.zeros([var_len])
            perm = np.random.permutation(t)
            for i in range(min(len(mat), len(perm))):
                mat[i] = perm[i] + 1
            l.append(mat)
        batch[1] = np.array(l)
        acc, pre, pre_rules, prob = sess.run([model.accuracy, model.correct_prediction, model.max_res, model.y], feed_dict={
                                                    model.input_list:batch[1],
                                                    model.input_mask:batch[2],
                                                    model.input_D:batch[3],
                                                    model.input_mat:batch[4],
                                                    model.input_y:batch[5],
                                                    model.keep_prob: 1,
                                                    model.is_train: False
                                                })  
        try:
            if fprob == None:
                fprob = prob
        except:
            pass
        fprob += prob# np.maximum(fprob, prob)
        #fprob = np.maximum(fprob, prob)
        #if tree % 5 == 0:
        pre = []

        for i in range(len(fprob)):
            t = np.argmax(fprob[i])
            if t == batch[5][i]:
                pre.append(1)
            else:
                pre.append(0)
        retpre.append(pre)
    return acc, retpre

def run():
    model = Model(classnum, embedding_size,
                                    batch_size, var_len)
    
    test_batch, _ = batch_data(batch_size, "test") # read data 
    valid_batch, _ = batch_data(batch_size, "dev") # read data 
    
    best_accuracy = 0
    config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True)
    config.gpu_options.allow_growth = True

    f = open("out.txt", "w")

    trees = 10
    with tf.Session(config=config) as sess:
        create_model(sess, model, "")
        print ("------------ Starting Training -----------------")
        for i in tqdm(range(20000)):
            batch, _ = batch_data(batch_size, "train")
            for j in tqdm(range(len(batch))):
                if j == 0 and i % 50 == 0: #eval
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        _, pre = g_eval(sess, model, valid_batch[k], 1)
                        res.extend(pre[0])


                    for k in res:
                        sumac += k

                    ac = sumac / len(res)
                    
                    aac = 0
                    res = [[] for ttt in range(trees)]
              
                    asumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        _, pre = g_eval(sess, model, valid_batch[k], trees)
                        for tree in range(trees):
                            res[tree].extend(pre[tree])
                    list_str = []
                    fstr = "--> "
                    for tree in range(trees):
                        asumac = 0
                        for k in res[tree]:
                            asumac += k
                        aac = asumac / len(res[tree])
                        strs = str(ac) + "-->" + str(aac) + "\n"
                        fstr += str(aac) + " "
                        list_str.append(" " + str(aac) + " ")
                    print("current accuracy " +
                            str(ac) + fstr )
                    open("tree.pkl", "wb").write(pickle.dumps(res))
                    #exit()
                    f.write(str(ac) + fstr)
                    f.flush()
                    if ac > best_accuracy:
                        best_accuracy = ac
                        save_model(sess, 1)
                        print("find the better accuracy " +
                              str(best_accuracy) + "in echos " + str(i))
                #exit()
                g_train(sess, model, batch[j])
                tf.train.global_step(sess, model.global_step)

    f.close()
    print("training finish")
    return

def main():
    #np.set_printoptions(threshold=np.nan)
    # ReadRule()
    if sys.argv[1] == "train":
        #resolve_data()
        run()
    elif sys.argv[1] == "test":
        test()


main()
