import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import copy


def x2list(x):
    out = []
    for i in x:
        out.append([i])
    return out


def x2t(x):
    xtrain = []
    ytrain = []
    for i in range(len(x)-time_step):
        xtrain.append(x2list(x[i:i+time_step]))
        ytrain.append(x[i+time_step:i+time_step+1])
    return xtrain, ytrain


def lstm(X, reuse=False):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    with tf.variable_scope('lstm') as scope:
        w_in = tf.get_variable(
            'win', initializer=tf.random_normal([input_size, rnn_unit]))
        b_in = tf.get_variable(
            'bin', initializer=tf.constant(0.1, shape=[rnn_unit, ]))
        w_out = tf.get_variable(
            'wout', initializer=tf.random_normal([rnn_unit, 1]))
        b_out = tf.get_variable(
            'bout', initializer=tf.constant(0.1, shape=[1, ]))
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in)+b_in
    # 将tensor转成3维，作为lstm cell的输入
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = rnn.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    if not reuse:
        with tf.variable_scope('lstm') as scope:
            # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
            output_rnn, final_states = tf.nn.dynamic_rnn(
                cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    else:
        with tf.variable_scope('lstm'):
            output_rnn, final_states = tf.nn.dynamic_rnn(
                cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    pred = tf.matmul(final_states[1], w_out)+b_out
    return pred, final_states


def batch_generate(x_train, y_train, steps):
    xtrain = []
    ytrain = []
    for step in steps:
        xtrain.append(x_train[step])
        ytrain.append([y_train[step]])
    return xtrain, ytrain

#——————————————————训练模型——————————————————


def train_lstm(x_train, y_train, x_test, xtest):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, output_size, output_size])
    pred, _ = lstm(X)
    loss = tf.reduce_mean(
        tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            indexs = random.sample(range(len(y_train)), batch_size)
            x_train_in, y_train_in = batch_generate(x_train, y_train, indexs)
            _, loss_ = sess.run([train_op, loss], feed_dict={
                                X: x_train_in, Y: y_train_in})
            print(i, loss_)
        end = time_step+1
        prediction = []
        while end <= len(xtest):
            x_test_in = [x_test[end-time_step-1]]
            pred_this = sess.run([pred], feed_dict={X: x_test_in})
            prediction.extend(pred_this[0][0])
            end += 1
        nonprediction = list2num(x_test[0])
        predline = nonprediction+prediction
        sum_p = 0
        for i in range(len(xtest)):
            print(predline[i], xtest[i])
            sum_p = sum_p+(predline[i]-xtest[i])**2
        sum_x = sum(map(lambda x: x**2, xtest))
        accury = sum_p/sum_x
        print('accury is %s' % accury)
        plt.figure()
        a = list(range(len(xtest)))
        plt.plot(a, xtest, color='b')
        plt.plot(a, predline, color='r')
        plt.show()
        print('Successful')


def list2num(x):
    out = []
    for i in x:
        out.append(i[0])
    return out


rnn_unit = 10  # hidden layer units
input_size = 1
output_size = 1
time_step = 20
batch_size = 50
lr = 0.01
ftrain = open('train_prediction.txt')
xtrain = ftrain.read()
xtrain = xtrain.split()
ftest = open('test_prediction.txt')
xtest = ftest.read()
xtest = xtest.split()
for i in range(len(xtrain)):
    xtrain[i] = float(xtrain[i])
for i in range(len(xtest)):
    xtest[i] = float(xtest[i])
x_train, y_train = x2t(xtrain)
x_test, _ = x2t(xtest)

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit]), name='w1in'),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]), name='w1out')
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name='b1in'),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='b1out')
}


train_lstm(x_train, y_train, x_test, xtest)
