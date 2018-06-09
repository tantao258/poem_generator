import tensorflow as tf
import os
from read_utils import *
import numpy as np

class LSTM(object):
    def __init__(self, sampling=False, n_classes=10, n_steps=28, embedding_size=256, rnn_size=128, n_layers=2,
                 batch_size=128, keep_prob=0.8, grad_clips=5,learning_rate=0.001):

        tf.reset_default_graph()## 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。

        # 构造成员变量
        if sampling is True:
            batch_size, n_steps, keep_prob = 1, 1, 1
        else:
            batch_size, n_steps, keep_prob = batch_size, n_steps, keep_prob

        self.n_classes = n_classes
        self.n_steps = n_steps
        self.embedding_size = embedding_size
        self.lstm_size = rnn_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.grad_clips=grad_clips
        self.learning_rate = learning_rate


        # 成员方法
        self.build_inputs()
        self.build_lstm()
        self.build_outputs()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver(max_to_keep=2)

    def build_inputs(self):
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_steps, self.embedding_size), name='inputs')
        self.target = tf.placeholder(tf.int32, shape=(self.batch_size, self.n_steps, self.n_classes), name='targets')

    def build_lstm(self):
        # 创建单个cell
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        # 堆叠多层神经元
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.n_layers)])
        # 初始化神经元状态
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        # lstm_outputs.shape=(batch_size, n_steps, lstm_szie)

    def build_outputs(self):
        self.logits = tf.layers.dense(self.lstm_outputs,self.n_classes)
        self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.target))

    def build_optimizer(self):
        # 使用cliping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clips)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

    def train_model(self, converter, epoches=20, modelSave_path='./model/', reload=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #从上一次训练训练开始训练，加载上一次参数
            if reload and os.path.exists( os.path.join(modelSave_path, 'checkpoint')):
                model_path = tf.train.latest_checkpoint(modelSave_path)
                sess=self.saver.restore(model_path)

            #输入数据
            x, y =converter.batch_generator()
            n_batches = int(x.shape[0] / self.batch_size)
            for epoch in range(epoches):
                for batch in range(n_batches):
                    X = x[batch*self.batch_size : batch*self.batch_size+self.batch_size, :]
                    X = converter.twoD_to_threeD(X)
                    Y = y[batch * self.batch_size: batch * self.batch_size + self.batch_size, :]
                    Y = converter.one_hot(Y)

                    loss, _, _ = sess.run([self.loss, self.optimizer, self.final_state], feed_dict={self.inputs: X, self.target: Y})
                    print('epoch: ' + str(epoch+1) + '    batch: ' + str(batch+1) + '/' + str(n_batches) + '  loss=', loss)
                self.saver.save(sess, os.path.join(modelSave_path, 'model.ckpt'), global_step=epoch)

    def sample(self, start_string=None, converter=None, checkpoint_path="./model/"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #加载模型
            model_path=tf.train.latest_checkpoint(checkpoint_path)
            print('Restored from: {}'.format(model_path))
            self.saver.restore(sess, model_path)
            new_state = sess.run(self.initial_state)
            #输入
            x=converter.word_to_vector[start_string]
            x=np.reshape(x, (1, 1, self.embedding_size))
            predicton, new_state = sess.run([self.prediction, self.final_state], feed_dict={self.inputs: x, self.initial_state: new_state})
            word=converter.softmaxVector_to_word(predicton)

            poem=""
            while word !="]":
                poem+=word
                x=np.reshape(converter.word_to_vector[word], (1, 1, self.embedding_size))
                predicton, new_state = sess.run([self.prediction, self.final_state], feed_dict={self.inputs: x, self.initial_state: new_state})
                word = converter.softmaxVector_to_word(predicton)
            print(poem)

    def sample_head(self, heads=None, converter=None, checkpoint_path="./model/"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 加载模型
            model_path = tf.train.latest_checkpoint(checkpoint_path)
            print('Restored from: {}'.format(model_path))
            self.saver.restore(sess, model_path)
            new_state = sess.run(self.initial_state)

            poem = ''
            add_comma = False
            for word in heads:
                add_comma = not add_comma
                while word != "," and word != "。" and word != ']':
                    poem += word
                    x = converter.word_to_vector[word]
                    x = np.reshape(x, (1, 1, self.embedding_size))
                    predicton, new_state = sess.run([self.prediction, self.final_state], feed_dict={self.inputs: x, self.initial_state: new_state})
                    word = converter.softmaxVector_to_word(predicton)
                sign = "," if add_comma else "。"
                poem = poem + sign
            print(poem)