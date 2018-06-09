from read_utils import *
from model import *
import shutil
import os

#设置超参数
tf.app.flags.DEFINE_string("filePath", "./corpus/poetry.txt", "filePath of corpus")
tf.app.flags.DEFINE_string("word2vector_path", "./word2vector/", "word2vector_path")
tf.app.flags.DEFINE_string("pickle_path", "./pickle/", "fpickle_path of vocab table")
tf.app.flags.DEFINE_string('modelSave_path', './model/', 'modelSave_path')
tf.app.flags.DEFINE_integer("epoches", 20, "epoches")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.app.flags.DEFINE_integer("n_steps", 30, "n_steps")
tf.app.flags.DEFINE_integer("embedding_size", 256, "embedding_size")
tf.app.flags.DEFINE_integer("n_classes", 100, "n_classes")
tf.app.flags.DEFINE_integer("rnn_size", 128, "rnn_size")
tf.app.flags.DEFINE_integer("n_layers", 2, "n_layers")
tf.app.flags.DEFINE_float("keep_prob", 0.6, "Dropout")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
Flags = tf.app.flags.FLAGS

def train():
    #---------------------------------------------------------------------------------------
    choice=input("是否清空所有原始数据： y--清空   n--不清空\n")
    if choice=="y":
        #重置文件夹
        for fold in [Flags.word2vector_path, Flags.pickle_path, Flags.modelSave_path]:
            if os.path.exists(fold):
                shutil.rmtree(fold)
                os.mkdir(fold)
            else:
                os.mkdir(fold)
        print("已经清空所有数据。")
    #-----------------------------------------------------------------------------------------
    #创建converter对象
    converter=TextConverter(filepath=Flags.filePath,
                            word2vector_path=Flags.word2vector_path,
                            pickle_path=Flags.pickle_path,
                            batch_size=Flags.batch_size,
                            n_steps=Flags.n_steps,
                            embedding_size=Flags.embedding_size)

    #创建lstm对象
    lstm=LSTM( sampling=False,
               n_classes=converter.vocab_size(),
               n_steps=converter.n_steps,
               embedding_size=Flags.embedding_size,
               rnn_size=Flags.rnn_size,
               n_layers=Flags.n_layers,
               batch_size=Flags.batch_size)
    #训练模型并保存模型
    lstm.train_model(epoches=Flags.epoches,
                     converter=converter,
                     modelSave_path=Flags.modelSave_path)

def sample():
    #创建converter对象
    converter=TextConverter(filepath=Flags.filePath,
                            word2vector_path=Flags.word2vector_path,
                            pickle_path=Flags.pickle_path,
                            batch_size=Flags.batch_size,
                            n_steps=Flags.n_steps,
                            embedding_size=Flags.embedding_size)
    #创建lstm对象
    lstm = LSTM(sampling=True,
                n_classes=converter.vocab_size(),
                n_steps=converter.n_steps,
                embedding_size=Flags.embedding_size,
                rnn_size=Flags.rnn_size,
                n_layers=Flags.n_layers,
                batch_size=converter.batch_size)

    lstm.sample(start_string="[",
                converter=converter)

if __name__ == '__main__':
    train()
