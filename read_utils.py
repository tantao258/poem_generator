import os
import pickle
import random
import numpy as np
from gensim.models import Word2Vec

class TextConverter(object):
    def __init__(self,
                 filepath='corpus/poetry.txt',
                 word2vector_path='./word2vector/',
                 pickle_path='./pickle/',
                 batch_size=192,
                 n_steps=30,
                 embedding_size=256,
                 ):
        self.filepath=filepath
        self.pickle_path=pickle_path
        self.batch_size=batch_size
        self.n_steps=n_steps
        self.word2vector_path=word2vector_path
        self.embedding_size=embedding_size
        self.data_process()
        self.embedding()
        self.save_to_file()
        self.batch_generator()

    def data_process(self):
        processed_poetry_segment=[]
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    title, content = line.strip().split(":")
                    content = content.replace(" ","")
                    if "_" in content or "(" in content or ")" in content or "《" in content or "》" in content:
                        continue
                    if len(content)<5 or len(content)>80:
                        continue
                    content = "[" + content + "]" + " "
                    processed_poetry_segment.append([item for item in content])  #按照字符分割
                except Exception as e:
                    print(e)

        #按照诗的长度排序（降序）
        self.processed_poetry_segment = sorted(processed_poetry_segment, key=lambda line: len(line), reverse=True)
        # print("唐诗总数：", len(self.processed_poetry_segment))

    def train_word2vector(self):
        print("=" *60)
        print("开始训练词向量......")
        self.model = Word2Vec(self.processed_poetry_segment,
                         size=self.embedding_size,
                         window=6,
                         min_count=1,
                         workers=50,
                         iter=10)
        self.model.save(os.path.join(self.word2vector_path, "poem.model"))
        print("词向量训练完成!")
        print("=" * 60)

    def embedding(self):
        #加载词表
        if os.path.exists(os.path.join(self.pickle_path,'vocab.pkl')) and os.path.exists(os.path.join(self.pickle_path,'word_to_vector.pkl')) and \
            os.path.exists(os.path.join(self.pickle_path, 'word_to_int.pkl')) and os.path.exists(os.path.join(self.pickle_path,'int_to_word.pkl')):
            with open(os.path.join(self.pickle_path,'vocab.pkl'), 'rb') as f:
                self.vocab = pickle.load(f)
            with open(os.path.join(self.pickle_path, 'word_to_vector.pkl'), 'rb') as f:
                self.word_to_vector = pickle.load(f)
            with open(os.path.join(self.pickle_path, 'word_to_int.pkl'), 'rb') as f:
                self.word_to_int = pickle.load(f)
            with open(os.path.join(self.pickle_path, 'int_to_word.pkl'), 'rb') as f:
                self.int_to_word = pickle.load(f)

        else:
            #创建词表
            if os.path.exists(os.path.join(self.word2vector_path, "poem.model")):
                self.model = Word2Vec.load(os.path.join(self.word2vector_path, "poem.model"))   #加载word2vector model
                print("词向量加载完成。")
            else:
                self.train_word2vector()
            self.vocab=[w for w,v in self.model.wv.vocab.items()]
            self.word_to_vector = {w:self.model[w] for w in self.vocab}
            self.word_to_int = {w: i for i, w in enumerate(self.vocab, start=1)}
            self.int_to_word = {i: w for i, w in enumerate(self.vocab, start=1)}

    def vocab_size(self):
        return len(self.vocab)

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int[word])  # 通过词语去查找对应的数字
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word[index])  # 通过数字去查找对应的词语
        return "".join(words)  # join拼接字符串

    def save_to_file(self):
        with open(os.path.join(self.pickle_path,'vocab.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(os.path.join(self.pickle_path, 'word_to_int.pkl'), 'wb') as f:
            pickle.dump(self.word_to_int, f)
        with open(os.path.join(self.pickle_path, 'int_to_word.pkl'), 'wb') as f:
            pickle.dump(self.int_to_word, f)
        with open(os.path.join(self.pickle_path, 'word_to_vector.pkl'), 'wb') as f:
            pickle.dump(self.word_to_vector, f)

    def twoD_to_threeD(self, twoDarr):
        i=twoDarr.shape[0]
        j=twoDarr.shape[1]
        x=np.zeros((i,j,self.embedding_size))
        for ii in range(i):
            for jj in range(j):
                x[ii, jj,:]=self.word_to_vector[self.int_to_word[twoDarr[ii,jj]]]
        return x

    def one_hot(self,twoDarr):
        i=twoDarr.shape[0]
        j=twoDarr.shape[1]
        x=np.zeros((i,j,self.vocab_size()),np.float32)
        for ii in range(i):
            for jj in range(j):
                temp = np.zeros(self.vocab_size(), np.float32)
                temp[twoDarr[ii,jj]-1]=1
                x[ii,jj,:]=temp
        return x

    def softmaxVector_to_word(self,softmaxVector,top_n=5):
        prediction=np.squeeze(softmaxVector)
        a=np.argsort(prediction)
        a=a[::-1]
        b=list(map(lambda x:x+1, a))
        c=random.sample(b[:top_n], 1)[0]
        return  self.int_to_word[c]

    def batch_generator(self):
        poems_int=[]
        #将汉字转化为编码
        for poem in self.processed_poetry_segment:
            temp=[]
            for word in poem:
                temp.append(self.word_to_int[word])
            poems_int.append(temp)
        #compute the max length of poem
        self.n_steps=max([len(poem) for poem in poems_int])

        #长度不足的填充空格
        x_2D=np.full((len(poems_int), self.n_steps), self.word_to_int[" "], np.int32)
        for i,poem in enumerate(poems_int):
            for j in range(len(poem)):
                x_2D[i,j]=poems_int[i][j]

        n_batches=int(len(poems_int)/self.batch_size)
        # print(n_batch) 1245
        x_2D = x_2D[0:n_batches*self.batch_size,:]
        y_2D = np.copy(x_2D)
        y_2D[:, 0:-1] = x_2D[:, 1:]

        return x_2D, y_2D