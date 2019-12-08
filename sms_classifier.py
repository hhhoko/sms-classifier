import re
import pandas as pd
import math
import nltk




class FileOperate:
    def __init__(self, data_path):
        data = pd.read_excel(data_path)
        self.data = data.iloc[:, :2]

    def load_data(self):
        X = list()
        y = list()

        for i in range(len(self.data)):
            X.append(FileOperate.__clean_data(self.data.iloc[i, 1]))
            y.append(1 if self.data.iloc[i, 0] == 'spam' else 0)

        return X, y

    def load_data_test(self):
        X = list()

        for i in range(len(self.data)):
            X.append(FileOperate.__clean_data(self.data.iloc[i, 1]))

        return X

    def __clean_data(origin_info):
        '''
        清洗数据，去掉非字母的字符，和字节长度小于 2 的单词
        :return:
        '''
        # 先转换成小写
        # 把标点符号都替换成空格
        if type(origin_info) != type(','):
            origin_info = str(origin_info)
        temp_info = re.sub('\W', ' ', origin_info.lower())
        # 根据空格（大于等于 1 个空格）
        words = re.split(r'\s+', temp_info)
        stop_words = nltk.corpus.stopwords.words('english')
        # return list(filter(lambda x: len(x) >= 3, words))
        return list(filter(lambda x: x if x not in set(stop_words) else '', words))

class NaiveBayes:

    def __init__(self):
        self.__ham_count = 0  # 非垃圾短信数量
        self.__spam_count = 0  # 垃圾短信数量

        self.__ham_words_count = 0  # 非垃圾短信单词总数
        self.__spam_words_count = 0  # 垃圾短信单词总数

        self.__ham_words = list()  # 非垃圾短信单词列表
        self.__spam_words = list()  # 垃圾短信单词列表

        # 训练集中不重复单词集合
        self.__word_dictionary_set = set()

        self.__word_dictionary_size = 0

        self.__ham_map = dict()  # 非垃圾短信的词频统计
        self.__spam_map = dict()  # 垃圾短信的词频统计

        self.__ham_probability = 0
        self.__spam_probability = 0

    def fit(self, X_train, y_train):
        self.build_word_set(X_train, y_train)
        self.word_count()
        # return self.__ham_probability, self.__spam_probability

    def predict(self, X_train):
        return [self.predict_one(sentence) for sentence in X_train]

    def build_word_set(self, X_train, y_train):
        '''
        第 1 步：建立单词集合
        :param X_train:
        :param y_train:
        :return:
        '''
        for words, y in zip(X_train, y_train):
            if y == 0:
                # 非垃圾短信
                self.__ham_count += 1
                self.__ham_words_count += len(words)
                for word in words:
                    self.__ham_words.append(word)
                    self.__word_dictionary_set.add(word)
            if y == 1:
                # 垃圾短信
                self.__spam_count += 1
                self.__spam_words_count += len(words)
                for word in words:
                    self.__spam_words.append(word)
                    self.__word_dictionary_set.add(word)

        # print('非垃圾短信数量', self.__ham_count)
        # print('垃圾短信数量', self.__spam_count)
        # print('非垃圾短信单词总数', self.__ham_words_count)
        # print('垃圾短信单词总数', self.__spam_words_count)
        # print(self.__word_dictionary_set)
        self.__word_dictionary_size = len(self.__word_dictionary_set)

    def word_count(self):
        # 第 2 步：不同类别下的词频统计
        for word in self.__ham_words:
            self.__ham_map[word] = self.__ham_map.setdefault(word, 0) + 1

        for word in self.__spam_words:
            self.__spam_map[word] = self.__spam_map.setdefault(word, 0) + 1

        # [下面两行计算先验概率]
        # 非垃圾短信的概率
        self.__ham_probability = self.__ham_count / (self.__ham_count + self.__spam_count)
        # 垃圾短信的概率
        self.__spam_probability = self.__spam_count / (self.__ham_count + self.__spam_count)

    def predict_one(self, sentence):
        ham_pro = 0
        spam_pro = 0

        for word in sentence:
            # print('word', word)
            ham_pro += math.log(
                (self.__ham_map.get(word, 0) + 1) / (self.__ham_count + self.__word_dictionary_size))

            spam_pro += math.log(
                (self.__spam_map.get(word, 0) + 1) / (self.__spam_count + self.__word_dictionary_size))

        ham_pro += math.log(self.__ham_probability)
        spam_pro += math.log(self.__spam_probability)

        # print('垃圾短信概率', spam_pro)
        # print('非垃圾短信概率', ham_pro)
        return int(spam_pro >= ham_pro)

if __name__ == '__main__':
    f = FileOperate(r'train.xlsx')
    NB = NaiveBayes()
    x_train, y_train = f.load_data()
    NB.fit(x_train, y_train)
    # x_test = pd.read_excel(r"test_cleaned.xlsx")
    f_test = FileOperate(r'test_cleaned.xlsx')
    x_test = f_test.load_data_test()
    prediction = NB.predict(x_test)
    dataframe = pd.DataFrame(prediction, columns=['prediction'])
    dataframe.to_csv(r"test_prediction.csv")