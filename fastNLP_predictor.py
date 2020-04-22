from fastNLP.io.model_io import ModelLoader
from fastNLP.core.predictor import Predictor
from fastNLP.core import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.io.loader import PeopleDailyNERLoader
from fastNLP.io import PeopleDailyPipe
from fastNLP.core.const import Const
from collections import Counter


class CustomizedNER(object):
    def __init__(self, modelFile, vocabFile, addTarget2Vocab=False):
        # CHAR_INPUT="chars", 并且会转化为word_index
        self._vocabFile = vocabFile
        self._addTarget2Vocab = addTarget2Vocab
        self._CONST_CHAR=Const.CHAR_INPUT
        self._CONST_WORDS=Const.INPUT
        self._CONST_TARGET = Const.TARGET
        self._input_fields = [self._CONST_WORDS, Const.INPUT_LEN]
        self._word_counter, self._word_vocab, self._target_counter, \
        self._target_vocab, self._target = self._get_vocabs()
        self._vocab4word = Vocabulary()
        self._update_word()
        if self._addTarget2Vocab:
            self._vocab4target = Vocabulary(unknown=None, padding=None)
            self._input_fields.append(self._CONST_TARGET)
            self._update_target()
        self._model = Predictor(ModelLoader().load_pytorch_model(modelFile))

    def _target_token(self, word_token, cont, number = "", word = ""):
        ret = dict()
        sign = True
        lastIdx = len(word_token) - 1
        for num, token in zip(enumerate(word_token), cont):
            if num[1] in self._target:
                if sign:
                    number += str(num[1])
                    word += token
                    if num[0] < lastIdx and not word_token[num[0]+1]:
                        sign = False
                else:
                    ret.setdefault(number, set())
                    ret[number].add(word)
                    number = ""
                    word = token
                    sign = True
        if number:
            ret.setdefault(number, set())
            ret[number].add(word)
        return ret

    def _extract_ner(self, tokenNum, token, weighted=False):
        if not weighted:
            cls = self._target.get(int(max(tokenNum, key=tokenNum.count)), "")
            if cls.endswith("LOC"):
                return {"LOC": [x for x in token]}
            elif cls.endswith("PER"):
                return {"PER": [x for x in token]}
            elif cls.endswith("ORG"):
                return {"ORG": [x for x in token]}

    def _get_ner(self, tokenNumber, tokenWord):
        nerDict = self._target_token(tokenNumber, tokenWord)
        ret = dict()
        for num, token in nerDict.items():
            if len(num)==1:
                continue
            for k, v in self._extract_ner(num, token).items():
                ret.setdefault(k, list())
                ret[k].extend(v)
        return ret

    def _read_vocab(self):
        with open(self._vocabFile, "r", encoding="utf-8") as vocabIn:
            return eval(vocabIn.read())

    def _reverse_dict(self, dic):
        ret = dict()
        for key, value in dic.items():
            ret.setdefault(value, key)
        return ret

    def _tartget_label(self, dic):
        ret = self._reverse_dict(dic)
        del ret[0]
        return ret

    def _get_vocabs(self):
        vocabs = self._read_vocab()
        word_count = vocabs.get("wordsWc", dict())
        wordsVocab = vocabs.get("wordsVocab", dict())
        target_count = vocabs.get("targetWc", dict())
        targetVocab = vocabs.get("targetVocab", dict())
        reverseTargetVocab = self._tartget_label(targetVocab)
        return Counter(word_count), wordsVocab, Counter(target_count), targetVocab, reverseTargetVocab


    def _update_word(self):
        self._vocab4word.update(self._word_vocab)
        self._vocab4word.word_count = self._word_counter

    def _update_target(self):
        self._vocab4target.update(self._target_vocab)
        self._vocab4target.word_count = self._target_counter

    @property
    def model(self):
        if not self._model:
            raise
        return self._model

    def formatRowString(self, msg):
        msg = msg.strip()
        tokenized_char = [x for x in msg]
        self._dataset = DataSet()
        if self._addTarget2Vocab:
            ins = Instance(chars=tokenized_char, raw_chars=tokenized_char, target=list(dict(self._target_vocab).keys()))
        else:
            ins = Instance(chars=tokenized_char, raw_chars=tokenized_char)
        self._dataset.append(ins)

    @property
    def dataset(self):
        # if input as dict format:
        # data = DataSet({"raw_chars":[msg], "words":[[x for x in msg]], "seq_len":[len(word_list)]})
        # 从该dataset中的chars列建立词表
        self._vocab4word.from_dataset(self._dataset, field_name=self._CONST_CHAR)
        # 使用vocabulary将chars列转换为index
        self._vocab4word.index_dataset(self._dataset, field_name=self._CONST_CHAR, new_field_name=self._CONST_WORDS)
        if self._addTarget2Vocab:
            self._vocab4target.from_dataset(self._dataset, field_name=self._CONST_TARGET)
            self._vocab4target.index_dataset(self._dataset, field_name=self._CONST_TARGET)
        self._dataset.add_seq_len(self._CONST_CHAR)
        self._dataset.set_input(*self._input_fields)
        return self._dataset

    def _content(self):
        for line in self._dataset["raw_chars"].content:
            yield "".join(line)

    def result(self, dataset):
        # 打印数据集中的预测结果
        ret = self.model.predict(dataset)["pred"]
        for line, cont in zip(ret, self._content()):
            yield self._get_ner(line[0].tolist(), cont)


if __name__=="__main__":
    # 加载模型和vocab文件
    modelFile = "./model_ckpt_epoch1_batch8_lay3_hidden100_gpu.pkl"
    vocabFile = "./Vocabs.json"
    processor = CustomizedNER(modelFile, vocabFile)
    from time import time
    # 测试样例
    msg = ["丁先生住在云南省保山市腾冲市州门前路东100米, 昨天他去了江苏省盐城市响水县小尖镇致富路9号门的李先生家里还东西"]
    start = time()
    for x in msg:
        processor.formatRowString(x)
        single_sample = processor.dataset
        for ret in processor.result(single_sample):
            print(ret)
    end = time()
    print(end-start)
