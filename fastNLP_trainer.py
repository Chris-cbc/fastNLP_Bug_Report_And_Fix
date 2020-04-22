from fastNLP.io.loader import PeopleDailyNERLoader, MsraNERLoader, WeiboNERLoader, Conll2003NERLoader, OntoNotesNERLoader
from fastNLP.models import BiLSTMCRF
from fastNLP.embeddings import BertEmbedding
from fastNLP.io import PeopleDailyPipe
from fastNLP import SpanFPreRecMetric
from torch.optim import Adam
from fastNLP import LossInForward
from fastNLP.io.model_io import ModelSaver
from fastNLP import Trainer
import torch
from fastNLP import Tester
import json

assert torch.cuda.is_available()

def trainer(data_folder, write2model, write2vocab):
    data_bundle = PeopleDailyNERLoader().load(data_folder)  # 这一行代码将从{data_dir}处读取数据至DataBundle
    data_bundle = PeopleDailyPipe().process(data_bundle)
    data_bundle.rename_field('chars', 'words')
    # 存储vocab
    targetVocab = dict(data_bundle.vocabs["target"])
    wordsVocab = dict(data_bundle.vocabs["words"])
    targetWc = dict(data_bundle.vocabs['target'].word_count)
    wordsWc = dict(data_bundle.vocabs['words'].word_count)
    with open(write2vocab, "w", encoding="utf-8") as VocabOut:
        VocabOut.write(json.dumps({"targetVocab": targetVocab, "wordsVocab": wordsVocab, "targetWc":targetWc, "wordsWc":wordsWc}, ensure_ascii=False))

    embed = BertEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn', requires_grad=False, auto_truncate=True)
    model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=100, dropout=0.5,
                  target_vocab=data_bundle.get_vocab('target'))

    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
    optimizer = Adam(model.parameters(), lr=2e-5)
    loss = LossInForward()
    device= 0 if torch.cuda.is_available() else 'cpu'
    # device = "cpu"
    trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=8,
                        dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device, n_epochs=1)
    trainer.train()
    tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
    tester.test()
    saver = ModelSaver(write2model)
    saver.save_pytorch(model, param_only=False)


if __name__=="__main__":
    data = "./data"
    write2model = "./model_ckpt_epoch1_batch8_lay3_hidden100_gpu.pkl"
    write2vocab = "./Vocabs.json"
    trainer(data, write2model, write2vocab)
