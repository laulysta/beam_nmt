import numpy as np
import fcntl  # copy
import itertools
import sys, os
import argparse
import time
import datetime
from nmt import train
from os.path import join as pjoin

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dw', '--dim_word', required=False, default='50', help='Size of the word representation')
parser.add_argument('-d', '--dim_model', required=False, default='200', help='Size of the hidden representation')
parser.add_argument('-l', '--lr', required=False, default='0.001', help='learning rate')
parser.add_argument('-r', '--reload_path', required=False, default='', help='ex: pathModel.npz')
parser.add_argument('-data', '--dataset', required=False, default='testing', help='ex: testing, europarl')
parser.add_argument('-bs', '--batch_size', required=False, default='64', help='Size of the batch')



args = parser.parse_args()

dim_word = int(args.dim_word)
dim_model = int(args.dim_model)
lr = float(args.lr)
dataset = args.dataset
batch_size = int(args.batch_size)
reload_path = args.reload_path


if dataset == "testing":
    n_words_src = 100#3449
    n_words_trg = 100#4667
    datasets=['../data/dataset_testing/trainset_en.txt', 
              '../data/dataset_testing/trainset_de.txt']
    valid_datasets=['../data/dataset_testing/validset_en.txt', 
                    '../data/dataset_testing/validset_de.txt',
                    '../data/dataset_testing/validset_de.txt']
    other_datasets=['../data/dataset_testing/testset_en.txt', 
                    '../data/dataset_testing/testset_de.txt',
                    '../data/dataset_testing/testset_de.txt']
    dictionaries=['../data/dataset_testing/vocab_en.pkl', 
                  '../data/dataset_testing/vocab_de.pkl']

    sizeTrainset = 1000.0
    #batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)


elif dataset == "europarl_en_de":
    n_words_src=20000
    n_words_trg=20000
    datasets=['../data/europarl_de-en_txt.tok.low/europarl-v7.de-en.en.toc.low', 
              '../data/europarl_de-en_txt.tok.low/europarl-v7.de-en.de.toc.low']
    valid_datasets=['../data/europarl_de-en_txt.tok.low/newstest2015-ende-src.en.toc.low', 
                    '../data/europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low',
                    '../data/europarl_de-en_txt.tok.low/newstest2015-ende-ref.de.toc.low']
    other_datasets=['../data/europarl_de-en_txt.tok.low/newstest2016-ende-src.en.toc.low', 
                    '../data/europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low',
                    '../data/europarl_de-en_txt.tok.low/newstest2016-ende-ref.de.toc.low']
    dictionaries=['../data/europarl_de-en_txt.tok.low/vocab_en.pkl', 
                  '../data/europarl_de-en_txt.tok.low/vocab_de.pkl']

    sizeTrainset = 1920210.0
    #batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)



if reload_path != '':
    reload_ = True
    modelName = reload_path
    dirModelName = modelName.split('/')[-1] + '_reload'
    dirPath = '/'.join(modelName.split('/')[0:-1])
    dirPathOutput = pjoin(dirPath, 'output')
    if not os.path.exists(dirPathOutput):
        try:
            os.makedirs(dirPathOutput)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'
else:
    reload_ = False
saveFreq = nb_batch_epoch
use_dropout = True

decoder = 'gru_cond_legacy' # baseline



validerr, testerr, validbleu, testbleu , nb_epoch, nb_batch = train(saveto=modelName,
                                                                    reload_=reload_,
                                                                    dim_word=dim_word,
                                                                    dim=dim_model,
                                                                    encoder='gru',
                                                                    decoder=decoder, # 'gru_cond_legacy_lbc', # if args.covVec_in_attention or args.covVec_in_decoder else 'gru_cond',
                                                                    max_epochs=100,
                                                                    n_words_src=n_words_src,
                                                                    n_words=n_words_trg,
                                                                    optimizer='adadelta',
                                                                    decay_c=0.,
                                                                    alpha_c=0.,
                                                                    clip_c=1.,
                                                                    lrate=lr,
                                                                    patience=10,
                                                                    maxlen=50,
                                                                    batch_size=batch_size,
                                                                    valid_batch_size=batch_size,
                                                                    validFreq=nb_batch_epoch, # freq in batch of computing cost for train, valid and test
                                                                    dispFreq=nb_batch_epoch, # freq of diplaying the cost of one batch (e.g.: 1 is diplaying the cost of each batch)
                                                                    saveFreq=saveFreq, # freq of saving the model per batch
                                                                    sampleFreq=nb_batch_epoch, # freq of sampling per batch
                                                                    datasets=datasets,
                                                                    valid_datasets=valid_datasets,
                                                                    other_datasets=other_datasets,
                                                                    dictionaries=dictionaries,
                                                                    use_dropout=use_dropout,
                                                                    use_dec_word_dropout=use_dropout,
                                                                    use_word_dropout=use_dropout,
                                                                    rng=1234,
                                                                    trng=1234,
                                                                    save_inter=True, #save all the time
                                                                    train_beam_model=True,
                                                                    multibleu='multi-bleu.perl',
                                                                    valid_output=dirPathOutput+'/valid_output.s2.2',
                                                                    other_output=dirPathOutput+'/other_output.s2.2')

