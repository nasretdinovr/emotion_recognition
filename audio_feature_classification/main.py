# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from pytorch.common.datasets_parsers.av_parser import AVDBParser
from voice_feature_extraction import OpenSMILE
from accuracy import Accuracy, Accuracy_regression


def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(dataset_root, file_list,
                                max_num_clips=max_num_clips)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data

def calc_features(data, opensmile_root_dir, opensmile_config_path):
    vfe = OpenSMILE(opensmile_root_dir, opensmile_config_path)

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc audio features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        try:
            voice_feat = vfe.process(clip.wav_rel_path)
        except:
            print('error calc voice features!')
            data.remove(clip)

        feat.append(voice_feat)
        targets.append(clip.labels)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim=100):
    if pca_dim > 0:
        pca_model = PCA(n_components=min(pca_dim, X_train.shape[1])).fit(X_train)
        X_train = pca_model.transform(X_train)
        X_test = pca_model.transform(X_test)

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn

    y_pred = []
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'exp_1'
    max_num_clips = 0 # загружайте только часть данных для отладки кода
    use_dump = False # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = 'D:/AVER'
    if 1:
        train_dataset_root = base_dir + '/Ryerson/Speech/Video'
        train_file_list = base_dir + '/Ryerson/Speech/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/Ryerson/Speech/Video'
        test_file_list = base_dir + '/Ryerson/Speech/test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root =base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    # opensmile configuration
    opensmile_root_dir = 'C:/AVER/opensmile-2.3.0'
    # TODO: поэкспериментируйте с различными конфигурационными файлами библиотеки OpenSmile
    opensmile_config_path = 'C:/AVER/opensmile-2.3.0/config/avec2013.conf'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=max_num_clips)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=max_num_clips)

        # get features
        train_feat, train_targets = calc_features(train_data, opensmile_root_dir, opensmile_config_path)
        test_feat, test_targets = calc_features(test_data, opensmile_root_dir, opensmile_config_path)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        with open(experiment_name + '.pickle', 'wb') as f:
            pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=0)
