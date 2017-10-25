import csv
import logging
import os
from builtins import print
from os import path
from random import shuffle

import gensim.models.doc2vec
import numpy
import sklearn
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR, LinearSVC


class Doc2vec_clustering:
    featuresize = 100
    classifier = LogisticRegression()

    def build_Model(self, path_to_files):
        '''
        Build a doc2vec model on basis of the dense groundtruth.

        :param path_to_files:
        :return:
        '''
        sentences = LabeledLineSentence(
            filename=path.join(path_to_files, 'metadata_translation_v2_reduced_groundtruth_dense.csv'))

        model = gensim.models.doc2vec.Doc2Vec(min_count=1, window=40, size=self.featuresize, negative=2, workers=4)
        train_sent_array = sentences.to_array().__getitem__(0)

        model.build_vocab(train_sent_array)

        # train in different sequences
        for epoch in range(10):
            model.train(sentences.sentences_perm())
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        model.save(path.join(path_to_files, 'dec2vec_model.d2v'))

        result = sentences.to_array()
        return result

    def build_text_features(self, path_to_files, sent_array):
        '''
        extract feature vectors from model and store them into txt files of the format 
        'id, 0.345363245, -0.2342348, ...'
        
        :param path_to_files: 
        :param sent_array: consists of 2 arrays per element [[id, contributor, ...][label]]
        :return: feature_arrays: array of the feature vector in the same order as the set_array - 
                 looks like: [-0.07759716 -0.11191639 -0.03950126 ... ]. 
        '''

        channel_prev = ''
        features_doc = ''
        feature_arrays = numpy.zeros((len(sent_array), self.featuresize))
        model = gensim.models.doc2vec.Doc2Vec.load(path.join(path_to_files, 'dec2vec_model.d2v'))

        for i in range(len(sent_array)):
            feature_arrays[i] = model.docvecs[sent_array.__getitem__(i).__getitem__(1)]

            doc_id = str(sent_array.__getitem__(i).__getitem__(0)).split('/')
            channel = doc_id[1]
            if channel != channel_prev:  # new chanel
                if channel_prev != '':  # if not first: write previous chanel to file
                    file = path.join(path_to_files, 'text_features/' + channel_prev + '.txt')
                    features_file = open(file, 'w')
                    features_file.write(features_doc)
                    features_file.close()

                features_doc = ''

            name = doc_id[2].split(',')[0]
            features_doc += name[0:len(name) - 1]  # document id

            for j in range(len(feature_arrays[i])):
                features_doc += ',' + str(feature_arrays[i][j])

            features_doc += ' \n\n'

            channel_prev = channel

        # save last channel
        file = path.join(path_to_files, 'text_features/' + channel + '.txt')
        features_file = open(file, 'w')
        features_file.write(features_doc)
        features_file.close()

        return feature_arrays

    def do_early_fusion(self, path_to_files, sent_array, text_feature_array, label_array):
        '''
        - extract feature vectors from model and store them into txt files of the format 
          'id, 0.345363245, -0.2342348, ...'
        - normalize the feature sets
        - concatinate the features obtained from doc2vec with the features extracted from the audio files
        - normalize the features
        - perform feature reduction (PCA?)

        :param path_to_files: 
        :return: 
        '''

        ### read all audio features from directory and store them into the list 'audio_features'. This list looks like this:
        ### [[id][0.306273757212, 1.09274908459, ...]]
        audio_features = []
        for filename in os.listdir(path.join(path_to_files, 'audio_features')):
            with open(path.join(path_to_files, 'audio_features', filename)) as audio_feature_file:
                text_lines = audio_feature_file.readlines()
                for line in text_lines:
                    if len(line) > 3:  # if line is not empty
                        file_id = '/' + filename.split('.')[0] + '/' + str(line.split(',')[0].split('.')[0])
                        features_doc = []
                        count = 0
                        for value in line.split(','):
                            if count != 0:
                                features_doc.append(float(value))
                            count += 1
                        document = [file_id, features_doc]
                        audio_features.append(document)

        ### concatinate doc2vec features with audio features
        fusion_arr = []
        features_fus_arr = []
        id_fus_arr = []
        fusion_labels = []
        fusion_count = 0
        for i in range(len(sent_array)):
            full_id = str(sent_array.__getitem__(i).__getitem__(0)[0])
            found_one = False
            for j in range(len(audio_features)):
                if audio_features[j][0] == full_id:
                    found_one = True
                    fusion_count += 1
                    features_fus = numpy.concatenate((audio_features[j][1], text_feature_array[i]), axis=0)
                    fusion = [full_id, features_fus]  # maybe not needed
                    fusion_arr.append(fusion)  # maybe not needed
                    features_fus_arr.append(features_fus)
                    id_fus_arr.append(full_id)
                    fusion_labels.append(label_array[i])
            if not found_one:
                # print('did not take this one: ' + full_id)
                pass

        ### normalize features
        features_fus_arr = sklearn.preprocessing.normalize(features_fus_arr)

        ### PCA
        pca = PCA(n_components=100)
        features_fus_arr = pca.fit_transform(features_fus_arr)
        fusion_arr = [features_fus_arr, fusion_labels]

        return fusion_arr

    def do_late_fusion(self, path_to_files, sent_array, text_feature_array, label_array):
        '''
         After splitting the data into trainings set and test set, we work on the trainings set here:
         - The trainings set is split by a ratio of 70 : 30
         - now the 70% part is the meta- trainings set
         - the meta- trainings set is used to train each of the models (audio and text features)
            -> here I just take the audio and text features according to the ratio
         - now the features are used to predict the labels the 30% of the meta- test set
         - this prediction is treated as an other training for the meta- classificator
         - after the classificator is trained it is used to predict the labels of the test set.

        :param path_to_files:
        :return:
        '''

        ### read all audio features from directory and store them into the list 'audio_features'.
        logging.info('read all audio features from directory')
        audio_features = []
        for filename in os.listdir(path.join(path_to_files, 'audio_features')):
            with open(path.join(path_to_files, 'audio_features', filename)) as audio_feature_file:
                text_lines = audio_feature_file.readlines()
                for line in text_lines:
                    if len(line) > 3:  # if line is not empty
                        file_id = '/' + filename.split('.')[0] + '/' + str(line.split(',')[0].split('.')[0])
                        features_doc = []
                        count = 0
                        for value in line.split(','):
                            if count != 0:
                                features_doc.append(float(value))
                            count += 1
                        document = [file_id, features_doc]
                        audio_features.append(document)

        ### reduce text_features to the same documents which exist as audio_features
        logging.info('reduce text_features to the same documents which exist as audio_features')
        id_fus_arr = []
        merge_labels = []
        audio_features_fus = []  # audio features after reduction through merging
        text_features_fus = []  # text  features after reduction through merging
        wrapping_array = []
        for i in range(len(sent_array)):
            full_id = str(sent_array.__getitem__(i).__getitem__(0)[0])
            for j in range(len(audio_features)):
                if audio_features[j][0] == full_id:
                    features_fus = audio_features[j][1]
                    audio_features_fus.append(features_fus)
                    text_features_fus.append(text_feature_array[i])
                    id_fus_arr.append(full_id)

                    meta_feature = MetaFeatures(full_id, audio_features[j][1], text_feature_array[i])
                    wrapping_array.append(meta_feature)

                    merge_labels.append(label_array[i])

        # split data into 75:25 to train and evaluate late_fusion - classification
        logging.info('split data into 75:25 to train and evaluate late_fusion - classification')
        train_arrays_class, test_arrays_class, train_labels_class, test_labels_class = \
            sklearn.model_selection.train_test_split(wrapping_array, merge_labels, test_size=0.25,
                                                     random_state=42)

        # split data into 70:30 to train the meta-classifier
        logging.info('split data into 70:30 to train the meta-classifier')
        train_arrays_meta, test_arrays_meta, train_labels_meta, test_labels_meta = \
            sklearn.model_selection.train_test_split(train_arrays_class, train_labels_class, test_size=0.3,
                                                     random_state=42)

        # split data into 70:30 to train the meta-classifier
        # logging.info('split data into 70:30 to train the meta-classifier')
        # train_arrays_meta, test_arrays_meta, train_labels_meta, test_labels_meta = \
        #     sklearn.model_selection.train_test_split(wrapping_array, merge_labels, test_size=0.3,
        #                                              random_state=42)

        logging.info('\nMeta trainings set: ' + str(len(train_arrays_meta)) + '\nMeta trainings Lables: ' + str(
            len(train_labels_meta)) + '\nMeta test set: ' + str(len(test_arrays_meta)) + '\nMeta test Lables: ' + str(
            len(test_labels_meta)))
        logging.info('\nClassification trainings set: ' + str(
            len(train_arrays_class)) + '\nClassification trainings Lables: ' + str(
            len(train_labels_class)) + '\nClassification test set: ' + str(
            len(test_arrays_class)) + '\nClassification test Lables: ' + str(
            len(test_labels_class)))

        train_id = []
        train_audio = []
        train_text = []
        for i in range(len(train_arrays_meta)):
            train_id.append(train_arrays_meta[i].get_Id())
            train_audio.append(train_arrays_meta[i].get_audio_features())
            train_text.append(train_arrays_meta[i].get_text_features())

        ### train classificator with 70% of the audio features
        logging.info('predicting audio - only features')
        classifier_audio = LogisticRegression()
        # classifier_audio = DummyClassifier(strategy='most_frequent')
        classifier_audio.fit(train_audio, train_labels_meta)

        audio_features = []
        for j in range(len(test_arrays_meta)):
            audio_features.append(test_arrays_meta[j].get_audio_features())
        prediction_arr_audio = classifier_audio.predict(audio_features)

        ### train classificator with 70% of the text features
        logging.info('predicting text - only features')
        classifier_text = LogisticRegression()
        # classifier_text = DummyClassifier(strategy='most_frequent')
        classifier_text.fit(train_text, train_labels_meta)

        text_features = []
        for j in range(len(test_arrays_meta)):
            text_features.append(test_arrays_meta[j].get_text_features())
        prediction_arr_text = classifier_text.predict(text_features)

        ### Scores for Text and Auio features
        score = sklearn.metrics.accuracy_score(prediction_arr_text, test_labels_meta)
        print('Classification accuracy_score for text - only features:  ', round(score * 100, 2), '%')
        score = sklearn.metrics.accuracy_score(prediction_arr_audio, test_labels_meta)
        print('Classification accuracy_score for audio - only features: ', round(score * 100, 2), '%')

        ### write prediction from audio and text features in one line
        # ##### predictions #######      | ### truth ###
        # audio features | text features | truth
        # ______________________________________________
        # spoken         | irish         | irish
        # classical      | classical     | classical
        # ...
        # --> the classifier learns "if text features says irish but the audio features spoken take the text features
        prediction_arr = []
        for i in range(len(prediction_arr_audio)):
            # element = [prediction_arr_audio[i], prediction_arr_text[i]]
            # prediction_arr.append(element)

            ### audio array: string to float because features have to be float
            if prediction_arr_audio[i] == 'classical':
                audio_element = 1
            if prediction_arr_audio[i] == 'irish':
                audio_element = 2
            if prediction_arr_audio[i] == 'folklore':
                audio_element = 3
            if prediction_arr_audio[i] == 'invironment':
                audio_element = 4
            if prediction_arr_audio[i] == 'spoken':
                audio_element = 5
            if prediction_arr_audio[i] == 'popular':
                audio_element = 6

            ### text array: string to float because features have to be float
            if prediction_arr_text[i] == 'classical':
                text_element = 1
            if prediction_arr_text[i] == 'irish':
                text_element = 2
            if prediction_arr_text[i] == 'folklore':
                text_element = 3
            if prediction_arr_text[i] == 'invironment':
                text_element = 4
            if prediction_arr_text[i] == 'spoken':
                text_element = 5
            if prediction_arr_text[i] == 'popular':
                audio_element = 6

            element = [audio_element, text_element]
            prediction_arr.append(element)

        print('Length of combined feature array' + str(len(prediction_arr)))
        # for i in range(len(prediction_arr)):
        #     print('prediction: ' + str(prediction_arr[i]) + ' - reality: ' + test_labels_meta[i])

        # TODO: train the meta classifier against the truth
        # meta_clf = LogisticRegression()
        meta_clf = SVR(kernel='rbf', gamma=0.001)
        meta_clf.fit(prediction_arr, train_labels_meta)
        prediction = meta_clf.predict(test_arrays_meta, train_labels_meta)

        score = sklearn.metrics.accuracy_score(prediction, test_labels_meta)
        print('Classification accuracy_score for early fusion: ', round(score * 100, 2), '%')

        # TODO: for evaluation also predict the test set first with the features seperately and then with the trained classifier


        # TODO: split audio and text featres to a ratio of 70 : 30 for meta- trainings and- test set
        # TODO: train classifier for text and audio on the 70%
        # TODO: in # epochs classify with both features and some normalization factors -> learn factors
        # TODO: split audio and text featres into trainings- set and test- set (75:25) for evaluation

    def train_classifier(self, path_to_files, fusion_type):
        '''
        Train the classifiers according to the labels from the groundtruth. For clustering either no_fusion is applied.
        In the case of no_fusion, only the text features are taken for clustering.
        In case of early_fusion, the audio and text features are concatenated before clustering is done.
        In case of late_fusion, the features are clustered seperately. With the result a meta classifier is trained,
        which does the final classification.

        :param path_to_files:
        :param fusion: type of fusion. This can either be early_fusion, late_fusion or no_fusion
        :return:
        '''

        result = self.build_Model(path_to_files)
        sent_array = result.__getitem__(0)
        label_array = result.__getitem__(1)

        ### get feature_arrays without writing them to text files
        # for i in range(len(sent_array)):
        #     feature_arrays[i] = model.docvecs[sent_array.__getitem__(i).__getitem__(1)]

        ### write feature_arrays to text files
        feature_arrays = self.build_text_features(path_to_files, sent_array)

        if fusion_type == 'no_fusion':
            logging.info('just processing text features')
            self.do_evaluation(feature_arrays, label_array)

        if fusion_type == 'early_fusion':
            ### early fusion
            fusion_result = self.do_early_fusion(path_to_files, sent_array, feature_arrays, label_array)
            feature_arrays_fusion = fusion_result[0]
            label_array_fusion = fusion_result[1]
            # print('Number of features: ' + str(len(fusion_result[0][1])))
            # print('Number of documents: ' + str(len(label_array_fusion)))

            self.do_evaluation(feature_arrays_fusion, label_array_fusion)
        if fusion_type == 'late_fusion':
            self.do_late_fusion(path_to_files, sent_array, text_feature_array=feature_arrays, label_array=label_array)

    def do_evaluation(self, feature_arrays, label_array):

        train_arrays, test_arrays, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(feature_arrays, label_array, test_size=0.25, random_state=42)

        ### 2 Baselines: 'most_frequent', 'stratified'
        classifier = DummyClassifier(strategy='most_frequent')
        classifier.fit(train_arrays, train_labels)
        prediction_arr = classifier.predict(test_arrays)
        score = classifier.score(test_arrays, test_labels)
        print('Baseline < most_frequent > score: ', round(score * 100, 2), '%')

        classifier = DummyClassifier(strategy='stratified')
        classifier.fit(train_arrays, train_labels)
        prediction_arr = classifier.predict(test_arrays)
        score = classifier.score(test_arrays, test_labels)
        print('Baseline < stratified > score: ', round(score * 100, 2), '%')

        ### Classification
        classifier = LogisticRegression()
        classifier.fit(train_arrays, train_labels)
        prediction_arr = classifier.predict(test_arrays)
        score = classifier.score(test_arrays, test_labels)
        print('Classification score: ', round(score * 100, 2), '%')

        ### some similarity tests on the model...
        # print(model.docvecs['classical'])
        # print('Most similar to irish_4: \n', model.docvecs.most_similar('irish_4'))

    def do_tfidf_clustering(self, filename_dgt):
        '''
         As an experiment a simple tf-idf algorithm and a LinearSVC classification is applied on the test set
         resulting from the ground truth.
         Therefore the groundtruth is split into a trainings set and a test set.
         The classificator is trained by the trainings set and applied on the test set.
        :param path_to_files:
        :return:
        '''

        ### read sentences and labels from file
        sentences = LabeledLineSentence(filename=filename_dgt)
        sent_array = sentences.to_array().__getitem__(0)
        label_array = sentences.to_array().__getitem__(1)

        train_arrays, test_arrays, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(sent_array, label_array, test_size=0.25, random_state=42)

        ### Tutorial:
        # https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/
        ### TRAININGS set
        train_data = []
        for sent in train_arrays:
            line = ''
            for word in sent[0]:
                line += word + ' '
            train_data.append(line)

        ### TEST set
        test_data = []
        for sent in test_arrays:
            line = ''
            for word in sent[0]:
                line += word + ' '
            test_data.append(line)

        # ["The sky is blue.", "The sun is bright."]
        vectorizer = TfidfVectorizer(encoding='utf-8')
        train_corpus_tf_idf = vectorizer.fit_transform(train_data)
        # vectorizer.vocabulary_
        # {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
        test_corpus_tf_idf = vectorizer.transform(test_data)

        ### Baseline
        classifier = DummyClassifier(strategy='most_frequent')
        classifier.fit(train_corpus_tf_idf, train_labels)
        score = classifier.score(test_corpus_tf_idf, test_labels)
        print('Baseline < most_frequent > score: ', round(score * 100, 2), '%')

        ### Classification
        model1 = LinearSVC()
        model2 = MultinomialNB()
        model1.fit(train_corpus_tf_idf, train_labels)
        model2.fit(train_corpus_tf_idf, train_labels)

        score1 = model1.score(test_corpus_tf_idf, test_labels)
        score2 = model2.score(test_corpus_tf_idf, test_labels)

        print('Classification score with Tf-Idf and LinearSVC: ', round(score1 * 100, 2), '%')
        print('Classification score with Tf-Idf and MultinomialNB: ', round(score2 * 100, 2), '%')


######################################################################################
#############################  Helping classes  ######################################
######################################################################################


class MetaFeatures(object):
    def __init__(self, id, audio_features, text_features):
        self.id = id
        self.audio_fatures = audio_features
        self.text_features = text_features

    def get_Id(self):
        return self.id

    def get_audio_features(self):
        return self.audio_fatures

    def get_text_features(self):
        return self.text_features


class LabeledLineSentence(object):
    '''
    LabeledLineSentence contains two arrays: 
    - sentences[]: 
      contains the words of the document and its label with an unique id like:
      [[id, contributor, ...][classical_00]]
    - genre_labels[]: 
      same order as the sentences array but contains only the label like:
      [classical, classical, irish, ...]
    '''

    def __init__(self, filename):
        self.filename = filename

    def to_array(self):
        self.sentences = []
        self.genre_labels = []

        with open(self.filename, newline='', encoding="UTF-8") as file:

            line = ''
            myreader = csv.reader(file, delimiter=';')
            myList = list(myreader)

            first_row = True
            item_no = 0
            classical_index = 0
            irish_index = 0
            folclore_index = 0
            invironment_index = 0
            spoken_index = 0

            for row in myList:
                sentence = ''
                label = ''
                item_no = item_no + 1

                # skip titles
                if first_row:
                    first_row = False
                    continue

                row_index = 0
                for column in row:
                    row_index = row_index + 1
                    # use only the following rows from the csv:
                    #   - (id)
                    #   - contributor (not translated)
                    #   - creator (only translated if country == france)
                    #   - date
                    #   - describtion - trans
                    #   - spatial
                    #   - subject - trans
                    #   - type - trans
                    #   - year
                    if (row_index < len(row)) and row_index in [1, 2, 5, 7, 8, 16, 17, 18, 19, 20]:
                        sentence = sentence + ' ' + column
                    else:
                        label = column

                        ## count from 1 - n for every genre seperately
                        if label == 'classical':
                            classical_index = classical_index + 1
                            item_no = classical_index
                        if label == 'irish':
                            irish_index = irish_index + 1
                            item_no = irish_index
                        if label == 'folklore':
                            folclore_index = folclore_index + 1
                            item_no = folclore_index
                        if label == 'invironment':
                            invironment_index = invironment_index + 1
                            item_no = invironment_index
                        if label == 'spoken':
                            spoken_index = spoken_index + 1
                            item_no = spoken_index

                self.genre_labels.append(label)
                self.sentences.append(
                    gensim.models.doc2vec.LabeledSentence(sentence.split(), [label + '_%s' % item_no]))
                # self.sentences.append(LabeledSentence(sentence.split(), [label]))

        return [self.sentences, self.genre_labels]

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
