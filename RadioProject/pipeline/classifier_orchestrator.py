import logging

import joblib


class ClassifierOrchestrator:
    MUSIC_CLASSIFIERS_OUTPUT_INDEX = 0
    SPEECH_CLASSIFIERS_OUTPUT_INDEX = 1
    CLASSIFIER_PATHS = {
        'music': [
            r'd:\Radio-Project-models\random_forest\rf-10-music-nestimators_100-max_depth35.joblib',
        ],
        'speech': [
            r'd:\Radio-Project-models\random_forest\rf-10-speech-nestimators_100-max_depth30.joblib'
        ]
    }

    LABEL_TO_INT_KEY = {
        "b'music'": 1,
        "b'no_music'": 0,
        "b'nomusic'": 0,
        "b'speech'": 1,
        "b'no_speech'": 0,
        "b'nospeech'": 0
    }

    ONLY_MUSIC = 0
    MUSIC_AND_SPEECH = 1
    NO_MUSIC = 2

    # BI_PREDICTION_TO_FINAL = {
    #     [1, 0]: 0,
    #     [1, 1]: 1,
    #     [0, 1]: 2,
    #     [0, 0]: 2
    # }

    def __init__(self, music_classifiers=None, speech_classifiers=None):
        if speech_classifiers is None:
            speech_classifiers = list()
        if music_classifiers is None:
            music_classifiers = list()
        self.music_classifiers = music_classifiers
        self.speech_classifiers = speech_classifiers
        self.logger = logging.getLogger("main")

    def load_classifiers(self):
        self.logger.info("Start loading classifiers")
        for key in self.CLASSIFIER_PATHS:
            for classifier_path in self.CLASSIFIER_PATHS[key]:
                self.__load_classifier(classifier_path, key)

    def __load_classifier(self, file_path, classifier_type):
        """

        :param file_path: path to the file where the model is saved
        :param classifier_type: str music or speech
        :return:
        """
        model = joblib.load(file_path)
        self.logger.info("Loaded classifier from: {}".format(file_path))
        self.__add_classifier(model, classifier_type)

    def __add_classifier(self, classifier, classifier_type):
        """

        :param classifier: classifier object
        :param classifier_type: 'music' or 'speech'
        :return: None
        """
        if classifier_type == "music":
            self.music_classifiers.append(classifier)
        elif classifier_type == "speech":
            self.speech_classifiers.append(classifier)
        else:
            raise ValueError("Bad value for classifier type {}".format(classifier_type))
        self.logger.info("Added {} classifier".format(classifier_type))

    def get_all_predictions(self, music_data_set, speech_data_set):
        """
        Using all classifiers (self.music_classifiers and s elf.speech_classifiers) predicts the output for datasets:
        (music_data_set and speech_data_set)

        :return:
            [
                1 instance ->
                [
                                    music_classifiers_output  -> [1 1 1 0...],
                                    speech_classifiers_output -> [0 0 0 1...]
                ]
                2 instance ->
                .....
            ]

        """
        self.logger.info("Get all predictions ...")
        # delete the prediction col from dataframe
        # TODO coment this if dataset has no prediction column

        columns = music_data_set.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]
        music_data_set = music_data_set[cols_to_use]

        columns = speech_data_set.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]
        speech_data_set = speech_data_set[cols_to_use]
        self.logger.info("Class column was removed from datasets")
        # TODO until here

        result = list()
        for i in range(music_data_set.shape[0]):
            result.append([list(), list()])
        self.logger.info(
            "Get MUSIC classifiers output. {} music classifiers are used.".format(len(self.music_classifiers)))
        for music_classifier in self.music_classifiers:
            current_pred = music_classifier.predict(music_data_set)
            for i in range(len(current_pred)):
                result[i][self.MUSIC_CLASSIFIERS_OUTPUT_INDEX].append(self.LABEL_TO_INT_KEY[current_pred[i]])
        self.logger.info(
            "Get SPEECH classifiers output. {} speech classifiers are used.".format(len(self.speech_classifiers)))
        for speech_classifier in self.speech_classifiers:
            current_pred = speech_classifier.predict(speech_data_set)
            for i in range(len(current_pred)):
                result[i][self.SPEECH_CLASSIFIERS_OUTPUT_INDEX].append(self.LABEL_TO_INT_KEY[current_pred[i]])
        self.logger.info("Returning all predictions for {} instances".format(music_data_set.shape[0]))
        return result

    def get_final_prediction_for_one_instance(self, instance_predictions):
        """
        Uses the majority vote mechanism
        :param instance_predictions:
        1 instance ->
                [
                                    music_classifiers_output  -> [1 1 1 0...],
                                    speech_classifiers_output -> [0 0 0 1...]
                ]
        :return: [music_final_pred speech_final_pred]
        """
        from collections import Counter
        try:
            c = Counter([i for i in instance_predictions[self.MUSIC_CLASSIFIERS_OUTPUT_INDEX]])
            music_final_output, count = c.most_common()[0]
            c = Counter([i for i in instance_predictions[self.SPEECH_CLASSIFIERS_OUTPUT_INDEX]])
            speech_final_output, count = c.most_common()[0]
        except:
            raise ValueError("WTF??? {}".format(instance_predictions))
        return [music_final_output, speech_final_output]

    # todo @balazs the function below is the one
    def get_final_bi_predictions(self, all_predictions, weights=None):
        """

        :param weights:
        :param all_predictions:
            [
                1 instance ->[
                    music_classifiers_output  -> [1 1 1 0 ...],
                    speech_classifiers_output -> [0 0 0 1 ...]
                ]
                .....
            ]
        :return:
        [
            1 instance ->[
                music final pred -> 1,
                speech final prediction -> 0
            ]
            .....
        ]
        """
        self.logger.info("Get final bi prediction for {} instances.".format(len(all_predictions)))
        final_bi_predictions = list()
        for i in range(len(all_predictions)):
            final_bi_predictions.append(self.get_final_prediction_for_one_instance(all_predictions[i]))
        self.logger.info("Returning final bi prediction for {} instances.".format(len(all_predictions)))
        return final_bi_predictions

    def get_final_predictions(self, music_data_set, speech_data_set, weights=None):
        """

        :param speech_data_set:
        :param music_data_set:
        :param weights:
        :return:  list of final predictions
        {
            "music-only": 0,
            "music&speech": 1,
            "no_music": 2
        }
        [1 , 0 , 2 ...]
        """
        self.logger.info("Get final predictions ...")
        all_predictions = self.get_all_predictions(music_data_set, speech_data_set)
        final_bi_predictions = self.get_final_bi_predictions(all_predictions, weights)
        final_predictions = list()
        self.logger.info("Start computing final predictions for {} instances.".format(len(final_bi_predictions)))
        for bi_pred in final_bi_predictions:
            if bi_pred == [1, 0]:
                final_predictions.append(self.ONLY_MUSIC)
            elif bi_pred == [1, 1]:
                final_predictions.append(self.MUSIC_AND_SPEECH)
            elif bi_pred == [0, 0] or bi_pred == [0, 1]:
                final_predictions.append(self.NO_MUSIC)
            else:
                raise ValueError(
                    "Unexpected value for bi_prediction {}. Should be [0,0],[0,1],[1,0] or [1,1]".format(bi_pred))
        self.logger.info("Returning computing final predictions for {} instances.".format(len(final_bi_predictions)))
        return final_predictions
