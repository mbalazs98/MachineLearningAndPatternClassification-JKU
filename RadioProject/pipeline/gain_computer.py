import logging


class GainComputer:
    """
    Compute the expected gain for a predicted sequence.
    If the labels are provided, than the exact gain will be computed (based on GAIN MATRIX).

    GAIN MATRIX

                                            predicted class

    --------------------------------------------------------------------------------------------------
                                    music-only              music&speech                    nomusic
    --------------------------------------------------------------------------------------------------
            |   music-only      |     1.00                     0.20                           0.00
     true   |   music&speech    |    -2.00                     0.20                           0.00
     class  |   nomusic         |    -3.00                    -1.00                           0.00
            |                   |
    --------------------------------------------------------------------------------------------------
    """
    COST_MATRIX = [
        [1.00, 0.20, 0.00],
        [-2.00, 0.20, 0.00],
        [-3.00, -1.00, 0.00],
    ]

    LABELS_TO_KEY = {
        "music-only": 0,
        "music&speech": 1,
        "nomusic": 2
    }

    def __init__(self, predictions=None, music_labels=None, speech_labels=None):
        self.predictions = predictions
        self.music_labels = music_labels
        self.speech_labels = speech_labels
        self.logger = logging.getLogger("main")
        self.expected_gain = None
        self.exact_gain = None
        self.maximum_gain = None

    def get_gain(self, predictions=None, music_labels=None, speech_labels=None):
        self.logger.info("Get gain method...")
        if predictions is not None:
            self.predictions = predictions
        if music_labels is not None:
            self.music_labels = music_labels
        if speech_labels is not None:
            self.speech_labels = speech_labels
        if self.music_labels is not None and self.speech_labels is not None:
            return self.get_exact_gain()
        return self.get_expected_gain()

    def get_exact_gain(self):
        """
        Exact gain means that we know the labels for the dataset. (Taking into consideration the penalty cost)
        :return:
        """
        self.logger.info("Get exact gain ...")
        total_gain = 0
        combined_labels = self.get_combined_labels_from_bi_labels(self.music_labels, self.speech_labels)
        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            actual_label = combined_labels[i]
            if isinstance(pred, str):
                pred = self.LABELS_TO_KEY[pred]
            if isinstance(actual_label, str):
                actual_label = self.LABELS_TO_KEY[actual_label]
            total_gain += self.COST_MATRIX[actual_label][pred]
        self.logger.info("Exact gain = {}".format(total_gain))
        self.exact_gain = total_gain
        return self.exact_gain

    def get_expected_gain(self):
        """
        Expected gain means that we compute the gain based on our predictions (we consider them to be correct)
        :return:
        """
        self.logger.info("Get expected gain ...")
        total_gain = 0

        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            if isinstance(pred, str):
                pred = self.LABELS_TO_KEY[pred]
            total_gain += self.COST_MATRIX[pred][pred]
        self.logger.info("Expected gain = {}".format(total_gain))
        self.exact_gain = total_gain
        return self.exact_gain

    def get_max_possible_gain(self, music_data_set, speech_data_set):
        """
        Gets labeled data and using following mapping between music and speech predictions and combined predictions
        compute the maximum gain
        [music_label speech_label]                  -> combined_label
        [music      no_speech]     /   [1 0]        -> only_music       / 0
        [music      speech]        /   [1 1]        -> music&speech     / 1
        [no_music   speech]        /   [0 1]        -> no_music         / 2
        [no_music   no_speech]     /   [0 0]        -> no_music         / 2

        :param music_data_set:
        :param speech_data_set:
        :return: the maximum gain that can be obtained
        """
        self.logger.info(
            "Start computing maximum gain that can be obtained for {} instances".format(music_data_set.shape[0]))
        total_gain = 0
        music_labels = music_data_set["class"].tolist()
        speech_labels = speech_data_set["class"].tolist()
        combined_labels = self.get_combined_labels_from_bi_labels(music_labels, speech_labels)
        for combined_label in combined_labels:
            total_gain += self.COST_MATRIX[self.LABELS_TO_KEY[combined_label]][self.LABELS_TO_KEY[combined_label]]

        self.maximum_gain = total_gain
        self.logger.info(
            "Maximum gain that can be obtain from current {} instances is {}".format(music_data_set.shape[0],
                                                                                     self.maximum_gain))
        return self.maximum_gain

    @staticmethod
    def get_combined_labels_from_bi_labels(music_labels, speech_labels):
        combined_labels = list()
        for i in range(len(music_labels)):
            if music_labels[i] == b'music' and speech_labels[i] == b'no_speech':
                combined_labels.append("music-only")
            elif music_labels[i] == b'music' and speech_labels[i] == b'speech':
                combined_labels.append("music&speech")
            else:
                combined_labels.append("nomusic")
        return combined_labels
