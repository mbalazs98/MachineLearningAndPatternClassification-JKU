import logging


class SmoothingOnFinalPredictions:
    def __init__(self, predictions):
        """

        :param predictions: list of final predictions [1 , 0 , 2 ...]  where:
        {
            "music-only": 0,
            "music&speech": 1,
            "no_music": 2
        }

        """
        self.logger = logging.getLogger("main")
        self.predictions = predictions
        self.postprocessed_predictions = None

    def __do_smoothing(self):
        """
        Add smoothing logic here
        :return: postprocessed predictions
        """
        self.postprocessed_predictions = self.predictions

    def get_smoothing_result(self):
        """

        :return:
        list of final predictions after smoothing [1 , 0 , 2 ...], where
        {
            "music-only": 0,
            "music&speech": 1,
            "no_music": 2
        }

        """
        self.logger.info("Start smoothing process for final(combined) predictions...")
        self.__do_smoothing()
        self.logger.info("End smoothing process for final(combined) predictions...")
        return self.postprocessed_predictions


class SmoothingOnBiPredictions:
    def __init__(self, bi_predictions):
        """

        :param bi_predictions: array like:
        [
            [1,0],
            [1,1],
            [0,0],
             ...
         ]

        described by:
        [
            1 instance ->[
                music final pred -> 1,
                speech final prediction -> 0
            ]
            .....
        ]
        """
        self.logger = logging.getLogger("main")
        self.bi_predictions = bi_predictions
        self.postporcessed_bi_predictions = None

    def __do_smoothing(self):
        """
        Add smoothing logic here
        :return: postprocessed bi_predictions
        """
        self.postporcessed_bi_predictions = self.bi_predictions

    def get_smoothing_result(self):
        """

        :return: array like:
        [
            [1,0],
            [1,1],
            [0,0],
             ...
         ]

        described by:
        [
            1 instance ->[
                music final pred -> 1,
                speech final prediction -> 0
            ]
            .....
        ]
        """
        self.logger.info("Start smoothing process for bi predictions...")
        self.__do_smoothing()
        self.logger.info("End smoothing process for bi predictions")
        return self.postporcessed_bi_predictions
