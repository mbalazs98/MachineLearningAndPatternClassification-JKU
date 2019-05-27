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

    def __init__(self, predictions=None, labels=None):
        self.predictions = predictions
        self.labels = labels
        self.logger = logging.getLogger("main")

    def get_gain(self, predictions=None, labels=None):
        self.logger.info("Get gain method...")
        if predictions is not None:
            self.predictions = predictions
        if labels is not None:
            self.labels = labels
        if self.labels is not None:
            return self.get_exact_gain()
        return self.get_expected_gain()

    def get_exact_gain(self):
        """
        Exact gain means that we know the labels for the dataset. (Taking into consideration the penalty cost)
        :return:
        """
        self.logger.info("Get exact gain ...")
        total_gain = 0
        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            actual_label = self.labels[i]
            if isinstance(pred, str):
                pred = self.LABELS_TO_KEY[pred]
            if isinstance(actual_label, str):
                actual_label = self.LABELS_TO_KEY[actual_label]
            total_gain += self.COST_MATRIX[actual_label][pred]
        self.logger.info("Exact gain = {}".format(total_gain))
        return total_gain

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
        return total_gain
