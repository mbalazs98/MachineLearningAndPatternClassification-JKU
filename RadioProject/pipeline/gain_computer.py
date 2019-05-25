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
        "musix&speech": 1,
        "nomusic": 2
    }

    def __init__(self, predictions=None, labels=None):
        self.predictions = predictions
        self.labels = labels

    def get_gain(self, predictions=None, labels=None):
        if predictions is not None:
            self.predictions = predictions
        if labels is not None:
            self.labels = labels
        if self.labels is not None:
            return self.get_exact_gain()
        return self.get_expected_gain()

    def get_exact_gain(self):
        total_gain = 0
        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            actual_label = self.labels[i]
            if isinstance(pred, str):
                pred = self.LABELS_TO_KEY[pred]
            if isinstance(actual_label, str):
                actual_label = self.LABELS_TO_KEY[actual_label]
            total_gain += self.COST_MATRIX[actual_label][pred]
        return total_gain

    def get_expected_gain(self):
        total_gain = 0
        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            if isinstance(pred, str):
                pred = self.LABELS_TO_KEY[pred]
            total_gain += self.COST_MATRIX[pred][pred]
        return total_gain
