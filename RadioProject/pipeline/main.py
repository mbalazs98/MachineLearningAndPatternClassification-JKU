from common.io_operations import DataLoader
from pipeline.gain_computer import GainComputer


class RadioPipeline:
    """
    Orchestration class. Used to coordinate the pipeline process
    """

    def __init__(self, data_loader, classifier, evaluator):
        self.data_loader = data_loader
        self.classifier = classifier
        self.evaluator = evaluator

    def run(self):
        data_set = self.data_loader.load_data()
        predictions = self.classifier.get_predictions(data_set)
        total_gain = self.evaluator.get_gain(predictions)
        print("Total gain: {}".format(total_gain))


if __name__ == "__main__":
    data_loader = DataLoader()
    gain_computer = GainComputer()
