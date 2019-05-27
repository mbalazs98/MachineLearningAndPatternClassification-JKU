import datetime
import os

from common.io_operations import DataLoader
from pipeline.classifier_orchestrator import ClassifierOrchestrator
from pipeline.gain_computer import GainComputer


class RadioPipeline:
    """
    Orchestration class. Used to coordinate the pipeline process
    """

    def __init__(self, data_loader: DataLoader, classifier: ClassifierOrchestrator, evaluator: GainComputer):
        self.data_loader = data_loader
        self.classifier = classifier
        self.gain_computer = evaluator

    def run(self):
        music_data_set, speech_data_set = self.data_loader.load_data()
        self.classifier.load_classifiers()
        predictions = self.classifier.get_final_predictions(music_data_set, speech_data_set)
        total_gain = self.gain_computer.get_gain(predictions)
        print("Total gain: {}".format(total_gain))


def init_loging(log_dir_path="logs", log_file_name=None):
    import logging
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger("main")

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path,exist_ok=True)

    if log_file_name is None:
        log_file_name = datetime.datetime.now().strftime("day_%Y_%m_%d_time_%H_%M_%S") + ".log"

    fileHandler = logging.FileHandler(os.path.join(log_dir_path, log_file_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger


if __name__ == "__main__":
    logger = init_loging()
    logger.info("Start Radio project")
    data_loader = DataLoader(logger = logger)
    gain_computer = GainComputer()
    classifier_orchestrator = ClassifierOrchestrator()
    pipe = RadioPipeline(data_loader, classifier_orchestrator, gain_computer)
    pipe.run()
