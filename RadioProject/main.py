def load_data():
    pass


def get_classifiers_output(features, classifiers, classifiers_type):
    """

    :param classifiers: list of classifiers to use
    :param features: features needed for the classifier
    :param classifiers_type: music / speech
    :return: a list of outputs of each classifier
    """
    outputs = list()
    for i in classifiers:
        outputs.append(classifiers[i].pred(features))
    return outputs


if __name__ == "__main__":
    pass
