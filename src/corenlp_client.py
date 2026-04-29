# Dummy CoreNLP client for MM-Neuron
class CoreNLP:
    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, text):
        return text.split()

    def pos_tag(self, text):
        return [(w, "NN") for w in text.split()]
