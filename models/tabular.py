class Tabular:
    def __init__(self, data):
        self.data = data

    def train(self):
        raise NotImplementedError

    def generate(self, size=100):
        '''size can be list'''
        raise NotImplementedError



