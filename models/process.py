class Process:
    def __init__(self, event_data):
        self.event_data = event_data

    def train(self):
        raise NotImplementedError

    def generate(self, n=100):
        '''n is number of patients'''
        raise NotImplementedError