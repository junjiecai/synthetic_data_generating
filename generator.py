from .models import Process, Tabular

class Generator:
    def __init__(self, one_to_one_path, path_to_many_path):
        self._one_to_one_data, self._name_source_mapping = self._load_one_to_one_data(one_to_one_path)
        self.one_to_many_data = self._load_one_to_mnay_data(path_to_many_path)

    def _load_one_to_one_data(self):
        raise NotImplementedError

    def _load_one_to_many_data(self):
        raise NotImplementedError

    def _combine_one_to_one_data(self):
        raise NotImplementedError

    def _combine_event_logs(self):
        raise NotImplementedError


    def train(self):
        combined_tabular_data = self._combine_one_to_one_data()
        self.tabular_generator = Tabular(combined_tabular_data)
        self.tabular_generator.train()

        event_logs_data = self._combine_event_logs()
        event_log_generator = Process(event_logs_data)
        event_log_generator.train()
        self.event_log_generator = event_log_generator

        self.properties_data_generators = {}
        for event, data in self.one_to_many_data.items():
            generator = Tabular(data)
            generator.train()
            properties_data_generators[event] = generator

    def split_tabular_data(self, data):
        raise NotImplementedError

    def get_patient_event_size(self):
        raise NotImplementedError

    def generate_event_data(self):
        raise NotImplementedError

    def generate(self, n=100):
        generated_combined_tabular_data = self.tabular_generator.generate(n)
        generated_tabular_data = self.split_tabular_data(generated_combined_tabular_data)

        event_logs = self.event_log_generator.generate(n)
        patient_event_size = self.get_patient_event_size()

        event_property_data = {}
        for event, generator in self.properties_data_generators.items():
            event_property_data[event] = generator.generate(patient_event_size)

        event_data = self.generate_event_data(event_logs, event_property_data)

        return generated_tabular_data, event_data

