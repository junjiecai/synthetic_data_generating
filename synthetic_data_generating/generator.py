from synthetic_data_generating.models import Process, Tabular
import os
from pandas import DataFrame
from os.path import join
import pandas as pd

import pathlib
base_path = pathlib.Path(__file__).parent.absolute()

# todo:
# id for process

class Generator:
    def __init__(self, one_to_one_path, path_to_many_path):
        self._one_to_one_data = self._load_data(one_to_one_path)
        self._one_to_many_data = self._load_data(path_to_many_path)

    def _load_data(self, path, index_cols=None) -> dict:
        files = os.listdir(path)
        files = filter(lambda f: f.endswith('.xlsx'), files)
        mapping = {}
        for filename in files:
            filename_base = filename.split('.')[0]
            df = pd.read_excel(join(path, filename))
            if index_cols:
                df = df.set_index(index_cols)

            mapping[filename_base] = df
        return mapping

    def _combine_one_to_one_data(self):
        return pd.concat([df.set_index('id') for df in self._one_to_one_data.values()], axis=1).reset_index()
    # 

    def _combine_event_logs(self):
        data = []
        for filename_base, df in  self._one_to_many_data.items():
            sub_df = df[['id', 'time']]
            sub_df['event_type'] = filename_base
            data.append(
                sub_df
            )

        return pd.concat(data)
    # 
    # 
    def train(self):
        combined_tabular_data = self._combine_one_to_one_data().set_index('id')
        self.tabular_generator = Tabular(combined_tabular_data)
        self.tabular_generator.train(n_epochs=10)
    # 
        event_logs_data = self._combine_event_logs()
        event_log_generator = Process(event_logs_data)
        event_log_generator.train()
        self.event_log_generator = event_log_generator
    # 
        self.properties_data_generators = {}
        for event, data in self._one_to_many_data.items():
            generator = Tabular(data.drop(['id', 'time'], axis=1))
            generator.train(n_epochs=10)
            self.properties_data_generators[event] = generator
            print("Tabular generator for {} is trained".format(event))

    def split_tabular_data(self, generated_combined_tabular_data):
        new_data = {}
        for filename_base, df in self._one_to_one_data.items():
            new_data[filename_base] = generated_combined_tabular_data[list(df.columns)]

        return new_data

    def generate_event_data(self, event_logs, event_property_data):
        new_data = {}
        for event, df in event_property_data.items():
            sub_event_logs = event_logs.loc[event_logs['event_type'] == event]
            df.index = sub_event_logs.index
            new_data[event] = pd.concat([sub_event_logs, df], axis=1)

        return new_data

    def generate(self, n=100):
        generated_combined_tabular_data = self.tabular_generator.generate(n)
        generated_combined_tabular_data['id'] = list(range(n))
        generated_tabular_data = self.split_tabular_data(generated_combined_tabular_data)

        event_logs = self.event_log_generator.generate(n)

        event_property_data = {}
        for event, generator in self.properties_data_generators.items():
            record_n = (event_logs['event_type'] == event).sum()
            event_property_data[event] = generator.generate(record_n)

        event_data = self.generate_event_data(event_logs, event_property_data)

        return generated_tabular_data, event_data


if __name__ == '__main__':
    g = Generator(join(base_path, 'one_to_one'), join(base_path, 'one_to_many'))
    g.train()
    a, b = g.generate(10)
    print(1)
