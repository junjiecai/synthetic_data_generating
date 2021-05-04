from synthetic_data_generating.models import Process, Tabular
import os
from pandas import DataFrame
from os.path import join
import pandas as pd

import pathlib
base_path = pathlib.Path(__file__).parent.absolute()

class Generator:
    def __init__(self, one_to_one_path, path_to_many_path):
        self._one_to_one_data = self._load_one_to_one_data(one_to_one_path)
        self._one_to_many_data = self._load_one_to_many_data(path_to_many_path)

    def _load_one_to_one_data(self, one_to_one_path):
        # files = os.listdir(one_to_one_path)
        # files = filter(lambda f: f.endswith('.xlsx'), files)
        #
        # mapping = {}
        # data = []
        # for filename in files:
        #
        #     df = pd.read_excel(join(one_to_one_path, filename))
        #     data.append(df)
        #
        #     columns = list(df.columns)
        #
        #     filename_base = filename.split('.')[0]
        #     mapping.update(dict(zip(columns, [filename_base]*len(columns))))
        #
        # df_all = pd.concat(data)
        #
        # return df_all, mapping

        return self._load_data(one_to_one_path)


    def _load_one_to_many_data(self, one_to_many_path):
        path = one_to_many_path

        return self._load_data(path)

    def _load_data(self, path, index_cols=None):
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
        # combined_tabular_data = self._combine_one_to_one_data().set_index('id')
        # self.tabular_generator = Tabular(combined_tabular_data)
        # self.tabular_generator.train()
    # 
        event_logs_data = self._combine_event_logs()
        event_log_generator = Process(event_logs_data)
        event_log_generator.train()
        self.event_log_generator = event_log_generator
    #     self.event_log_generator = event_log_generator
    # 
    #     self.properties_data_generators = {}
    #     for event, data in self.one_to_many_data.items():
    #         generator = Tabular(data)
    #         generator.train()
    #         properties_data_generators[event] = generator
    # 
    def split_tabular_data(self, generated_combined_tabular_data):
        new_data = {}
        for filename_base, df in self._one_to_many_data.items():
            new_data[filename_base] = generated_combined_tabular_data[['id']+list(df.columns)]

        return new_data

    # 
    # def get_patient_event_size(self):
    #     raise NotImplementedError
    # 
    # def generate_event_data(self):
    #     raise NotImplementedError
    # 
    def generate(self, n=100):
        # generated_combined_tabular_data = self.tabular_generator.generate(n)
        # generated_tabular_data = self.split_tabular_data(generated_combined_tabular_data)

        event_logs = self.event_log_generator.generate(n)
        # patient_event_size = self.get_patient_event_size()
        #
        # event_property_data = {}
        # for event, generator in self.properties_data_generators.items():
        #     event_property_data[event] = generator.generate(patient_event_size)
        #
        # event_data = self.generate_event_data(event_logs, event_property_data)

        return event_logs


if __name__ == '__main__':
    g = Generator(join(base_path, 'one_to_one'), join(base_path, 'one_to_many'))
    g.train()
    g.generate(10)
    print(1)
