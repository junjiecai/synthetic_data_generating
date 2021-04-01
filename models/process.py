import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.simulation.playout import simulator
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.util import constants
from pm4py.statistics.traces.log import case_statistics
from pm4py.streaming.importer.csv import importer as streaming_csv_importer
from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.objects.conversion.log import converter as log_conversion
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import sampling, sorting, index_attribute

class Process:
    def __init__(self, event_data):
        self.event_data = event_data
        #log = xes_importer.apply(os.path.join("tests","input_data","running-example.xes"))

    def train(self):
        #Alpha Miner

        self.net, self.initial_marking, self.final_marking = alpha_miner.apply(self.event_data)
        #Heuristics Miner
        #self.net, self.initial_marking, self.final_marking = heuristics_miner.apply(self.event_data, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})


    def generate(self, n=100):
        '''n is number of patients'''
        simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: n})
        return simulated_log

    def evaluate(self,log):
        
        gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking)
        pn_visualizer.view(gviz)
        #x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
        #gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)

        #gviz = graphs_visualizer.apply_semilogx(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)
        
        fitness = replay_fitness_evaluator.apply(log, self.net, self.initial_marking, self.final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        print(fitness)                            

def importExportCSVtoXES(inputdata,outputdata):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        dummy_variable = "dummy_value"
        df = pd.read_csv(os.path.join("tests", "input_data",  inputdata))
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        event_log = log_conversion.apply(df, variant=log_conversion.TO_EVENT_STREAM)
        event_log = sorting.sort_timestamp(event_log)
        event_log = sampling.sample(event_log)
        event_log = index_attribute.insert_event_index_as_event_attribute(event_log)
        log = log_conversion.apply(event_log)
        log = sorting.sort_timestamp(log)
        log = sampling.sample(log)
        log = index_attribute.insert_trace_index_as_event_attribute(log)
        xes_exporter.apply(log, os.path.join("tests", "input_data",  outputdata))
        #log_imported_after_export = xes_importer.apply(os.path.join("tests", "input_data",  outputdata))
        #os.remove(os.path.join("tests", "input_data",  outputdata))
    
if __name__ == "__main__":
    importExportCSVtoXES("event_e.csv","event_e.xes")
    #print(os.path.abspath(os.path.join( "tests", "input_data", "running-example.xes")))
    event_data =xes_importer.apply(os.path.join("tests", "input_data", "event_e.xes"))
    pm = Process(event_data)
    pm.train()
    simulation_data=pm.generate(1000)
    print(simulation_data)
    #pm.evaluate(simulation_data)

