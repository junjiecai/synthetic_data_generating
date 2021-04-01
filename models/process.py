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
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.simulation.montecarlo import simulator as montecarlo_simulation
import os

class Process:
    def __init__(self, event_data):
        self.event_data = event_data
        #log = xes_importer.apply(os.path.join("tests","input_data","running-example.xes"))

    def train(self):
        #Alpha Miner

        self.net, self.initial_marking, self.final_marking = alpha_miner.apply(self.event_data)
        #self.net, self.initial_marking, self.final_marking= inductive_miner.apply(self.event_data)
        #Heuristics Miner
        #self.net, self.initial_marking, self.final_marking = heuristics_miner.apply(self.event_data, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})


    def generate(self, n=100):
        '''n is number of patients'''
        simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: n})
        #simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.EXTENSIVE, parameters={simulator.Variants.EXTENSIVE.value.Parameters.MAX_TRACE_LENGTH: n})
        '''
        parameters = {}
        parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_ENABLE_DIAGNOSTICS] = False
        parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_MAX_THREAD_EXECUTION_TIME] = 5
        simulated_log, res = montecarlo_simulation.apply(self.event_data, self.net, self.initial_marking, self.final_marking, parameters=parameters)
        print("\n(Montecarlo - Petri net) case arrival ratio inferred from the log")
        print(res["median_cases_ex_time"])
        print(res["total_cases_time"])
        # perform the Montecarlo simulation with the arrival rate specified (the simulation lasts 5 secs)
        parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_CASE_ARRIVAL_RATIO] = 60
        simulated_log, res = montecarlo_simulation.apply(simulated_log, self.net, self.initial_marking, self.final_marking, parameters=parameters)
        print("\n(Montecarlo - Petri net) case arrival ratio specified by the user")
        print(res["median_cases_ex_time"])
        print(res["total_cases_time"])
        '''
        #from pm4py.algo.simulation.playout import simulator
        #simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.STOCHASTIC_PLAYOUT)
        return simulated_log

    def evaluate(self,log):
        
        #gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking)
        #pn_visualizer.view(gviz)
        #x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
        #gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)

        #gviz = graphs_visualizer.apply_semilogx(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)
        
        fitness = replay_fitness_evaluator.apply(log, self.net, self.initial_marking, self.final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        print(fitness)                            

def importExportCSV(inputdata):
        dummy_variable = "dummy_value"
        df = pd.read_csv(os.path.join("tests", "input_data",  inputdata))
        df["case:concept:name"]=df["case:concept:name"].astype("object")
        df["concept:name"]=df["concept:name"].astype("object")
        #df["time:timestamp"]=df["time:timestamp"].astype("date")
        df.info()
        
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        event_log = log_conversion.apply(df, variant=log_conversion.TO_EVENT_STREAM)
        event_log = sorting.sort_timestamp(event_log)
        event_log = sampling.sample(event_log)
        event_log = index_attribute.insert_event_index_as_event_attribute(event_log)
        log = log_conversion.apply(event_log)
        log = sorting.sort_timestamp(log)
        log = sampling.sample(log)
        log = index_attribute.insert_trace_index_as_event_attribute(log)
        return log

def importExportCSVtoXES(inputdata,outputdata):
        # to avoid static method warnings in tests,
        # that by construction of the unittest package have to be expressed in such way
        dummy_variable = "dummy_value"
        df = pd.read_csv(os.path.join("tests", "input_data",  inputdata))
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        df["case:concept:name"]=df["case:concept:name"].astype("object")
        df["Activity"]=df["Activity"].astype("object")
        df.info()
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
    simulation_data=pm.generate(10)
    print(simulation_data)
    xes_exporter.apply(simulation_data, os.path.join("tests", "input_data",  "simulation_event_e.xes"))
    pm.evaluate(simulation_data)

