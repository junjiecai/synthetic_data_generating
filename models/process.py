import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.simulation.playout import simulator
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.util import constants
from pm4py.statistics.traces.log import case_statistics

from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator

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
        
        #gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking)
        #pn_visualizer.view(gviz)
        #x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
        #gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)

        #gviz = graphs_visualizer.apply_semilogx(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)
        
        fitness = replay_fitness_evaluator.apply(log, self.net, self.initial_marking, self.final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        print(fitness)                            

if __name__ == "__main__":
    #event_data =xes_importer.apply(os.path.join("tests", "input_data", "running-example.xes"))
    #print(os.path.abspath(os.path.join( "tests", "input_data", "running-example.xes")))
    event_data =xes_importer.apply(os.path.join("tests", "compressed_input_data", "09_a32f0n00.xes.gz"))
    pm = Process(event_data)
    pm.train()
    simulation_data=pm.generate(1000)
    pm.evaluate(simulation_data)

