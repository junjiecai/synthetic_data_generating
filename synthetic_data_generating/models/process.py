import os
import pm4py
from pm4py import util
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
from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
from pm4py.objects.conversion.log import converter as log_conversion
import pandas as pd
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.util import sampling, sorting, index_attribute
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.simulation.montecarlo import simulator as montecarlo_simulation
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.log.util import dataframe_utils
from pm4py.util import constants, xes_constants
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.statistics.start_activities.log import get as start_activities
from pm4py.statistics.end_activities.log import get as end_activities
import os
_debug = 0
class Process:
    def __init__(self, event_data):
        self.file = open("log/log.txt","a+")
        df = pm4py.format_dataframe(event_data, case_id='id', activity_key='event_type', timestamp_key='time')
        #df = dataframe_utils.convert_timestamp_columns_in_df(df)
        df["case:concept:name"]=df["case:concept:name"].astype("int64")
        self.event_data = log_conversion.apply(df, variant=log_conversion.TO_EVENT_LOG)
        #df.to_excel("test.xlsx")
        

    def train(self):
        #Alpha Miner
        if _debug:
            self.net, self.initial_marking, self.final_marking = inductive_miner.apply(self.event_data)
            self.draw_inductive_frequency(self.net, self.initial_marking, self.final_marking,self.event_data,"image/inductive_frequency_pn.png")
            tree = inductive_miner.apply_tree(self.event_data)
            gviz = pt_visualizer.apply(tree)
            pt_visualizer.save(gviz,"image/process_tree.png")

        self.dfg = dfg_discovery.apply(self.event_data, variant=dfg_discovery.Variants.FREQUENCY)
        
        #gviz = dfg_visualizer.apply(self.dfg,variant=dfg_visualizer.Variants.FREQUENCY)
        #dfg_visualizer.view(gviz)
      
        
        #from pm4py.statistics.start_activities.log import get as start_activities
        #from pm4py.statistics.end_activities.log import get as end_activities
        #self.sa = start_activities.get_start_activities(self.event_data)
        #self.ea = end_activities.get_end_activities(self.event_data)
        #from pm4py.objects.conversion.dfg import converter
        #self.net, self.initial_marking, self.final_marking = converter.apply(self.dfg, variant=converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE,
        #parameters={converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE.value.Parameters.START_ACTIVITIES: self.sa,
        #converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE.value.Parameters.END_ACTIVITIES: self.ea})
    
    def draw_inductive_frequency(self,net, initial_marking, final_marking,log,filename):
        parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
        gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters, variant=pn_visualizer.Variants.FREQUENCY, log=log)
        pn_visualizer.save(gviz, filename)
    
    def generate(self, n=10000):
        '''n is number of patients'''
        #simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.CASE_ID_KEY: n})
        if _debug:
            simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: n})
            self.draw_inductive_frequency(self.net, self.initial_marking,self.final_marking,simulated_log,"image/simulated_inductive_frequency.png")
            self.foot_print_evaluate(simulated_log)

        # from pm4py.objects.process_tree import semantics
        # simulated_log = semantics.generate_log(self.tree, no_traces=100)
        # print(len(simulated_log))
        # from pm4py.algo.simulation.montecarlo import simulator as montecarlo_simulation
        # from pm4py.algo.conformance.tokenreplay.algorithm import Variants
        # parameters = {}
        # parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_ENABLE_DIAGNOSTICS] = False
        # parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_MAX_THREAD_EXECUTION_TIME] = 5
        # simulated_log, res = montecarlo_simulation.apply(self.event_data, self.net, self.initial_marking,self.final_marking, parameters=parameters)
        log = self.event_data
        dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.FREQUENCY)

        sa = start_activities.get_start_activities(log)
        ea = end_activities.get_end_activities(log)
        from pm4py.statistics.traces.log import case_arrival
        ratio = case_arrival.get_case_arrival_avg(log)
        print(ratio)
        from pm4py.objects.conversion.dfg import converter
        net, im, fm = converter.apply(dfg, variant=converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE,
                              parameters={converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE.value.Parameters.START_ACTIVITIES: sa,
                                          converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE.value.Parameters.END_ACTIVITIES: ea})
        from pm4py.algo.simulation.montecarlo import simulator as montecarlo_simulation
        from pm4py.algo.conformance.tokenreplay.algorithm import Variants

        parameters = {}
        parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_NUM_SIMULATIONS] = n
        parameters[montecarlo_simulation.Variants.PETRI_SEMAPH_FIFO.value.Parameters.PARAM_CASE_ARRIVAL_RATIO] = ratio
        simulated_log, res = montecarlo_simulation.apply(log,net, im, fm , parameters=parameters)
        print(res["total_cases_time"])
        net, initial_marking, final_marking = inductive_miner.apply(simulated_log)
        self.draw_inductive_frequency(net, initial_marking, final_marking,simulated_log,"image/simulated_dfg_inductive_frequency.png")
        self.foot_print_evaluate(simulated_log)

        simulated_log = log_conversion.apply(simulated_log, variant=log_conversion.TO_DATA_FRAME)
        simulated_log = simulated_log.rename(columns={constants.CASE_CONCEPT_NAME:"id", xes_constants.DEFAULT_NAME_KEY:"event_type",
                             xes_constants.DEFAULT_TIMESTAMP_KEY:"time"})

        #from pm4py.algo.simulation.playout import simulator
        #simulated_log = simulator.apply(self.net, self.initial_marking,self.final_marking, variant=simulator.Variants.STOCHASTIC_PLAYOUT)
        return simulated_log

    def evaluate(self,log,show=False):
        #Petri Net
        gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking)
        pn_visualizer.view(gviz)
        if show:
            x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
            gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
            graphs_visualizer.view(gviz)
        
        

        #gviz = graphs_visualizer.apply_semilogx(x, y, variant=graphs_visualizer.Variants.CASES)
        #graphs_visualizer.view(gviz)
        #dfg, start_activities, end_activities = pm4py.discover_dfg(log)
        #pm4py.view_dfg(dfg, start_activities, end_activities)
        start_activities = pm4py.get_start_activities(log)
        end_activities = pm4py.get_end_activities(log)
        self.file.write("Start activities: {}\nEnd activities: {}\n".format(start_activities, end_activities))
        fitness = replay_fitness_evaluator.apply(log, self.net, self.initial_marking, self.final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        self.file.write("fitness:{}\n".format(fitness))

    def foot_print_evaluate_dataframe(self, df_log):
        df = pm4py.format_dataframe(df_log, case_id='id', activity_key='event_type', timestamp_key='time')
        df =log_conversion.apply(df, variant=log_conversion.TO_EVENT_LOG)
        self.foot_print_evaluate(df)

    def foot_print_evaluate(self,log):
        #foot print evaluate
        from pm4py.algo.discovery.footprints import algorithm as fp_discovery
        tree = inductive_miner.apply_tree(self.event_data)
        fp_simulation_data_log = fp_discovery.apply(log, variant=fp_discovery.Variants.ENTIRE_EVENT_LOG)
        fp_log = fp_discovery.apply(self.event_data, variant=fp_discovery.Variants.ENTIRE_EVENT_LOG)
        #fp_trace_trace = fp_discovery.apply(self.event_data, variant=fp_discovery.Variants.TRACE_BY_TRACE)
        #fp_tree = fp_discovery.apply(tree, variant=fp_discovery.Variants.PROCESS_TREE)
        from pm4py.visualization.footprints import visualizer as fp_visualizer
        gviz = fp_visualizer.apply(fp_log, fp_simulation_data_log, parameters={fp_visualizer.Variants.COMPARISON.value.Parameters.FORMAT: "png"})
        fp_visualizer.view(gviz)
        fp_visualizer.save(gviz,"image/matrix.png")
        
        from pm4py.algo.conformance.footprints import algorithm as fp_conformance
        conf_result = fp_conformance.apply(fp_log, fp_simulation_data_log, variant=fp_conformance.Variants.LOG_EXTENSIVE)
        from pm4py.algo.conformance.footprints.util import evaluation

        fitness = evaluation.fp_fitness(fp_log, fp_simulation_data_log, conf_result)
        precision = evaluation.fp_precision(fp_log, fp_simulation_data_log)
        self.file.write("fp:precision:{},fitness:{},conf_result:{}\n".format(precision,fitness,conf_result))
        #end foot print evaluate
        #ALIGNMENT_BASED evaluate
        fitness = replay_fitness_evaluator.apply(log,self.net, self.initial_marking, self.final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        self.file.write("TOKEN_BASED.fitness:{}\n".format(fitness))
        #end ALIGNMENT_BASED evaluate

        #Precision
        prec = precision_evaluator.apply(log,self.net, self.initial_marking, self.final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        self.file.write("prec:{}\n".format(prec))
        #end Precision
        #Generalization
        from pm4py.algo.evaluation.generalization import evaluator as generalization_evaluator
        gen = generalization_evaluator.apply(log, self.net, self.initial_marking, self.final_marking)
        self.file.write("gen:{}\n".format(gen))
        #end Generalization
        #Simplicity
        from pm4py.algo.evaluation.simplicity import evaluator as simplicity_evaluator
        simp = simplicity_evaluator.apply(self.net)
        self.file.write("simp:{}\n".format(simp))
        #end Simplicity
        
        #Earth Mover Distance
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.statistics.variants.log import get as variants_module
        language = variants_module.get_language(self.event_data)
        self.file.write("source:{}\n".format(language))
        language = variants_module.get_language(log)
        self.file.write("simulation:{}\n".format(language))
        #end Earth Mover Distance
        self.file.flush()
    
    def close(self):
        self.file.close()

