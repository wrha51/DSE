import output_funcs as of
import classes as cls
import importlib.machinery
from copy import deepcopy
import sys
import msg
from msg import mem_scheme_fit_check
import argparse
import input_funcs
import time
from multiprocessing import Process, Value, Manager
from datetime import datetime
import evaluate
from classes.multi_manager import MultiManager
from im2col_funcs import im2col_layer_transform

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="Path to the mapping setting file")
    parser.add_argument("--mempool", help="Path to the memory pool file")
    parser.add_argument("--arch", help="Path to the architecture setting file")
    parser.add_argument("--set", help="Path to the main setting file")
    args = parser.parse_args()

    input_settings = input_funcs.get_input_settings(args.set, args.map, args.mempool, args.arch)

    '''
    Neural Network info can be defined by user (./NN_layer/XXNet.py) or be imported externally.
    By default, user-defined layer dimension is used (./NN_layer/XXNet.py).
    An example of loading model externally from Keras is provided.
    To try it, the below line "layer_spec, _ = input_funcs.get_layer_spec(..." need to be commented.
    '''
    # ---------------------------------------------------------------------------
    # import keras.applications
    # load_model = keras.applications.MobileNet()
    # layer_spec, layer_numbers = input_funcs.get_layer_spec(None, model=load_model)
    # input_settings.layer_number = layer_numbers
    # input_settings.layer_filename = '../../' + load_model.name
    # ---------------------------------------------------------------------------
    layer_spec, _ = input_funcs.get_layer_spec(input_settings, model=None)

    # Extract the layer information from the layer_spec
    layers = [cls.Layer.extract_layer_info(layer_spec.layer_info[layer_number])
              for layer_number in input_settings.layer_number]
    layers_saved = deepcopy(layers)

    layer_info_im2col = im2col_layer_transform(layer_spec.layer_info)
    layers_im2col = [cls.Layer.extract_layer_info(layer_info_im2col[layer_number])
                     for layer_number in input_settings.layer_number]

    # pw: short for pointwise.
    # When greedy mapping is discabled, the tool auto applies im2col transfer on pw layers
    # to speed up scheduling process without losing optimality.
    # Note that im2col is not compatible with greedy mapping, thus spatial_unrolling_mode 4, 5 are excluded.
    pw_im2col_flag = []
    if input_settings.im2col_enable_pw and (input_settings.spatial_unrolling_mode not in [4, 5]):
        layers_pw_speedup = []
        for idx, single_layer in enumerate(layers):
            if single_layer.FX == single_layer.FY == 1:  # or single_layer.OX == single_layer.OY == 1:
                layers_pw_speedup.append(layers_im2col[idx])
                layer_spec.layer_info[input_settings.layer_number[idx]] = \
                    layer_info_im2col[input_settings.layer_number[idx]]
                pw_im2col_flag.append(True)
            else:
                layers_pw_speedup.append(layers[idx])
                pw_im2col_flag.append(False)
        layers = layers_pw_speedup


    # If there are duplicate layers, set flag for the latter ones.
    # This flag will prevent the layer from being evaluated later on to speed up run.
    for idx, layer in enumerate(layers):
        layers_seen = layers[:idx]
        for idx_other, other in enumerate(layers_seen):
            if layer == other:
                layer.set_duplicate(input_settings.layer_number[idx_other])
                break

    # Setup a layer dictionary of following format for easy access
    # key: layer number
    # value: Layer class

    # layers_dict = {input_settings.layer_number[i]: layers[i]
    #             for i in range(len(layers))}

    results_path = input_settings.results_path

    if input_settings.mem_hierarchy_single_simulation:
        ma = [1]
    else:
        ma = input_settings.max_area
    tmp_mem_node_list = []

    print('ZigZag started running.')
    print('Target workload: %s | Layer(s): %s' % (input_settings.layer_filename.split('/')[-1],
                                                  input_settings.layer_number))
    t1 = time.time()
    if not input_settings.mem_hierarchy_single_simulation:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ' MSG started')
        mem_scheme_sim, mem_scheme_node_sim = msg.msg(input_settings.mem_pool,
                                                      input_settings.mac_array_info['array_size'],
                                                      ma, input_settings.utilization_rate_area,
                                                      input_settings.memory_hierarchy_ratio,
                                                      input_settings.prune_PE_RF,
                                                      input_settings.PE_RF_size_threshold,
                                                      input_settings.PE_RF_depth, input_settings.CHIP_depth,
                                                      input_settings.memory_scheme_hint, input_settings.mh_name,
                                                      tmp_mem_node_list,
                                                      input_settings.mem_hierarchy_single_simulation,
                                                      input_settings.banking, input_settings.L1_size,
                                                      input_settings.L2_size)
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ' MSG finished')
    if input_settings.mem_hierarchy_single_simulation:
        mem_scheme_sim, mem_scheme_node_sim = msg.msg(input_settings.mem_pool,
                                                      input_settings.mac_array_info['array_size'],
                                                      ma, input_settings.utilization_rate_area,
                                                      input_settings.memory_hierarchy_ratio,
                                                      input_settings.prune_PE_RF,
                                                      input_settings.PE_RF_size_threshold,
                                                      input_settings.PE_RF_depth, input_settings.CHIP_depth,
                                                      input_settings.memory_scheme_hint, input_settings.mh_name,
                                                      tmp_mem_node_list,
                                                      input_settings.mem_hierarchy_single_simulation,
                                                      input_settings.banking, input_settings.L1_size,
                                                      input_settings.L2_size)

    # CHECKS IF THE LAST LEVEL IN MEMORY HIERARCHY MEETS STORAGE REQUIREMENTS FOR OPERANDS
    if not mem_scheme_sim:
        raise ValueError('No memory hierarchy found. Consider changing constraints in the architecture file.')
    mem_scheme_sim_copy = deepcopy(mem_scheme_sim)
    for idx, each_mem_scheme in enumerate(mem_scheme_sim_copy):
        mem_scheme_fit = mem_scheme_fit_check(idx+1, each_mem_scheme, input_settings.precision, layer_spec.layer_info,
                                              input_settings.layer_number)
        if not mem_scheme_fit:
            del mem_scheme_sim[idx]
    if not mem_scheme_sim:
        raise ValueError('The largest memory in the hierarchy is still too small for holding the required workload.')

    # Manages the variables passed to the multiple parallel processes
    multi_manager = MultiManager(input_settings, mem_scheme_sim, layer_spec, layers, layer_info_im2col, layers_im2col,
                                 pw_im2col_flag)

    # A list containing the chunks that will be processed sequentially
    # Each element within a chunk will be processed in parallel
    # inter-chunk = serial
    # intra-chunk = parallel
    mem_scheme_sim_chunk_list = [mem_scheme_sim[i:i + input_settings.mem_scheme_parallel_processing] for i in
                                 range(0, len(mem_scheme_sim), input_settings.mem_scheme_parallel_processing)]

    for ii_mem_scheme_chunk, mem_scheme_sim_chunk in enumerate(mem_scheme_sim_chunk_list): # serial processing of chunks

        procs = []
        for mem_scheme_index, mem_scheme in enumerate(mem_scheme_sim_chunk):  # parallel processing of one chunk
            current_mem_scheme_index = mem_scheme_index + input_settings.mem_scheme_parallel_processing * ii_mem_scheme_chunk
            procs.append(Process(target=evaluate.mem_scheme_list_evaluate,
                                 args=(input_settings, mem_scheme, current_mem_scheme_index, layers, multi_manager)))
        for p in procs: p.start()
        for p in procs: p.join()

    ''' Collect the optimum spatial unrolling results for all memory schemes if doing architecture exploration'''
    if not input_settings.mem_hierarchy_single_simulation:
        evaluate.optimal_su_evaluate(input_settings, layers, multi_manager)

    of.print_helper(input_settings, layers, layers_saved, multi_manager)

    total_time = int(time.time() - t1)
    print('ZigZag finished running. Total elapsed time: %d seconds.' % total_time)
    print('Results are saved to %s.' % results_path)
    print()
