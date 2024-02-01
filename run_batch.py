import os, sys
import subprocess
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import ast
import copy
import math
import yaml
from pathlib import Path
from collections import Counter
from itertools import combinations
from demux8 import area_demux8_dict
from hda.analysis import test_sort, schedule_hda

HW_dict = {'Edge':['32_32', '33554432', '512', '27.0113', '0.105279', '0.0929084'], \
           'Mobile':['64_64', '67108864', '2048', '51.9235', '0.163375', '0.137476'], \
           'Cloud': ['128_128', '134217728', '8192', '103.323', '0.221043', '0.194981']}
# PE dimension, Global memory (GM) size, GM bandwidth, GM area, GM read energy, GM write energy

# BW: 250MHz / 8bit integer 기준

Benchmark_dict = {'ARVRA': [['ResNet50', 'UNet', 'MobileNetv2'], [2, 4, 4]], \
                  'ARVRB':[['ResNet50', 'UNet', 'MobileNetv2', 'BRQ', 'DepthNet'], [2, 2, 4, 2, 2]], \
                  'MLPerf': [['ResNet50', 'MobileNetv1', 'SSDResNet34', 'SSDMobileNetV1', 'GNMT'], [1, 1, 1, 1, 1]]}
# Included models, Batch sizes

Size_dict = {'ResNet50':53, 'UNet':23, 'MobileNetv2':53, 'BRQ':22, 'DepthNet':28, 'MobileNetv1':28, 'SSDResNet34':51, 'SSDMobileNetV1':47, 'GNMT':9}
# number of layers per model for validation

Split_dict = {  'ARVRA': {'Edge':[[4, 12], [128, 896]], \
                          'Mobile':[[40, 24], [1792, 2304]], \
                          'Cloud': [[224, 32], [9728, 6656]]}, \
                'ARVRB': {'Edge':[[4, 12], [128, 896]], \
                          'Mobile':[[48, 16], [1536, 2560]], \
                          'Cloud': [[128, 128], [12032, 4352]]}, \
                'MLPerf': {'Edge':[[4, 12], [64, 960]], \
                          'Mobile':[[32, 32], [1280, 2816]], \
                          'Cloud': [[160, 96], [8192, 8192]]} \
}
# splits in HDA - bandwidth split, PE split

# area info
rda_area_dict = {'Edge': 2900156.984578, \
           'Mobile': 12243471.5, \
           'Cloud': 51687751.7901}
area_spad = 0.000693826
area_multiplier8 = 266.798001
area_adder8 = 32.452
# area_demux8 is referenced from 'demux8.py'

# energy info

##########################################User Arguments##########################################

# select target HW
input_HW = 'Edge'
# select target Benchmark
input_Benchmark = 'MLPerf'
#network mode (optimal / greedy / hint)
network_mode = 'greedy'

# number of concerning mappings per layer
NM = 10
# number of max combination size
NC = 4

# HDA parameter
LA = 5 # look ahead parameter
maxLbF = 110 # parameter for choosing unoptimal sub-accelerator

# mode change overhead
ovh = 32

###################################################################################################

target_hw = HW_dict[input_HW]
target_bench = Benchmark_dict[input_Benchmark]

def compute_area(col, row, gb_size, mappings):

    base_area = col * row * (area_multiplier8 + area_adder8 + area_spad * 3) + gb_size

    demux_list_col = []
    demux_list_row = []
    final_adder = 0

    for map in mappings:
        temp_demux_list_col = copy.copy(demux_list_col)
        temp_demux_list_row = copy.copy(demux_list_row)

        map = ast.literal_eval(map.replace("K","'K'").replace("C","'C'").replace("OX","'OX'").replace("OY","'OY'").replace("FX","'FX'").replace("FY","'FY'"))
        mcol = map[0]
        mrow = map[1]

        # input demux reuse
        demux_col_K = 1
        demux_row_K = 1
        for m in mcol:
            if m[0] == 'K':
                demux_col_K = int(m[1]) # K가 다른 위치끼리 뭉쳐서 같은 값 공유
        for m in mrow:
            if m[0] == 'K':
                demux_row_K = int(m[1])

        input_demux_col = [demux_col_K, int(col / demux_col_K), 1] # block size of single output of demux, outputs of demux, number of demux
        if input_demux_col in temp_demux_list_col:
            temp_demux_list_col.remove(input_demux_col)
        else:
            demux_list_col.append(input_demux_col)
        input_demux_row = [demux_row_K, int(row / demux_row_K), col]
        if input_demux_row in temp_demux_list_row:
            temp_demux_list_row.remove(input_demux_row)
        else:
            demux_list_row.append(input_demux_row)

        # weight demux reuse
        demux_col_OY = 1
        demux_col_OX = 1
        demux_row_OY = 1
        demux_row_OX = 1
        for m in mcol:
            if m[0] == 'OY':
                demux_col_OY = int(m[1])
            if m[0] == 'OX':
                demux_col_OX = int(m[1])
        for m in mrow:
            if m[0] == 'OY':
                demux_row_OY = int(m[1])
            if m[0] == 'OX':
                demux_row_OX = int(m[1])

        weight_demux_col = [demux_col_OY * demux_col_OX, int(col / demux_col_OY / demux_col_OX), 1]  # block size of single output of demux, outputs of demux, number of demux
        if weight_demux_col in temp_demux_list_col:
            temp_demux_list_col.remove(weight_demux_col)
        else:
            demux_list_col.append(weight_demux_col)
        weight_demux_row = [demux_row_OY * demux_row_OX, int(row / demux_row_OY / demux_row_OX), col]
        if weight_demux_row in temp_demux_list_row:
            temp_demux_list_row.remove(weight_demux_row)
        else:
            demux_list_row.append(weight_demux_row)


        # output adder reuse
        add_col_C = 1
        add_col_FY = 1
        add_col_FX = 1
        add_row_C = 1
        add_row_FY = 1
        add_row_FX = 1
        for m in mcol:
            if m[0] == 'C':
                add_col_C = int(m[1])
            if m[0] == 'FY':
                add_col_FY = int(m[1])
            if m[0] == 'FX':
                add_col_FX = int(m[1])
        for m in mrow:
            if m[0] == 'C':
                add_row_C = int(m[1])
            if m[0] == 'FY':
                add_row_FY = int(m[1])
            if m[0] == 'FX':
                add_row_FX = int(m[1])

        max_adder = (add_col_C*add_col_FY*add_col_FX * add_row_C*add_row_FY*add_row_FX - 1) * ( int(col / (add_col_C*add_col_FY*add_col_FX)) *int(row / (add_row_C*add_row_FY*add_row_FX)))
        if max_adder > final_adder:
            final_adder = max_adder

    demux_area = 0
    for demux in (demux_list_col+demux_list_row):
        demux_area += area_demux8_dict[demux[1]] * demux[2]

    adder_area = final_adder * area_adder8

    all_area = base_area + demux_area + adder_area

    return all_area

def set_HW(target_hw):
    pe_shape = target_hw[0]
    col = pe_shape.split('_')[0]
    row = pe_shape.split('_')[1]
    all = str(int(col) * int(row))
    gmem_size = target_hw[1]
    gmem_bw = target_hw[2]
    gmem_area = target_hw[3]
    gmem_en_r = target_hw[4]
    gmem_en_w = target_hw[5]

    with open("./inputs/architecture.yaml", "r") as file:
        lines = file.read().splitlines()

    lines[1] = "  Col : " + col
    lines[2] = "  Row : " + row

    lines[48] = "      memory_unroll : " + all
    lines[52] = "      memory_unroll : " + all
    lines[56] = "      memory_unroll : " + all

    with open("./inputs/architecture.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

    with open("./inputs/memory_pool.yaml", "r") as file:
        lines = file.read().splitlines()

    lines[10] = "  size_bit: " + gmem_size
    lines[11] = "  mem_bw: [" + gmem_bw + "]"
    lines[12] = "  area: [" + gmem_area + "]"
    lines[14] = "    read_word: [" + gmem_en_r + "]"
    lines[15] = "    write_word: [" + gmem_en_w + "]"

    with open("./inputs/memory_pool.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

def set_network(network, mode, file_name):
    with open("./inputs/settings.yaml", "r") as file:
        lines = file.read().splitlines()

    lines[2] = "result_path : './_results/" + file_name + "'"
    lines[3] = "result_filename : " + file_name
    lines[27] = "layer_filename : './NN_layers/" + network + "'"
    if mode == 'optimal':
        lines[50] = "spatial_unrolling_search_method : heuristic_v2"
    elif mode == 'greedy':
        lines[50] = "spatial_unrolling_search_method : greedy_mapping_with_hint"
    else:
        lines[50] = "spatial_unrolling_search_method : hint_driven"

    with open("./inputs/settings.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

'''def set_mapping(cur_map):
    with open("./inputs/mapping.yaml", "r") as file:
        lines = file.read().splitlines()

    del lines[42:]

    map = ast.literal_eval(cur_map)
    str_list = ""
    for i, v in enumerate(map):
        str_list2 = []
        for w in v:
            str_list2.append(w[0])
        if i == 0:
            str_list = str_list + '[\n  [Col : ' + str(str_list2).replace("'","") + ', '
        else:
            str_list = str_list + 'Row : ' + str(str_list2).replace("'", "") + '],\n]'
    str_map = "spatial_mapping_list: " + str_list

    lines.append(str_map)

    with open("./inputs/mapping.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')'''

def set_mapping(cur_map):
    with open("./inputs/mapping.yaml", "r") as file:
        lines = file.read().splitlines()

    del lines[42:]

    map = ast.literal_eval(cur_map)
    str_list = ""
    for i, v in enumerate(map):
        str_list2 = []
        for w in v:
            str_list2.append(w[0]+'_'+str(w[1]))
        if i == 0:
            str_list = str_list + '[\n  [Col : ' + str(str_list2).replace("'","") + ', '
        else:
            str_list = str_list + 'Row : ' + str(str_list2).replace("'", "") + '],\n]'
    str_map = "spatial_mapping_list: " + str_list

    lines.append(str_map)

    with open("./inputs/mapping.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

def set_mapping2(cur_map, cur_map2):
    with open("./inputs/mapping.yaml", "r") as file:
        lines = file.read().splitlines()

    del lines[42:]

    map = ast.literal_eval(cur_map)
    map2 = ast.literal_eval(cur_map2)
    str_list = ""
    for i, v in enumerate(map):
        str_list2 = []
        for w in v:
            str_list2.append(w[0]+'_'+str(w[1]))
        if i == 0:
            str_list = str_list + '[\n[Col : ' + str(str_list2).replace("'","") + ', '
        else:
            str_list = str_list + 'Row : ' + str(str_list2).replace("'", "") + '],\n'
    for i, v in enumerate(map2):
        str_list2 = []
        for w in v:
            str_list2.append(w[0]+'_'+str(w[1]))
        if i == 0:
            str_list = str_list + '[Col : ' + str(str_list2).replace("'","") + ', '
        else:
            str_list = str_list + 'Row : ' + str(str_list2).replace("'", "") + '],\n]'
    str_map = "spatial_mapping_list: " + str_list

    lines.append(str_map)

    with open("./inputs/mapping.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

def set_mapping_hint(cur_map):
    with open("./inputs/mapping.yaml", "r") as file:
        lines = file.read().splitlines()

    del lines[42:]

    map = ast.literal_eval(cur_map)
    str_list = ""
    for i, v in enumerate(map):
        str_list2 = []
        for w in v:
            str_list2.append(w)
        if i == 0:
            str_list = str_list + '[\n  [Col : ' + str(str_list2).replace("'","") + ', '
        else:
            str_list = str_list + 'Row : ' + str(str_list2).replace("'", "") + '],\n]'
    str_map = "spatial_mapping_list: " + str_list

    lines.append(str_map)

    with open("./inputs/mapping.yaml", "w") as file:
        for line in lines:
            file.write(line + '\n')

def get_su(su):
    col_list_temp = []
    row_list_temp = []

    for level in su:
        if len(level):
            col_list_temp = col_list_temp + level[0]
            row_list_temp = row_list_temp + level[1]
    col_list = []
    row_list = []
    col_dict = {}
    row_dict = {}
    col_count = 0
    row_count = 0
    for i in col_list_temp:
        if i[0] in col_dict:
            col_list[col_dict[i[0]]][1] *= i[1]
        else:
            col_list.append(i)
            col_dict[i[0]] = col_count
            col_count = col_count + 1

    for i in row_list_temp:
        if i[0] in row_dict:
            row_list[row_dict[i[0]]][1] *= i[1]
        else:
            row_list.append(i)
            row_dict[i[0]] = row_count
            row_count = row_count + 1

    col_list.sort(key = lambda x: x[0])
    row_list.sort(key = lambda x: x[0])

    return col_list, row_list

def best_shape(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)

    return str(val), str(val2)

if __name__ == "__main__":

    os.makedirs('./_results', exist_ok=True) # path of results
    #sys.argv = [sys.argv[0].replace('run_batch.py', 'top_module.py'), '--arch', './inputs/architecture.yaml', '--map', './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool', './inputs/memory_pool.yaml']

    # Stage 1: Execute ZigZag

    print("Stage 1 Start!")
    set_HW(target_hw)
    network_list = target_bench[0]
    batch_list = target_bench[1]
    network_batch_dict = {}
    for i, v in enumerate(network_list):
        network_batch_dict[v] = int(batch_list[i])

    for cur_net in network_list:
        file_name = input_HW + '_' + cur_net
        if not os.path.exists('./_results/' + file_name):
            set_network(cur_net, 'optimal', file_name)
            # execute
            try:
                print("\nExecuting: ", cur_net)
                with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                    with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                        proc = subprocess.check_call([sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map', './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool', './inputs/memory_pool.yaml'], \
                                                stdout=f, stderr=g)
            except:
                print("Something wrong with:", input_HW, input_Benchmark, cur_net)
    print("Stage 1 Done!")

    # Stage 2: Generate candidate mappings

    print("Stage 2 Start!")
    os.makedirs('./analysis', exist_ok=True)  # path of results
    info_path = './analysis/' + input_HW + '_' + input_Benchmark + '_info'
    if not os.path.exists(info_path):
        info_dict = {}
        for cur_net in network_list:
            info_dict[cur_net] = {}
        for cur_net in network_list:
            file_name = input_HW + '_' + cur_net
            work_path = './_results/' + file_name + '/all_su_best_tm'
            weight = network_batch_dict[cur_net]

            for f in os.listdir(work_path):
                if f[-3:] == 'xml':
                    txt_list = f.split('_')
                    cur_layer = txt_list[2][1:]
                    cur_su = txt_list[4][2:]
                    if cur_layer not in info_dict[cur_net]:
                        info_dict[cur_net][cur_layer] = {}
                    if cur_su not in info_dict[cur_net][cur_layer]:
                        info_dict[cur_net][cur_layer][cur_su] = {}
                    cur_cost = txt_list[5]

                    tree = ET.parse(Path(work_path + '/' + f).absolute())
                    root = tree.getroot()
                    su = tree.find(".//spatial_unrolling")
                    col, row = get_su(ast.literal_eval(su[0].tail))
                    total_MAC = int(tree.find(".//total_MAC_operation").tail)
                    if cur_cost == 'max': # max_ut
                        utilization = float(tree.find(".//utilization_with_data_loading").tail)
                        latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                        energy = float(tree.find(".//total_energy").tail)
                        score = weight * total_MAC * utilization
                        info_dict[cur_net][cur_layer][cur_su]['ut'] = [[col, row], score, latency, energy]
                    else:
                        energy = float(tree.find(".//total_energy").tail)
                        score = total_MAC / energy # better if this score is higher
                        info_dict[cur_net][cur_layer][cur_su]['en'] = [[col, row], score, energy]

        with open(info_path, 'wb') as f:
            pickle.dump(info_dict, f, protocol=3)

    else:
        with open(info_path, 'rb') as f:
            info_dict = pickle.load(f)

    # analysis start
    ut_total_dict = {}
    en_total_dict = {}

    for val1 in info_dict.values(): # network level
        for val2 in val1.values(): # layer level
            for val3 in val2.values():  # su level
                # ut
                if 'ut' in val3:
                    su = str(val3['ut'][0])
                    if su not in ut_total_dict:
                        ut_total_dict[su] = 0
                    ut_total_dict[su] += val3['ut'][1]

                if 'en' in val3:
                    su = str(val3['en'][0])
                    if su not in en_total_dict:
                        en_total_dict[su] = []
                    en_total_dict[su].append(val3['en'][1])

    for key, val in en_total_dict.items():  # network level
        en_total_dict[key] = np.average(val)

    ut_maximum = Counter(ut_total_dict).most_common(NM)
    en_maximum = Counter(en_total_dict).most_common(NM)

    print("Stage 2 Done!")

    # Stage 3: Evaluate candidate mappings

    print("Stage 3 Start!")

    # For UT
    for ind, cur_map in enumerate(ut_maximum):
        for cur_net in network_list:
            file_name = input_HW + '_' + cur_map[0].replace("'","") + '_' + cur_net
            if not os.path.exists('./_results/' + file_name): # skip if it exists
                set_network(cur_net, 'hint', file_name)
                set_mapping(cur_map[0])
                # execute
                try:
                    print("\nExecuting: ", str(ind + 1) + ' / ' + str(NM) + ' |', cur_net, cur_map)
                    with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                        with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                            proc = subprocess.check_call([sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map', './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool', './inputs/memory_pool.yaml'], \
                                                    stdout=f, stderr=g)
                except:
                    print("Something wrong with:", input_HW, input_Benchmark, ind, cur_net)

    # For EN
    for ind, cur_map in enumerate(en_maximum):
        for cur_net in network_list:
            file_name = input_HW + '_' + cur_map[0].replace("'", "") + '_' + cur_net
            if not os.path.exists('./_results/' + file_name): # skip if it exists
                set_network(cur_net, 'hint', file_name)
                set_mapping(cur_map[0])
                # execute
                try:
                    print("\nExecuting: ", str(ind + 1) + ' / ' + str(NM) + ' |', cur_net, cur_map)
                    with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                        with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                            proc = subprocess.check_call(
                                [sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map',
                                 './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool',
                                 './inputs/memory_pool.yaml'], \
                                stdout=f, stderr=g)
                except:
                    print("Something wrong with:", input_HW, input_Benchmark, ind, cur_net)

    print("Stage 3 Done!")

    # Stage 4: Run Zigzag for optimal mappings

    print("Stage 4 Start!")
    set_all_maximum = set()
    for i in ut_maximum:
        set_all_maximum.add(i[0])
    for i in en_maximum:
        set_all_maximum.add(i[0])
    list_all_maximum = list(set_all_maximum)

    info_path = './analysis/' + input_HW + '_' + input_Benchmark + '_info_all'
    if not os.path.exists(info_path):
        info_dict_all = {}
        for cur_net in network_list:
            info_dict_all[cur_net] = {}

        for cur_map in list_all_maximum:
            for cur_net in network_list:
                file_name = input_HW + '_' + cur_map.replace("'","") + '_' + cur_net
                work_path = './_results/' + file_name + '/best_su_best_tm'

                for f in os.listdir(work_path):
                    if f[-3:] == 'xml':
                        txt_list = f.split('_')
                        cur_layer = txt_list[3][1:]
                        cur_su = txt_list[1]
                        if cur_su not in info_dict_all[cur_net]:
                            info_dict_all[cur_net][cur_su] = {}
                        if cur_layer not in info_dict_all[cur_net][cur_su]:
                            info_dict_all[cur_net][cur_su][cur_layer] = {}
                        cur_cost = txt_list[6]

                        tree = ET.parse(Path(work_path + '/' + f).absolute())
                        root = tree.getroot()
                        if cur_cost == 'max':  # max_ut
                            latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                            energy = float(tree.find(".//total_energy").tail)
                            info_dict_all[cur_net][cur_su][cur_layer]['ut'] = (latency, energy)
                        else:
                            energy = float(tree.find(".//total_energy").tail)
                            info_dict_all[cur_net][cur_su][cur_layer]['en'] = energy

        with open(info_path, 'wb') as f:
            pickle.dump(info_dict_all, f, protocol=3)

    else:
        with open(info_path, 'rb') as f:
            info_dict_all = pickle.load(f)

    print("Stage 4 Done!")

    # Stage 5: Find optimal mappings (Ours)

    print("Stage 5 Start!")

    ut_total_dict = {}
    en_total_dict = {}

    best_latency = 9999999999999999
    best_latency_map = None

    best_latency_ovh = 9999999999999999
    best_latency_ovh_map = None

    best_latency_ovh_opt = 9999999999999999
    best_latency_ovh_opt_map = None

    best_latency_ovh_opt_area = 9999999999999999
    best_latency_ovh_opt_area_map = None
    best_latency_ovh_opt_area_area = None
    best_latency_ovh_opt_area_area_energy = 9999999999999999
    best_latency_reconfiguration_before_count = 0
    best_latency_reconfiguration_opt_count = 0

    best_energy = 9999999999999999
    best_energy_map = None
    
    for max_map in range(2, NC+1):
        comb_list = list(combinations(ut_maximum, max_map))
        for i_comb in comb_list:
            cur_comb_list = []
            for i in i_comb:
                cur_comb_list.append(i[0].replace("'",""))

            imp = 0 # check whether current mapping can cover all models
            for key, val in info_dict_all.items(): # layer level
                layer_set = set()
                for i in cur_comb_list:
                    for key2 in val[i].keys():
                        layer_set.add(key2)

                if len(layer_set) != Size_dict[key]:
                    imp = 1
                    break
            if imp:
                continue

            temp_model_dict_ut = {}
            temp_model_dict_en = {}
            for key, val in info_dict_all.items():
                temp_model_dict_ut[key] = {}
                temp_model_dict_en[key] = {}
                for l in range(Size_dict[key]):
                    temp_model_dict_ut[key][l] = {}
                    temp_model_dict_en[key][l] = {}
                    for i, v in enumerate(cur_comb_list):
                        if v in info_dict_all[key]:
                            cur_l = str(l+1)
                            if cur_l in info_dict_all[key][v]:
                                if 'ut' in info_dict_all[key][v][cur_l]:
                                    temp_model_dict_ut[key][l][i] = info_dict_all[key][v][cur_l]['ut']
                                if 'en' in info_dict_all[key][v][cur_l]:
                                    temp_model_dict_en[key][l][i] = info_dict_all[key][v][cur_l]['en']

            final_latency = 0.0
            final_latency_ovh = 0.0
            final_latency_ovh_opt = 0.0
            final_latency_ovh_opt_area = 0.0
            final_energy_ovh_opt_area = 0.0
            reconfiguration_count = 0
            reconfiguration_opt_count = 0

            array_size = target_hw[0]
            col = int(array_size.split('_')[0])
            row = int(array_size.split('_')[1])
            our_area = compute_area(col, row, float(target_hw[3]), cur_comb_list)

            final_energy = []
            for key, val in temp_model_dict_ut.items():
                layer_latency = 0.0
                layer_latency_ovh = 0.0
                layer_latency_ovh_opt = 0.0
                layer_energy_ovh_opt = 0.0

                temp_key = None
                temp_opt_key = None

                for k2, layer in val.items():
                    min_key = min(layer, key=layer.get)
                    layer_latency += layer[min_key][0]

                    if min_key != temp_key:
                        layer_latency_ovh += layer[min_key][0] + ovh
                        reconfiguration_count += 1
                    else:
                        layer_latency_ovh += layer[min_key][0]
                    temp_key = min_key

                    if min_key != temp_opt_key:
                        if temp_opt_key and (temp_opt_key in layer) and (layer[temp_opt_key][0] < layer[min_key][0] + ovh):
                            layer_latency_ovh_opt += layer[temp_opt_key][0]
                            layer_energy_ovh_opt += layer[temp_opt_key][1]
                        else:
                            layer_latency_ovh_opt += layer[min_key][0] + ovh
                            layer_energy_ovh_opt += layer[min_key][1]
                            temp_opt_key = min_key
                            reconfiguration_opt_count += 1
                    else:
                        layer_latency_ovh_opt += layer[min_key][0]
                        layer_energy_ovh_opt += layer[min_key][1]
                        temp_opt_key = min_key

                weight = network_batch_dict[key]
                final_latency += weight * layer_latency
                final_latency_ovh += weight * layer_latency_ovh
                final_latency_ovh_opt += weight * layer_latency_ovh_opt
                final_latency_ovh_opt_area += weight * layer_latency_ovh_opt * our_area
                final_energy_ovh_opt_area += weight * layer_energy_ovh_opt


            for key, val in temp_model_dict_en.items():
                for k2, layer in val.items():
                    min_key = min(layer, key=layer.get)
                    final_energy.append(layer[min_key])

            final_energy = np.average(final_energy)

            if final_latency < best_latency:
                best_latency = final_latency
                best_latency_map = cur_comb_list

            if final_latency_ovh < best_latency_ovh:
                best_latency_ovh = final_latency_ovh
                best_latency_ovh_map = cur_comb_list

            if final_latency_ovh_opt < best_latency_ovh_opt:
                best_latency_ovh_opt = final_latency_ovh_opt
                best_latency_ovh_opt_map = cur_comb_list

            if final_latency_ovh_opt_area < best_latency_ovh_opt_area:
                best_latency_ovh_opt_area = final_latency_ovh_opt_area
                best_latency_ovh_opt_area_map = cur_comb_list
                best_latency_ovh_opt_area_area = our_area
                best_latency_ovh_opt_area_area_energy = final_energy_ovh_opt_area
                best_latency_reconfiguration_count = reconfiguration_count
                best_latency_reconfiguration_opt_count = reconfiguration_opt_count

            if final_energy < best_energy:
                best_energy = final_energy
                best_energy_map = cur_comb_list

    print("Our best latency:", best_latency, best_latency_map)
    print("Our best latency after applying overhead:", best_latency_ovh, best_latency_ovh_map)
    print("Our best latency after optimizing overhead:", best_latency_ovh_opt, best_latency_ovh_opt_map)
    print("Our best optimal latency / area / total energy / mapping:", best_latency_ovh_opt_area / best_latency_ovh_opt_area_area, '/', best_latency_ovh_opt_area_area, '/', best_latency_ovh_opt_area_area_energy, '/\n', best_latency_ovh_opt_area_map)
    print("reconfiguration count: ",best_latency_reconfiguration_count)
    print("reconfiguration opt count: ", best_latency_reconfiguration_opt_count)
    print("Our best energy:", best_energy, best_energy_map)

    print("Stage 5 Done!")

    # Stage 6: Find optimal mappings (RDA)

    print("Stage 6 Start!")

    rda_area = rda_area_dict[input_HW]

    rda_latency = 0.0
    rda_latency_ovh = 0.0
    rda_latency_energy = 0.0
    rda_reconfiguration_count = 0

    rda_energy = []
    for key, val in info_dict.items():
        rda_map_prev = None
        weight = network_batch_dict[key]
        model_rda_latency = 0.0
        model_rda_latency_ovh = 0.0
        model_rda_energy = 0.0

        temp_map = None
        for k2, layer in val.items():
            dict_latency = {}
            dict_latency_energy = {}
            dict_latency_ovh = {}
            dict_energy = {}
            for k3, cur_su in layer.items():
                if 'ut' in cur_su:
                    dict_latency[str(cur_su['ut'][0])] = cur_su['ut'][2]
                    dict_latency_energy[str(cur_su['ut'][0])] = cur_su['ut'][3]
                if 'en' in cur_su:
                    dict_energy[str(cur_su['en'][0])] = cur_su['en'][2]

            min_latency_key = min(dict_latency, key=dict_latency.get)
            min_latency = dict_latency[min_latency_key]
            if (temp_map) and min_latency_key != temp_map:
                min_latency_ovh = min_latency + ovh
                rda_reconfiguration_count +=1
            else:
                min_latency_ovh = min_latency
            temp_map = min_latency_key

            min_energy = dict_energy[min(dict_energy, key=dict_energy.get)]

            model_rda_latency += min_latency
            model_rda_latency_ovh += min_latency_ovh
            model_rda_energy += dict_latency_energy[min_latency_key]
            rda_energy.append(min_energy)
        rda_latency += weight * model_rda_latency
        rda_latency_ovh += weight * model_rda_latency_ovh
        rda_latency_energy += weight * model_rda_energy

    rda_energy = np.average(rda_energy)

    print("RDA best latency:", rda_latency)
    print("RDA best latency after applying overhead:", rda_latency_ovh)
    print("RDA best optimal latency / area / total energy:", rda_latency_ovh, '/', rda_area, '/', rda_latency_energy)
    print("RDA reconfiguration count: ", rda_reconfiguration_count)

    print("RDA best energy:", rda_energy)

    print("Stage 6 Done!")

    # Stage 7: Find optimal mappings (HDA)

    print("Stage 7 Start!")

    cur_split = Split_dict[input_Benchmark][input_HW]
    split_BW = cur_split[0]
    split_PE = cur_split[1]

    HW_1 = copy.deepcopy(target_hw)
    HW_2 = copy.deepcopy(target_hw)

    # NVDLA
    BW_ratio_1 = split_BW[0] / (split_BW[0] + split_BW[1])
    for i, v in enumerate(HW_1[1:]):
        if (i == 0) or (i == 1):
            HW_1[i + 1] = str(int(float(v) * BW_ratio_1))
        else:
            HW_1[i + 1] = str(float(v) * BW_ratio_1)

    hda_col_1, hda_row_1 = best_shape(split_PE[0])
    HW_1[0] = hda_col_1 + '_' + hda_row_1
    set_HW(HW_1)

    for cur_net in network_list:
        file_name = input_HW + '_hda_' + HW_1[0] + '_' + str(split_BW[0]) + '_' + str(
            split_BW[0] + split_BW[1]) + '_' + cur_net
        if not os.path.exists('./_results/' + file_name):  # skip if it exists
            set_network(cur_net, 'greedy', file_name)
            if hda_col_1 == hda_row_1:
                set_mapping("[[('C', " + hda_col_1 + ")], [('K', " + hda_row_1 + ")]]")
            else:
                set_mapping2("[[('C', " + hda_col_1 + ")], [('K', " + hda_row_1 + ")]]",
                             "[[('K', " + hda_col_1 + ")], [('C', " + hda_row_1 + ")]]")
                # execute
            try:
                print("\nExecuting NVDLA: ", cur_net)
                with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                    with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                        proc = subprocess.check_call(
                            [sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map',
                             './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool',
                             './inputs/memory_pool.yaml'], \
                            stdout=f, stderr=g)
            except:
                print("Something wrong with NVDLA:", input_HW, input_Benchmark, cur_net)

    # Shi
    BW_ratio_2 = split_BW[1] / (split_BW[0] + split_BW[1])
    for i, v in enumerate(HW_2[1:]):
        if (i == 0) or (i == 1):
            HW_2[i + 1] = str(int(float(v) * BW_ratio_2))
        else:
            HW_2[i + 1] = str(float(v) * BW_ratio_2)

    hda_col_2, hda_row_2 = best_shape(split_PE[1])
    HW_2[0] = hda_col_2 + '_' + hda_row_2
    set_HW(HW_2)
    for cur_net in network_list:
        file_name = input_HW + '_hda_' + HW_2[0] + '_' + str(split_BW[1]) + '_' + str(
            split_BW[0] + split_BW[1]) + '_' + cur_net
        if not os.path.exists('./_results/' + file_name):  # skip if it exists
            set_network(cur_net, 'greedy', file_name)
            if hda_col_2 == hda_row_2:
                set_mapping("[[('OX', " + hda_col_2 + ")], [('OY', " + hda_row_2 + ")]]")
            else:
                set_mapping2("[[('OX', " + hda_col_2 + ")], [('OY', " + hda_row_2 + ")]]",
                             "[[('OY', " + hda_col_2 + ")], [('OX', " + hda_row_2 + ")]]")
                # execute
            try:
                print("\nExecuting Shi: ", cur_net)
                with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                    with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                        proc = subprocess.check_call(
                            [sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map',
                             './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool',
                             './inputs/memory_pool.yaml'], \
                            stdout=f, stderr=g)
            except:
                print("Something wrong with Shi:", input_HW, input_Benchmark, cur_net)

    info_path = './analysis/' + input_HW + '_' + input_Benchmark + '_info_hda'
    if not os.path.exists(info_path):
        info_dict_hda = [{}, {}]
        for cur_net in network_list:

            file_name = input_HW + '_hda_' + HW_1[0] + '_' + str(split_BW[0]) + '_' + str(
                split_BW[0] + split_BW[1]) + '_' + cur_net
            work_path = './_results/' + file_name + '/all_su_best_tm'

            for f in os.listdir(work_path):
                if f[-3:] == 'xml':
                    txt_list = f.split('_')
                    cur_layer = txt_list[7][1:]
                    cur_su = txt_list[9]
                    if cur_su not in info_dict_hda[0]:
                        info_dict_hda[0][cur_su] = {}
                    if cur_net not in info_dict_hda[0][cur_su]:
                        info_dict_hda[0][cur_su][cur_net] = {}
                    if cur_layer not in info_dict_hda[0][cur_su][cur_net]:
                        info_dict_hda[0][cur_su][cur_net][cur_layer] = {}
                    cur_cost = txt_list[10]

                    tree = ET.parse(Path(work_path + '/' + f).absolute())
                    root = tree.getroot()
                    if cur_cost == 'max':  # max_ut
                        latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_hda[0][cur_su][cur_net][cur_layer]['ut'] = (latency, energy)
                    else:  # min_en
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_hda[0][cur_su][cur_net][cur_layer]['en'] = energy

            file_name = input_HW + '_hda_' + HW_2[0] + '_' + str(split_BW[1]) + '_' + str(
                split_BW[0] + split_BW[1]) + '_' + cur_net
            work_path = './_results/' + file_name + '/all_su_best_tm'

            for f in os.listdir(work_path):
                if f[-3:] == 'xml':
                    txt_list = f.split('_')
                    cur_layer = txt_list[7][1:]
                    cur_su = txt_list[9]
                    if cur_su not in info_dict_hda[1]:
                        info_dict_hda[1][cur_su] = {}
                    if cur_net not in info_dict_hda[1][cur_su]:
                        info_dict_hda[1][cur_su][cur_net] = {}
                    if cur_layer not in info_dict_hda[1][cur_su][cur_net]:
                        info_dict_hda[1][cur_su][cur_net][cur_layer] = {}
                    cur_cost = txt_list[10]

                    tree = ET.parse(Path(work_path + '/' + f).absolute())
                    root = tree.getroot()
                    if cur_cost == 'max':  # max_ut
                        latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_hda[1][cur_su][cur_net][cur_layer]['ut'] = (latency, energy)
                    else:  # min_en
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_hda[1][cur_su][cur_net][cur_layer]['en'] = energy

        with open(info_path, 'wb') as f:
            pickle.dump(info_dict_hda, f, protocol=3)

    else:
        cur_split = Split_dict[input_Benchmark][input_HW]
        split_BW = cur_split[0]
        split_PE = cur_split[1]
        hda_col_1, hda_row_1 = best_shape(split_PE[0])
        hda_col_2, hda_row_2 = best_shape(split_PE[1])
        BW_ratio_1 = split_BW[0] / (split_BW[0] + split_BW[1])
        BW_ratio_2 = split_BW[1] / (split_BW[0] + split_BW[1])

        with open(info_path, 'rb') as f:
            info_dict_hda = pickle.load(f)

    print("Stage 7 Done!")

    # Stage 8: Schedule HDA

    print("Stage 8 Start!")

    # load dependency
    dir_list = os.listdir("./dependency")
    dep_dict = {}
    for dep_file in dir_list:
        with open("./dependency/" + dep_file) as f:
            temp = yaml.load(f, Loader=yaml.FullLoader)
            dep_dict[dep_file] = temp
            test_sort(temp)
            print("Pleace check dependency")

    model_list = []
    for i, v in enumerate(target_bench[0]):
        model_list.append([v, target_bench[1][i]])

    num_map_1 = len(info_dict_hda[0])
    num_map_2 = len(info_dict_hda[1])

    hda_best_latency = 9999999999999999
    hda_best_m1 = None
    hda_best_m2 = None
    hda_best_latency_energy = 9999999999999999
    


    for m1 in range(1, num_map_1 + 1):
        for m2 in range(1, num_map_2 + 1):
            map_1 = info_dict_hda[0]['SU' + str(m1)]
            map_2 = info_dict_hda[1]['SU' + str(m2)]

            latency_1 = {}
            energy_1 = {}
            for key, val in map_1.items():
                cur_list = []
                en_list = []
                len_all = len(val)
                if len_all != Size_dict[key]:
                    print("Size not match")
                for j in range(1, len_all + 1):
                    cur_list.append(val[str(j)]['ut'][0])
                    en_list.append(val[str(j)]['ut'][1])
                latency_1[key] = cur_list
                energy_1[key] = en_list

            latency_2 = {}
            energy_2 = {}
            for key, val in map_2.items():
                cur_list = []
                en_list = []
                len_all = len(val)
                if len_all != Size_dict[key]:
                    print("Size not match")
                for j in range(1, len_all + 1):
                    cur_list.append(val[str(j)]['ut'][0])
                    en_list.append(val[str(j)]['ut'][1])
                latency_2[key] = cur_list
                energy_2[key] = en_list

            latency_dict_list = [latency_1, latency_2]
            energy_dict_list = [energy_1, energy_2]

            minLbF = -1.0
            minlatency = 999999999999
            minlatency_energy = 999999999999

            for bLbF in range(10, maxLbF, 1):
                LbF = bLbF * 0.01

                total_latency_hda, total_latency_hda_energy = schedule_hda(model_list, latency_dict_list, energy_dict_list, dep_dict, LbF, LA)

                if total_latency_hda < minlatency:
                    minlatency = total_latency_hda
                    minlatency_energy = total_latency_hda_energy
                    minLbF = LbF

            print('[', str(m1) + ',', str(m2), '] - LbF:', minLbF, ', Latency:', minlatency)

            if minlatency < hda_best_latency:
                hda_best_latency = minlatency
                hda_best_m1 = m1
                hda_best_m2 = m2
                hda_best_latency_energy = minlatency_energy

    if hda_best_m1 == 1:
        hda_map1 = "[[(K, " + hda_col_1 + ")], [(C, " + hda_row_1 + ")]]"
    else:
        hda_map1 = "[[(K, " + hda_col_1 + ")], [(C, " + hda_row_1 + ")]]"

    if hda_best_m2 == 1:
        hda_map2 = "[[(OX, " + hda_col_2 + ")], [(OY, " + hda_row_2 + ")]]"
    else:
        hda_map2 = "[[(OY, " + hda_col_2 + ")], [(OX, " + hda_row_2 + ")]]"

    hda_area = compute_area(int(hda_col_1), int(hda_row_1), float(target_hw[3]) * BW_ratio_1, [hda_map1])\
               + compute_area(int(hda_col_2), int(hda_row_2), float(target_hw[3]) * BW_ratio_2, [hda_map2])

    print("HDA best optimal latency / area / total energy:", hda_best_latency, '/', hda_area, '/', hda_best_latency_energy)


    #stage 9: FDA baselines

    print("Stage 9 Start!")

    # NVDLA

    set_HW(target_hw)
    array_size = target_hw[0]
    num_col = int(array_size.split('_')[0]) 
    num_row = int(array_size.split('_')[1])

    for cur_net in network_list:
        file_name = input_HW + '_nvdla_' + cur_net  
        if not os.path.exists('./_results/' + file_name):  # skip if it exists
            set_network(cur_net, network_mode, file_name)
            set_mapping_hint("[[('C')], [('K')]]")
            # execute
            try:
                print("\nExecuting NVDLA: ", cur_net)
                with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                    with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                        proc = subprocess.check_call(
                            [sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map',
                             './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool',
                             './inputs/memory_pool.yaml'], \
                            stdout=f, stderr=g)
            except:
                print("Something wrong with NVDLA:", input_HW, input_Benchmark, cur_net)

    info_path = './analysis/' + input_HW + '_' + input_Benchmark + '_info_nvdla' 
    if not os.path.exists(info_path):
        info_dict_nvdla= {}
        for cur_net in network_list:
            file_name = input_HW + '_nvdla_' + cur_net 
            work_path = './_results/' + file_name + '/all_su_best_tm'

            for f in os.listdir(work_path):
                if f[-3:] == 'xml':
                    txt_list = f.split('_')
                    cur_layer = txt_list[4][1:]
                    cur_su = txt_list[6]
                    if cur_net not in info_dict_nvdla:
                        info_dict_nvdla[cur_net] = {}
                    if cur_su not in info_dict_nvdla[cur_net]:
                        info_dict_nvdla[cur_net][cur_su] = {}
                    if cur_layer not in info_dict_nvdla[cur_net][cur_su]:
                        info_dict_nvdla[cur_net][cur_su][cur_layer] = {}
                    cur_cost = txt_list[7]

                    tree = ET.parse(Path(work_path + '/' + f).absolute())
                    root = tree.getroot()
                    if cur_cost == 'max':  # max_ut
                        latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_nvdla[cur_net][cur_su][cur_layer]['ut'] = (latency, energy)
                    else:  # min_en
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_nvdla[cur_net][cur_su][cur_layer]['en'] = energy
        with open(info_path, 'wb') as f:
            pickle.dump(info_dict_nvdla, f, protocol=3)

    else:
        with open(info_path, 'rb') as f:
            info_dict_nvdla = pickle.load(f)

    nvdla_latency = 0.0
    nvdla_energy = 0.0

    for key1, val1 in info_dict_nvdla.items():
        temp_latency = 0.0
        temp_energy = 0.0
        for key2, val2 in val1.items():
            for key3, val3 in val2.items():
                temp_latency += val3['ut'][0]
                temp_energy += val3['ut'][1]


        weight = network_batch_dict[key1]
        nvdla_latency += weight * temp_latency
        nvdla_energy += weight * temp_energy
    


    print("\nNVDLA latency:", nvdla_latency)
    print("NVDLA energy:", nvdla_energy)
    

    nvdla_map = "[[(K, " +str(num_col) + ")], [(C, " + str(num_row) + ")]]"
    fda_area = compute_area(num_col, num_row, float(target_hw[3]), [nvdla_map])
    print("\nFDA area: ", fda_area)
    
    '''
    # Shi
    for cur_net in network_list:
        file_name = input_HW + '_shi_' + cur_net + '_' + network_mode
        if not os.path.exists('./_results/' + file_name):  # skip if it exists
            set_network(cur_net, network_mode, file_name)
            set_mapping_hint("[[('OX')], [('OY')]]")
            # execute
            try:
                print("\nExecuting Shi: ", cur_net)
                with open('./_results/' + "LOG_" + file_name + ".txt", "w") as f:
                    with open('./_results/' + "ERR_" + file_name + ".txt", "w") as g:
                        proc = subprocess.check_call(
                            [sys.executable, 'top_module.py', '--arch', './inputs/architecture.yaml', '--map',
                             './inputs/mapping.yaml', '--set', './inputs/settings.yaml', '--mempool',
                             './inputs/memory_pool.yaml'], \
                            stdout=f, stderr=g)
            except:
                print("Something wrong with Shi:", input_HW, input_Benchmark, cur_net)

    info_path = './analysis/' + input_HW + '_' + input_Benchmark + '_info_shi_' + network_mode
    if not os.path.exists(info_path):
        info_dict_shi= {}
        for cur_net in network_list:
            file_name = input_HW + '_shi_' + cur_net + '_' + network_mode
            work_path = './_results/' + file_name + '/all_su_best_tm'

            for f in os.listdir(work_path):
                if f[-3:] == 'xml':
                    txt_list = f.split('_')
                    cur_layer = txt_list[4][1:]
                    cur_su = txt_list[6]
                    if cur_net not in info_dict_shi:
                        info_dict_shi[cur_net] = {}
                    if cur_su not in info_dict_shi[cur_net]:
                        info_dict_shi[cur_net][cur_su] = {}
                    if cur_layer not in info_dict_shi[cur_net][cur_su]:
                        info_dict_shi[cur_net][cur_su][cur_layer] = {}
                    cur_cost = txt_list[7]

                    tree = ET.parse(Path(work_path + '/' + f).absolute())
                    root = tree.getroot()
                    if cur_cost == 'max':  # max_ut
                        latency = float(tree.find(".//latency_cycle_with_data_loading").tail)
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_shi[cur_net][cur_su][cur_layer]['ut'] = (latency, energy)
                    else:  # min_en
                        energy = float(tree.find(".//total_energy").tail)
                        info_dict_shi[cur_net][cur_su][cur_layer]['en'] = energy
        with open(info_path, 'wb') as f:
            pickle.dump(info_dict_shi, f, protocol=3)

    else:
        with open(info_path, 'rb') as f:
            info_dict_shi = pickle.load(f)

    shi_latency = 0.0
    shi_energy = 0.0

    for key1, val1 in info_dict_shi.items():
        temp_latency = 0.0
        temp_energy = 0.0
        for key2, val2 in val1.items():
            for key3, val3 in val2.items():
                temp_latency += val3['ut'][0]
                temp_energy += val3['ut'][1]


        weight = network_batch_dict[key1]
        shi_latency += weight * temp_latency
        shi_energy += weight * temp_energy
    

    print("\nSHI latency:", shi_latency)
    print("SHI energy:", shi_energy)
    '''


    print("Stage 9 Done!")
