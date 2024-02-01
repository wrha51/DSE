import os
import yaml

import numpy as np

from pathlib import Path
from collections import defaultdict

from hda.topsort import Graph
from hda.get_result import result_path, dict_numtobenchmark, dict_numtoclass, get_model_config

dependency_path = Path('./dependency')

def test_sort(temp):
    graph = Graph(len(temp))
    for key, val in temp.items():
        if val:
            for v in val:
                    graph.addEdge(key, v)
    graph.topologicalSort()

def get_latency(csv_list):
    return_list = []
    for i in range(1, len(csv_list)):
        splitted = csv_list[i].split(',')
        return_list.append(splitted[3])

    return np.array(return_list, dtype=float)

def check_not_empty(model_list):
    flag = False
    for model in model_list:
        if model[2] == model[3]:
            flag |= False
        else:
            flag |= True

    return flag

class Sched:
    def __init__(self, len_list):
        self.local_counter = [-1, -1] # negative if empty
        self.cur = [[-1, -1], [-1, -1]] # ID, layer
        self.totlatency = [0.0, 0.0]
        self.finished = []
        for model in range(len_list):
            self.finished.append([])
        self.schedule_history = [[], []] # accel 0, accel 1
        self.check_flag = False

    def check_dep(self, mid, mlayer):
        return mlayer in self.finished[mid]

    def assign(self, cycle, best_acc, latency, model_id, cur_layer):
        self.check_flag = False
        self.totlatency[best_acc] += latency
        self.cur[best_acc] = [model_id, cur_layer]
        self.local_counter[best_acc] = latency
        self.schedule_history[best_acc].append([cycle, latency, model_id, cur_layer])

    def nextLayer(self, cycle):
        if (self.local_counter[0] < 0) or (self.local_counter[1] < 0):
            if not self.check_flag:
                self.check_flag = True # check empty accelerator once
                return cycle

        self.check_flag = False
        if self.local_counter[0] < 0:
            select = 1
        elif self.local_counter[1] < 0:
            select = 0
        else:
            select = np.argmin(self.local_counter)

        spend_cycle = self.local_counter[select]
        self.local_counter[select] = -1
        cur_cur = self.cur[select]
        self.finished[cur_cur[0]].append(cur_cur[1])

        nselect = not select
        self.local_counter[nselect] -= spend_cycle
        if self.local_counter[nselect] == 0: # if simultaneously finished
            self.local_counter[nselect] = -1
            cur_cur = self.cur[nselect]
            self.finished[cur_cur[0]].append(cur_cur[1])

        return cycle + spend_cycle

def reorder_BFS(cur_model_list):
    temp_list1 = []
    temp_list2 = []
    cur_model = cur_model_list[0][1]
    for model in cur_model_list:
        if model[1] == cur_model:
            temp_list2.append(model)
        else:
            temp_list1.append(model)

    return temp_list1 + temp_list2

def schedule_hda(model_list, latency_dict_list, energy_dict_list, dep_dict, LbF, LA):
    cur_model_list = []
    id_to_model = {}
    model_id = 0
    for model in model_list:
        len_model = len(dep_dict[model[0]])
        for i in range(model[1]):
            cur_model_list.append([model_id, model[0], len_model, 0]) # ID, model name, model length, current head
            id_to_model[model_id] = model[0]
            model_id += 1

    cycle = 0.0
    sched = Sched(len(cur_model_list))
    while check_not_empty(cur_model_list):
        for model in cur_model_list:
            model_id = model[0]
            model_name = model[1]
            model_len = model[2]
            cur_layer = model[3] # 현재 처리하는 레이어
            if cur_layer == model_len:
                continue

            latency = []
            latency.append(latency_dict_list[0][model_name][cur_layer])
            latency.append(latency_dict_list[1][model_name][cur_layer])
            best_acc = np.argmin(latency)

            # dependency check
            dep_flag = True
            dep_list = dep_dict[model_name][cur_layer]
            if dep_list:
                for dep in dep_list:
                    dep_cond = sched.check_dep(model_id, dep)
                    if not dep_cond:
                        dep_flag = False
                        break

            # Memory check - skipped

            # Load balance check
            bal_flag = sched.totlatency[not best_acc] < LbF * (sched.totlatency[best_acc] + latency[best_acc])

            if dep_flag:
                if bal_flag:
                    best_acc = not best_acc
                if sched.local_counter[best_acc] > 0: # not empty
                    continue
                sched.assign(cycle, best_acc, latency[best_acc], model_id, cur_layer)

                model[3] += 1

                # default: DFS
                cur_model_list = reorder_BFS(cur_model_list)
                break
        cycle = sched.nextLayer(cycle)

    if LA > 0:
        history_list = sched.schedule_history
        for acc in range(2):
            for ind in range(len(history_list[acc])):
                look_ahead = 1
                while look_ahead < LA:
                    item = history_list[acc][ind]
                    item_start = item[0]
                    item_latency = item[1]

                    if (ind + look_ahead) >= len(history_list[acc]):
                        break
                    test_item = history_list[acc][ind + look_ahead]
                    test_start = test_item[0]
                    test_latency = test_item[1]
                    test_id = test_item[2]
                    test_layer = test_item[3]

                    com_time = item_start + item_latency
                    if com_time == test_start: # if look_ahead == 1, and there's no gap
                        look_ahead += 1
                        continue

                    # back schedule check
                    if look_ahead != 1:
                        if (com_time + test_latency) > history_list[acc][ind + 1][0]:
                            look_ahead += 1
                            continue

                    # dependency check
                    dep_target_list = []
                    for acc2 in range(2):
                        for checker in history_list[acc]:
                            if (checker[2] == test_id) and (checker[0] + checker[1] <= com_time):
                                dep_target_list.append(checker[3])

                    dep_flag = True
                    model_name = id_to_model[test_id]
                    dep_list = dep_dict[model_name][test_layer]
                    if dep_list:
                        for dep in dep_list:
                            if dep not in dep_target_list:
                                dep_flag = False
                                break
                    if not dep_flag:
                        look_ahead += 1
                        continue

                    # change schedule
                    temp_item = history_list[acc].pop(ind + look_ahead)
                    temp_item[0] = com_time
                    history_list[acc].insert(ind + 1, temp_item)

                    look_ahead += 1

        final_cycle = []
        for acc in range(2):
            fin = history_list[acc][-1]
            final_cycle.append(fin[0] + fin[1])
        cycle = np.max(final_cycle)

        final_energy = 0.0
        for acc in range(2):
            for event in history_list[acc]:
                now_model = None
                for model in cur_model_list:
                    if model[0] == event[2]:
                        now_model = model
                        break
                final_energy += energy_dict_list[acc][now_model[1]][event[3]]

    return cycle, final_energy


def main():

    LA = 5

    minLbF = -1.0
    minlatency = 999999999999
    for bLbF in range(10, 110, 1):
        LbF = bLbF * 0.01

        # load dependency
        dir_list = os.listdir(dependency_path)
        dep_dict = {}
        for dep_file in dir_list:
            with open(dependency_path / dep_file) as f:
                temp = yaml.load(f, Loader=yaml.FullLoader)
                dep_dict[dep_file] = temp
                test_sort(temp)
                print("DONE")

        total_total_latency_theirs = 0.0

        for benchmark in range(1):  # AR/VR-A
            print("\nCurrent benchmark:", dict_numtobenchmark[benchmark])

            for cl in range(3):  # Edge, Mobile, Cloud
                print("\nCurrent class:", dict_numtoclass[cl] + '\n')

                model_list = get_model_config(benchmark)

                latency_dict_list = [{}, {}]

                total_latency_fda_0 = 0.0
                total_latency_fda_1 = 0.0
                total_latency_0 = 0.0
                total_latency_1 = 0.0
                total_our_latency = 0.0

                for model in model_list:

                    model_name = model[0]
                    model_batch = model[1]
                    print("\nProcessing:", model_name)
                    base_dir = result_path / ('result_' + str(benchmark) + '_' + str(cl) + '_' + model_name)

                    with open(str(base_dir) + '_0.csv', 'r') as f:
                        latency_0 = get_latency(f.read().splitlines())
                    with open(str(base_dir) + '_1.csv', 'r') as f:
                        latency_1 = get_latency(f.read().splitlines())

                    with open(str(base_dir) + '_2.csv', 'r') as f:
                        latency_2 = get_latency(f.read().splitlines())
                    with open(str(base_dir) + '_3.csv', 'r') as f:
                        latency_3 = get_latency(f.read().splitlines())

                    # their latency (temporary)
                    '''select_0 = []
                    select_1 = []
                    for i in range(len(latency_0)):
                        if latency_0[i] > latency_1[i]:
                            select_1.append(latency_1[i])
                        else:
                            select_0.append(latency_0[i])
                    latency_0_sum = np.sum(select_0) * model_batch
                    latency_1_sum = np.sum(select_1) * model_batch'''

                    latency_fda_0_sum = np.sum(latency_2) * model_batch
                    latency_fda_1_sum = np.sum(latency_3) * model_batch

                    latency_0_sum = np.sum(latency_0) * model_batch
                    latency_1_sum = np.sum(latency_1) * model_batch

                    latency_dict_list[0][model_name] = latency_0
                    latency_dict_list[1][model_name] = latency_1

                    # Our latency
                    our_latencies = np.stack((latency_2, latency_3))
                    #opt_latency_sel = np.argmin(our_latencies, axis=0)
                    opt_latency_min = np.min(our_latencies, axis=0)
                    opt_latency_sum = np.sum(opt_latency_min) * model_batch

                    #print("\nlatency_FDA_0:", latency_fda_0_sum)
                    #print("latency_FDA_1:", latency_fda_1_sum)
                    #print("latency_HDA_0:", latency_0_sum)
                    #print("latency_HDA_1:", latency_1_sum)
                    #print("latency_ours :", opt_latency_sum)

                    total_latency_fda_0 += latency_fda_0_sum
                    total_latency_fda_1 += latency_fda_1_sum
                    total_latency_0 += latency_0_sum
                    total_latency_1 += latency_1_sum
                    total_our_latency += opt_latency_sum

                total_latency_theirs = schedule_hda(model_list, latency_dict_list, dep_dict, LbF, LA)
                if LbF == 0.54:
                    print("\ntotal_latency_FDA_1:", total_latency_fda_0)
                    print("total_latency_FDA_2:", total_latency_fda_1)
                    print("total_latency_HDA_1:", total_latency_0 )
                    print("total_latency_HDA_2:", total_latency_1)
                    print("total_latency_HDA  :", total_latency_theirs)
                    print("total_latency_ours :", total_our_latency)

                total_total_latency_theirs += total_latency_theirs

        if total_total_latency_theirs < minlatency:
            minLbF = LbF
            minlatency = total_total_latency_theirs
    print(minLbF, minlatency)

if __name__ == '__main__':
    main()