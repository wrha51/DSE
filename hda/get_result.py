import os
import shlex
import shutil
import subprocess
from pathlib import Path

hw_config_path = Path('./hw')
map_path = Path('./mapping')
result_path = Path('./result')

dict_numtobenchmark = {0: 'AR/VR-A', 1: 'AR/VR-B', 2: 'MLPerf'}
dict_numtoclass = {0: 'Edge', 1: 'Mobile', 2: 'Cloud'}


def get_hw_config(benchmark, cl, split_BW, split_PE, cur_BW, cur_PE, cur_MEM):
    total_BW = split_BW[0] + split_BW[1]
    total_PE = split_PE[0] + split_PE[1]

    os.makedirs(result_path, exist_ok=True)
    hw_config_path_list = []

    BW = []
    PE = []
    MEM = []

    BW.append(str(int(split_BW[0] / total_BW * cur_BW)))  # NVDLA
    BW.append(str(int(split_BW[1] / total_BW * cur_BW)))  # Shi
    BW.append(str(cur_BW))  # Ours - NVDLA
    BW.append(str(cur_BW))  # Ours - Shi

    PE.append(str(int(split_PE[0] / total_PE * cur_PE)))  # NVDLA
    PE.append(str(int(split_PE[1] / total_PE * cur_PE)))  # Shi
    PE.append(str(cur_PE))  # Ours - NVDLA
    PE.append(str(cur_PE))  # Ours - Shi

    MEM.append(str(int(split_BW[0] / total_BW * cur_MEM)))  # NVDLA
    MEM.append(str(int(split_BW[1] / total_BW * cur_MEM)))  # Shi
    MEM.append(str(cur_MEM))  # Ours - NVDLA
    MEM.append(str(cur_MEM))  # Ours - Shi

    for cur in range(4):
        cur_hw_config_name = 'config_' + benchmark + '_' + cl + '_' + str(cur) + '.m'
        cur_hw_config_path = hw_config_path / cur_hw_config_name
        hw_config_path_list.append(str(cur_hw_config_path))
        with open(cur_hw_config_path, 'w') as f:
            f.write('num_pes: ' + PE[cur] + '\n')
            f.write('l1_size_cstr: 512\n')
            f.write('l2_size_cstr: ' + MEM[cur] + '\n')
            f.write('noc_bw_cstr: ' + BW[cur] + '\n')
            f.write('offchip_bw_cstr: ' + BW[cur] + '\n')
        print("Config write done:", cur)

    return hw_config_path_list


def get_model_config(benchmark):
    if benchmark == 0:
        model_list = [['Resnet50', 2], ['UNet', 4], ['MobileNetV2', 4]]

    return model_list


def execute_MAESTRO(benchmark, cl, model_list, hw_list):
    for model in model_list:
        for i, hw in enumerate(hw_list):
            cur_map_path = str(map_path) + "\\" + model[0] + '_' + str(i)
            command = "./MAESTRO.exe --HW_file='" + hw + "' --Mapping_file='" + cur_map_path + ".m' --print_res=true --print_res_csv_file=true --print_log_file=false"
            print("Command:", command)

            p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
            (pout, perr) = p.communicate()
            p_status = p.wait()

            cur_result_path = str(result_path) + '/result_' + benchmark + '_' + cl + '_' + model[0] + '_' + str(i)
            shutil.move(cur_map_path + '.csv', cur_result_path + '.csv')
            with open(cur_result_path + '.log', 'w') as f:
                f.write(pout.decode("utf-8"))

            print("Done executing!")


def main():
    ''' BW_benchmarks = [[[4, 12], [40, 24], [224, 32]], \
                        [[4, 12], [48, 16], [128, 128]], \
                        [[4, 12], [32, 32], [160, 96]]]'''
    BW_benchmarks = [[[4, 12], [40, 24], [224, 32]]]  # 0: AR/VR-A, 1: AR/VR-B, 2: MLPerf
    # 0: Edge, 1: Mobile, 2: Cloud
    # NVDLA / Shi
    ''' PE_benchmarks = [[[128, 896], [1792, 2304], [9728, 6656]], \
                        [[128, 896], [1536, 2560], [12032, 4352]], \
                        [[64, 960], [1280, 2816], [8192, 8192]]]'''
    PE_benchmarks = [[[128, 896], [1792, 2304], [9728, 6656]]]

    BW_class = [16, 64, 256]  # GB/s 단위, 0: Edge, 1: Mobile, 2: Cloud
    PE_class = [1024, 4096, 16384]  # 갯수 단위
    MEM_class = [4, 8, 16]  # MiB 단위

    # for benchmark in range(3): # AR/VR-A, AR/VR-B, MLPerf
    for benchmark in range(1):  # AR/VR-A
        print("\nCurrent benchmark:", dict_numtobenchmark[benchmark])

        split_total_BW = BW_benchmarks[benchmark]
        split_total_PE = PE_benchmarks[benchmark]

        for cl in range(3):  # Edge, Mobile, Cloud
            print("\nCurrent class:", dict_numtoclass[cl] + '\n')

            split_BW = split_total_BW[cl]
            split_PE = split_total_PE[cl]

            cur_BW = BW_class[cl] * 5  # 200MHz / 1byte integer 기준 --> 5 Elements/cycle
            cur_PE = PE_class[cl]
            cur_MEM = MEM_class[cl] * 1048576  # MiB

            hw_list = get_hw_config(str(benchmark), str(cl), split_BW, split_PE, cur_BW, cur_PE, cur_MEM)

            model_list = get_model_config(benchmark)

            execute_MAESTRO(str(benchmark), str(cl), model_list, hw_list)


if __name__ == '__main__':
    main()
