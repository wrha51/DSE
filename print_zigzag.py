import os, sys
import argparse
import importlib.util
import yaml
import xml.etree.ElementTree as ET
import itertools

from pathlib import Path

dict_index_to_loop = {7: 'B', 6: 'K', 5: 'C', 4: 'OY', 3: 'OX', 2: 'FY', 1: 'FX'}
dict_loop_to_index = {'B': 7, 'K': 6, 'C': 5, 'OY': 4, 'OX' : 3, 'FY': 2, 'FX': 1}

def main(net_name, prefix, net_dir):

    map_dir = Path('./_results') / ('results_' + net_name + '_' + prefix) / 'best_su_best_tm'

    # check arguments
    if not net_dir.is_dir():
        sys.exit("No network folder! Please check it!")

    if not map_dir.is_dir():
        sys.exit("No mapping folder! Please check it!")

    # load layer information
    net_module_spec = importlib.util.spec_from_file_location(net_name, str(net_dir / (net_name + '.py')))
    net_module = importlib.util.module_from_spec(net_module_spec)
    net_module_spec.loader.exec_module(net_module)
    layer_info = net_module.layer_info
    len_layers = len(layer_info)

    # load mappings
    layer_mappings = [[]]
    for i in range(len_layers):
        layer_mappings.append([])
    map_files = [_ for _ in os.listdir(map_dir) if _.endswith('max_ut.xml')]
    for map_file in map_files:
        name_blocks = map_file.split('_')
        layer_index = int(name_blocks[-5][1:])
        mem_index = int(name_blocks[-4][1:])
        su_index = int(name_blocks[-3][2:])

        tree = ET.parse((map_dir / map_file).absolute())
        root = tree.getroot()

        utilization = tree.find(".//utilization_with_data_loading").tail

        su = tree.find(".//spatial_unrolling")
        su = eval(su[0].tail)
        x_group = []
        y_group = []
        for i1 in su:
            if len(i1) != 0:
                cur_group = i1[0]
                for i2, elem in enumerate(cur_group):
                    cur_group[i2] = list(elem)
                    cur_group[i2][0] = cur_group[i2][0]
                x_group += cur_group

                cur_group = i1[1]
                for i2, elem in enumerate(cur_group):
                    cur_group[i2] = list(elem)
                    cur_group[i2][0] = cur_group[i2][0]
                y_group += cur_group
        x_group.sort(reverse=True)
        y_group.sort(reverse=True)

        print('\n[', end='')
        for i1 in x_group[:-1]:
            print(i1[0] + ':', str(i1[1]) + ',', end=' ')
        print(x_group[-1][0] + ':', str(x_group[-1][1]) + ']', end='')

        print(', [', end='')
        for i1 in y_group[:-1]:
            print(i1[0] + ':', str(i1[1]) + ',', end=' ')
        print(y_group[-1][0] + ':', str(y_group[-1][1]) + ']', end='')

        print('\t' + utilization, end='')

    print("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze ZigZag mappings")
    parser.add_argument("-nn", "--net_name", help='name of network', default='ResNet18')
    parser.add_argument("-p", "--prefix", help='other name of input file', default='ResNet18')
    parser.add_argument("-nd", "--net_dir", help='network directory', default='./NN_layers')
    args = parser.parse_args()

    net_name = args.net_name
    prefix = args.prefix
    net_dir = Path(args.net_dir)

    main(net_name, prefix, net_dir)