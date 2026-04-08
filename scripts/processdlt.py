import numpy as np
import torch
from torch_geometric.data import HeteroData
from ase.io import read
from ase import Atoms
from ase.visualize import view
from ase.neighborlist import NeighborList
from mendeleev import element

import re
from collections import defaultdict
import os
import glob
import itertools
import logging
from typing import Dict, List, Tuple
from indmffile import Indmfl
import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from ase.io import read
from ase import Atoms
from ase.visualize import view
from ase.neighborlist import NeighborList
from mendeleev import element



# 配置日志记录
logging.basicConfig(
    filename='dlt.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_cor(case, dir):
    indmf = Indmfl(case)
    file = os.path.join(dir, f'{case}.indmfl')
    indmf.read(file)
    return list(indmf.atoms.keys())


def parse_columns(line: str) -> List[float]:
    """解析数据行并返回偶数列数据（第2,4,6...列）"""
    values = list(map(float, line.strip().split()))
    return values[1::2]  # 取索引1,3,5...即第2,4,6...列

def calculate_hybrid(dltup_path: str, dltdn_path: str) -> List[float]:
    """计算两个文件的差值"""
    with open(dltup_path) as f1, open(dltdn_path) as f2:
        # 跳过第一行，读取第二行
        f1.readline()
        f2.readline()
        line1 = f1.readline()
        line2 = f2.readline()
    
    values1 = parse_columns(line1)
    values2 = parse_columns(line2)
    return [v1 - v2 for v1, v2 in zip(values1, values2)]

def process_case(case_dir: str, datasave_dir: str, case_name: str) -> Dict[str, List[float]]:
    """处理单个case目录"""
    dlt_result = {}
    
    # 查找xw553结尾的目录
    xw553_dirs = glob.glob(os.path.join(case_dir, '*xw553'))
    if not xw553_dirs:
        logging.error(f"Case {case_name}: Missing xw553 directory")
        return {}
    
    xw553_dir = xw553_dirs[0]
    
    struct_file =os.path.join(case_dir, f"{case_name}.struct")
    moment_file =os.path.join(case_dir, 'mminfo.json')
    
    try:
        # 获取强关联原子ID列表
        orig_ids = get_cor(case_name, xw553_dir)
        orig_ids = [x - 1 for x in orig_ids]

        with open(moment_file, "r") as f:
            loaded_data = json.load(f)
            magnetic_moments = loaded_data["moments"]

        m_ids = list(magnetic_moments.keys())
        m_ids = [int(x) for x in m_ids]
        sorted_m_ids = sorted(m_ids)

        #print('magions ID:', m_ids)
        #print('orig ID:', orig_ids)
        # orig_ids = get_cor(case_name, xw553_dir)
        if not orig_ids:
            logging.error(f"Case {case_name}: Empty ID list")
            return {}
        
        # 生成重编码映射 (e.g. 4->1, 5->2, 8->3)
        sorted_ids = sorted(orig_ids)
        id_map = {orig: i+1 for i, orig in enumerate(sorted_ids)}

        #print('id map:', id_map)
        # 生成所有原子对组合
        pairs = itertools.permutations(sorted_m_ids, 2)
        
        for a, b in pairs:
            key = f"{a}-{b}"

            try:
                # 获取a的新编码
                new_code = id_map[a]
                tar_code = id_map[b]

                num_dir = os.path.join(xw553_dir, str(new_code))
                
                # 验证数字目录存在
                if not os.path.isdir(num_dir):
                    logging.error(f"Case {case_name}: Missing directory {num_dir}")
                    continue
                
                # 构建文件路径
                dltup_path = os.path.join(num_dir, f"{case_name}.dlt{tar_code}")
                dltdn_path = os.path.join(num_dir, f"{case_name}.dlt{tar_code}dn")
                
                # 验证文件存在
                if not os.path.isfile(dltup_path):
                    logging.error(f"Case {case_name}: Missing file {dltup_path}")
                    continue
                if not os.path.isfile(dltdn_path):
                    logging.error(f"Case {case_name}: Missing file {dltdn_path}")
                    continue
                
                # 计算杂化值
                hybrid_values = calculate_hybrid(dltup_path, dltdn_path)
                
                dlt_result[key] = hybrid_values
                
            except Exception as e:
                logging.error(f"Case {case_name} pair {key}: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Case {case_name}: {str(e)}")


    return dlt_result

def main(target_dir: str, datasave_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """主处理函数"""
    all_dltresults = {}
    
    # 遍历所有一级子目录
    for case_name in os.listdir(target_dir):
        case_path = os.path.join(target_dir, case_name)
        if not os.path.isdir(case_path):
            continue
            # 构建预期保存路径
        print(f"Processing case: {case_name}")
        try:
            case_result = process_case(case_path, datasave_dir, case_name)
            if case_result:
                all_dltresults[case_name] = case_result
            else:
                logging.warning(f"Case {case_name}: No valid results")
        except Exception as e:
            logging.error(f"Error processing case {case_name}: {e}")
    
    return all_dltresults

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process.py <target_directory>")
        sys.exit(1)
    
    datasave_dir = '/home/xw553/mnt/w/Hdata'
    results = main(sys.argv[1], datasave_dir)

    with open("dlt.output", "w") as f:
        for case, data in results.items():
            
            print(f"\nCase: {case}", file=f)  
            for pair, values in data.items():
                print(f"  {pair}: {values}", file=f)  

    
    # 使用示例
    # struct_file = '0.156.CaMnGe2O6.struct'
    # log_file = '156cif2struct.log'
    # target_ids = [3, 2]  # 需要查询的原子ID列表

    # # 解析结构文件
    # coord_map, atom_info = parse_struct(struct_file)

    # # 解析磁矩信息
    # magnetic_moments = parse_magnetic_moments(log_file, coord_map, target_ids)

    # # 输出结果
    # print("原子ID与坐标对应关系：", coord_map)

    # print("\n磁矩信息：", magnetic_moments)


# 使用示例
# struct_file = "0.236.CaFe4Al8.struct"
# wsdict = {'9-10': [1.73, 0.3,0.35,-0.03,0.01], '9-11': [1.26,0.3,0.35,-0.03,0.01], '9-12': [1.74,0.3,0.35,-0.03,0.01], '10-9': [1.73, 0.3,0.35,-0.03,0.01], '11-9': [1.73, 0.3,0.35,-0.03,0.01], '10-11': [0.77,0.3,0.35,-0.03,0.01], '10-12': [0.84,0.3,0.35,-0.03,0.01], '11-10': [1.73, 0.3,0.35,-0.03,0.01],'11-12': [0.81, 0.3,0.35,-0.03,0.01],
#         '12-9': [1.73, 0.3,0.35,-0.03,0.01],'12-10': [1.73, 0.3,0.35,-0.03,0.01],'12-11': [8.81, 0.3,0.35,-0.03,0.01]}
# mmdict = {9: [0.037, 0.037, 0.712], 10: [0.037, 0.037, 0.712], 11: [0.037, 0.037, -0.712], 12: [0.037, 0.037, -0.712]}
# data = struct_to_Hgraph(struct_file, magion_indices=[9, 10, 11, 12], magion_edge_attrs=wsdict, magmom=mmdict)

# print("Node information:")
# print("Atom nodes", data['atom'].x)
# print("Magion nodes:", data['magion'].x)
# print("\nEdge information:")
# for edge_type in data.edge_types:
#     print(f"{edge_type}: {data[edge_type].edge_index}")
#     print(f"{edge_type}: {data[edge_type].edge_attr}")
# print("Magmom information:", data['magion'].magmom)


# # 使用示例
# mcif_file_path = "0.236.CaFe4Al8.struct"
# #hetero_graph = mcif_to_hetero_graph(mcif_file_path)
# structure = read(mcif_file_path)
# structure.set_pbc(True)
# supercell = structure.repeat((2, 2, 2))

# # 获取原子位置和原子类型
# positions = structure.get_positions(wrap=True)
# print(positions)
# atomic_numbers = structure.get_atomic_numbers()
# print(atomic_numbers)
# atom_types = structure.get_chemical_symbols()
# print(atom_types)
# lattice = structure.get_cell_lengths_and_angles()
# print(lattice)
# d_matrix = structure.get_all_distances()
# print(d_matrix)
# view(structure, cell = True)

# edge_src, edge_dst, edge_length, edge_shift = neighbor_list("ijdS", a=structure, cutoff=3.5, self_interaction=False)
# for i in range(len(edge_src)):
#     print(edge_src[i])
#     print(edge_dst[i])
#     print(edge_length[i])
#     print(edge_shift[i])

#vector = element_to_vector("Og")
#print(vector)