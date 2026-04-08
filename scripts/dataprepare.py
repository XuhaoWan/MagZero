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
    filename='dataprepare.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


OXI_DATA = {
    'H': [-1, 1], 'He': [0, 0],'Li': [-1, 1], 'Be': [0, 2], 'B': [-5, 3], 'C': [-4, 4], 'N': [-3, 5], 'O': [-2, 2], 'F': [-1, 0], 'Ne': [0, 0], 'Na': [-1, 1], 'Mg': [0, 2], 
    'Al': [-2, 3], 'Si': [-4, 4], 'P': [-3, 5], 'S': [-2, 6], 'Cl': [-1, 7], 'Ar': [0, 0], 'K': [-1, 1], 'Ca': [0, 2], 'Sc': [0, 3], 'Ti': [-2, 4], 'V': [-3, 5], 'Cr': [-4, 6], 
    'Mn': [-3, 7], 'Fe': [-2, 7], 'Co': [-3, 5], 'Ni': [-2, 4], 'Cu': [-2, 4], 'Zn': [-2, 2], 'Ga': [-5, 3], 'Ge': [-4, 4], 'As': [-3, 5], 'Se': [-2, 6], 'Br': [-1, 7], 'Kr': [0, 2], 
    'Rb': [-1, 1], 'Sr': [0, 2], 'Y': [0, 3], 'Zr': [-2, 4], 'Nb': [-3, 5], 'Mo': [-4, 6], 'Tc': [-1, 7], 'Ru': [-2, 8], 'Rh': [-3, 7], 'Pd': [0, 5], 'Ag': [-2, 3], 'Cd': [-2, 2], 
    'In': [-5, 3], 'Sn': [-4, 4], 'Sb': [-3, 5], 'Te': [-2, 6], 'I': [-1, 7], 'Xe': [0, 8], 'Cs': [-1, 1], 'Ba': [0, 2], 'La': [0, 3], 'Ce': [0, 4], 'Pr': [0, 5], 'Nd': [0, 4], 
    'Pm': [0, 3], 'Sm': [0, 3], 'Eu': [0, 3], 'Gd': [0, 3], 'Tb': [0, 4], 'Dy': [0, 4], 'Ho': [0, 3], 'Er': [0, 3], 'Tm': [-0, 3], 'Yb': [0, 3], 'Lu': [0, 3], 'Hf': [-2, 4], 
    'Ta': [-3, 5], 'W': [-4, 6], 'Re': [-3, 7], 'Os': [-4, 8], 'Ir': [-3, 9], 'Pt': [-3, 6], 'Au': [-3, 5], 'Hg': [-2, 2], 'Tl': [-5, 3], 'Pb': [-4, 4], 'Bi': [-3, 5], 'Po': [-2, 6], 
    'At': [-1, 7], 'Rn': [0, 6], 'Fr': [0, 1], 'Ra': [0, 2], 'Ac': [0, 3], 'Th': [-1, 4], 'Pa': [0, 5], 'U': [-1, 6], 'Np': [0, 7], 'Pu': [0, 8], 'Am': [0, 7], 'Cm': [0, 6], 
    'Bk': [0, 5], 'Cf': [0, 5], 'Es': [0, 4], 'Fm': [0, 3], 'Md': [0, 3], 'No': [0, 3], 'Lr': [0, 3], 'Rf': [0, 4], 'Db': [0, 5], 'Sg': [0, 6], 'Bh': [0, 7], 'Hs': [0, 8], 
    'Mt': [0, 6], 'Ds': [0, 6], 'Rg': [-1, 5], 'Cn': [0, 4], 'Nh': [0, 0], 'Fl': [0, 0], 'Mc': [0, 0], 'Lv': [-2, 4], 'Ts': [-1, 5], 'Og': [-1, 6]
}

def element_to_vector(symbol):
    # 使用 mendeleev 获取元素数据
    elem = element(symbol)
    
    # 最大范围设置
    max_atomic_number = 118  # 元素周期表最大原子序数
    max_period = 7           # 元素周期表最大周期
    max_group = 18           # 元素周期表最大族
    max_oxidation_state = 9  # 最大氧化态范围
    min_oxidation_state = -5 # 最小氧化态范围

    # 初始化 one-hot 编码向量
    atomic_number_vector = np.zeros(max_atomic_number)
    atomic_number_vector[elem.atomic_number - 1] = 1  # 原子序数从1开始

    period_vector = np.zeros(max_period)
    period_vector[elem.period - 1] = 1

    group_vector = np.zeros(max_group)
    if elem.group_id:
        group_vector[elem.group_id - 1] = 1  # 获取 Group 对象的数值

    max_oxidation_vector = np.zeros(max_oxidation_state + 1)
    max_oxidation = OXI_DATA[symbol][1]
    max_oxidation_vector[max_oxidation] = 1

    min_oxidation_vector = np.zeros(abs(min_oxidation_state) + 1)
    min_oxidation = OXI_DATA[symbol][0]
    min_oxidation_vector[abs(min_oxidation)] = 1

    # 连续值
    atomic_radius = elem.atomic_radius or 0.0
    atomic_weight = elem.atomic_weight or 0.0
    electronegativity = elem.en_pauling or 0.0
    ionization_energy = elem.ionenergies.get(1, 0.0)  # 第一电离能
    electron_affinity = elem.electron_affinity or 0.0

    # 拼接最终向量
    vector = np.concatenate([
        atomic_number_vector,
        period_vector,
        group_vector,
        max_oxidation_vector,
        min_oxidation_vector,
        [atomic_radius, atomic_weight, electronegativity, ionization_energy, electron_affinity]
    ])

    return torch.tensor(vector)



def struct_to_Hgraph(
    struct_path: str, 
    magion_indices: list[int],
    magion_edge_attrs: dict[str, list],
    magmom: dict[str, list],
    cutoff: float = 3.0
) -> HeteroData:
    """
    将MCIF文件转换为PyG异构图
    :param mcif_path: MCIF文件路径
    :param magion_indices: magion原子索引列表
    :param element_to_vector: 元素类型到特征向量的转换函数
    :param cutoff: 近邻边距离阈值
    :return: 异构图数据对象
    """
    # 读取晶体结构
    struct: Atoms = read(struct_path)
    n_atoms = len(struct)
    
    # 验证magion索引有效性
    magion_indices = np.unique(magion_indices)
    if not all(0 <= idx < n_atoms for idx in magion_indices):
        raise ValueError("magion_indices包含无效原子索引")
    
    # 验证磁矩数据完整性（核心新增部分）
    magmom = {int(key): value for key, value in magmom.items()}
    missing_mag = set(magion_indices) - set(magmom.keys())

    if missing_mag:
        raise ValueError(f"缺失以下magion原子的磁矩数据: {missing_mag}")
    
    extra_mag = set(magmom.keys()) - set(magion_indices)
    if extra_mag:
        raise ValueError(f"包含非magion原子的磁矩数据: {extra_mag}")
    
    # 验证磁矩维度
    for idx, moment in magmom.items():
        if len(moment) != 3:
            raise ValueError(f"原子 {idx} 的磁矩维度错误，应为3维，实际为{len(moment)}维")
        if not all(isinstance(v, (float, int)) for v in moment):
            raise ValueError(f"原子 {idx} 的磁矩包含非数值类型数据")
            
    
    # 创建原子类型掩码
    is_magion = np.zeros(n_atoms, dtype=bool)
    is_magion[magion_indices] = True
    atom_mask = ~is_magion
    
    # 初始化异构图
    data = HeteroData()
    
    # 生成节点特征（核心修改部分）
    # 为所有原子生成特征向量
    all_features = torch.stack(
        [element_to_vector(atom.symbol).clone().detach().to(torch.float32) for atom in struct]
    )
    
    # 分割到不同节点类型
    data['atom'].x = all_features[atom_mask]
    data['magion'].x = all_features[magion_indices]

    # 添加磁矩数据
    sorted_magion = sorted(magion_indices)
    mag_mom_tensor = torch.tensor(
        [magmom[idx] for idx in sorted_magion],
        dtype=torch.float
    )
    data['magion'].magmom = mag_mom_tensor
    
    
    # 初始化邻居列表（bothways=True确保双向边）
    nl = NeighborList(
        cutoffs=[cutoff / 2] * n_atoms,  # 单一切断距离
        self_interaction=False,
        bothways=True,
        skin=0.0
        
    )
    nl.update(struct)
    
    # 收集所有近邻边及其位移向量
    edge_dict = {}
    for i in range(n_atoms):
        neighbors, offsets = nl.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            # 计算考虑周期性的位移向量
            vec = struct.positions[j] + np.dot(offset, struct.cell) - struct.positions[i]
            edge_key = (i, j)
            edge_dict[edge_key] = vec
    
    # 转换为边索引和向量列表
    rows, cols = zip(*edge_dict.keys()) if edge_dict else ([], [])
    vectors = list(edge_dict.values())
    
    # 边类型分类
    node_types = np.where(is_magion, 'magion', 'atom')
    edge_configs = {
        ('atom', 'near', 'atom'): [],
        ('atom', 'near', 'magion'): [],
        ('magion', 'near', 'atom'): [],
        ('magion', 'near', 'magion'): [],
        ('magion', 'magion_edge', 'magion'): []
    }
    
    # 填充边类型
    for r, c in zip(rows, cols):
        src_type = node_types[r]
        dst_type = node_types[c]
        edge_type = (src_type, 'near', dst_type)
        if edge_type in edge_configs:
            edge_configs[edge_type].append((r, c))
    
    # 构建边数据
    for edge_type, pairs in edge_configs.items():
        if len(pairs) == 0:
            continue
            
        pairs = np.array(pairs)
        # 获取源和目标节点的全局索引
        src_global = np.where(atom_mask)[0] if edge_type[0] == 'atom' else magion_indices
        dst_global = np.where(atom_mask)[0] if edge_type[2] == 'atom' else magion_indices
        
        # 创建全局到局部索引的映射
        src_map = {g: l for l, g in enumerate(src_global)}
        dst_map = {g: l for l, g in enumerate(dst_global)}
        
        valid_pairs = []
        edge_vectors = []
        for (g_src, g_dst) in pairs:
            if g_src in src_map and g_dst in dst_map:
                local_src = src_map[g_src]
                local_dst = dst_map[g_dst]
                valid_pairs.append((local_src, local_dst))
                # 从预计算的edge_dict中获取向量
                edge_vectors.append(edge_dict[(g_src, g_dst)])
        
        if len(valid_pairs) > 0:
            edge_index = torch.tensor(valid_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_vectors, dtype=torch.float)  # [n_edges, 3]
            
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = edge_attr  
    
    # 创建magion专用边（五维特征）
    if len(magion_indices) >= 2:
        edge_features = []
        src_dst_pairs = []
        required_pairs = set()
        
        # 生成所有可能的有序对
        for i in magion_indices:
            for j in magion_indices:
                if i != j:
                    # 生成规范化的键（小索引在前）
                    #a, b = sorted([i, j])

                    pair_key = f"{i}-{j}"
                    required_pairs.add(pair_key)
                    
                    # 检查是否提供该对的属性
                    if pair_key not in magion_edge_attrs:
                        raise ValueError(f"缺少magion原子对 {pair_key} 的边属性")
                    
                    # 记录所有实际边（保持原有i,j顺序）
                    src_dst_pairs.append((i, j))
                    edge_features.append(magion_edge_attrs[pair_key])
        
        # 验证是否提供了所有必要边属性
        provided_pairs = set(magion_edge_attrs.keys())
        missing_pairs = required_pairs - provided_pairs
        if missing_pairs:
            raise ValueError(f"缺少以下magion原子对的边属性: {missing_pairs}")
        
        # 转换为局部索引
        magion_map = {g: l for l, g in enumerate(magion_indices)}
        edge_index = torch.tensor(
            [ (magion_map[i], magion_map[j]) for i, j in src_dst_pairs ],
            dtype=torch.long
        ).T
        
        # 转换边特征为张量
        edge_features_padded = []

        for feat in edge_features:
            feat = torch.tensor(feat, dtype=torch.float)
            if feat.shape[0] == 5:  # 如果是 5 维
                feat = torch.cat([feat, torch.zeros(2)], dim=0)  # 右侧补 2 个 0
            edge_features_padded.append(feat)

        edge_features = edge_features_padded      
        edge_attr = torch.stack(edge_features_padded) 
        
        # 添加到异构图
        data['magion', 'magion_edge', 'magion'].edge_index = edge_index
        data['magion', 'magion_edge', 'magion'].edge_attr = edge_attr
    
    data.validate()
    return data


def parse_struct(struct_file):
    coord_to_id = {}
    id_info = []
    current_id = 0
    
    with open(struct_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ATOM'):
            # 解析原子类别
            class_match = re.match(r'ATOM\s+(-?\d+):', line)
            atom_class = int(class_match.group(1))
            
            # 解析主原子坐标
            coord_match = re.search(r'X=([\d.]+) Y=([\d.]+) Z=([\d.]+)', line)
            main_coord = (
                round(float(coord_match.group(1)), 6),
                round(float(coord_match.group(2)), 6),
                round(float(coord_match.group(3)), 6)
            )
            
            # 获取MULT值
            mult = 1
            i += 1
            while i < len(lines):
                if 'MULT=' in lines[i]:
                    mult = int(re.search(r'MULT=\s*(\d+)', lines[i]).group(1))
                    break
                i += 1
            
            # 收集所有等效坐标
            all_coords = [main_coord]
            i += 1  # 移到MULT行下一行
            
            for _ in range(mult-1):
                if i >= len(lines): break
                eq_match = re.match(r'.*X=([\d.]+) Y=([\d.]+) Z=([\d.]+)', lines[i].strip())
                if eq_match:
                    eq_coord = (
                        round(float(eq_match.group(1)), 6),
                        round(float(eq_match.group(2)), 6),
                        round(float(eq_match.group(3)), 6)
                    )
                    all_coords.append(eq_coord)
                    i += 1
            
            # 分配原子ID
            for coord in all_coords:
                coord_to_id[coord] = current_id
                id_info.append({
                    'id': current_id,
                    'class': atom_class,
                    'coord': coord
                })
                current_id += 1
        else:
            i += 1
    
    return coord_to_id, id_info

def parse_magnetic_moments(log_file, coord_to_id, target_ids):
    target_set = set(target_ids)
    found_ids = set()
    moments = defaultdict(list)
    
    # 改进的正则表达式，允许更灵活的格式
    pattern = re.compile(
        r'r\[(\d+)\]\s*=\s*\[\s*([\d\.,\s-]+)\]\s*.*?M\[\d+\]\s*=\s*\[\s*([\d\.,\s-]+)\s*\]',
        re.DOTALL
    )
    
    with open(log_file, 'r') as f:
        for line in f:
            # 遇到停止关键词立即终止
            if "indx_by_element" in line:
                break
            
            matches = pattern.findall(line)

            if not matches:
                continue
  
            for r_idx, r_coord_str, m_str in matches:
                if len(found_ids) == len(target_set):
                    print("Find all magions, stop reading")
                    return dict(moments)
                
                # 解析坐标
                r_coord = tuple(
                    round(float(x), 6)
                    for x in re.split(r'[,\s]+', r_coord_str.strip())
                    if x.strip()
                )
                
                # 匹配原子ID
                # atom_id = coord_to_id.get(r_coord, -1)
                atom_id = -1
                for perm in itertools.permutations(r_coord):
                    if perm in coord_to_id:
                        atom_id = coord_to_id[perm]
                        break  # 找到第一个匹配的就停止搜索
                if atom_id == -1:
                    continue
                
                # 检查是否为需要的目标原子且未记录过
                if atom_id in target_set and atom_id not in found_ids:
                    # 解析磁矩
                    moment = [
                        round(float(x), 6)
                        for x in re.split(r'[,\s]+', m_str.strip())
                        if x.strip()
                    ]
                    moments[atom_id] = moment
                    found_ids.add(atom_id)
    # for aid in target_ids:
    #     if aid not in moments:
    #         moments[aid] = [0.0, 0.0, 0.0]

    return dict(moments)


def get_cor(case, dir):
    indmf = Indmfl(case)
    file = os.path.join(dir, f'{case}.indmfl')
    indmf.read(file)
    return list(indmf.atoms.keys())


def save_heterograph(data: HeteroData, path: str):
    # 确保路径以.pt或.pth结尾
    if not path.endswith(('.pt', '.pth')):
        path += '.pt'
    
    # 保存整个HeteroData对象
    torch.save(data, path)
    return None


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


def process_case_outputdmf1(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        content = f.read()

    # 使用正则表达式分割各个NCOR块
    blocks = re.findall(r':NCOR.*?(?=\s*:NCOR|\Z)', content, re.DOTALL)
    
    result_dict = {}
    
    for block in blocks:
        # 分割块中的每一行
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
        
        # 解析头部信息
        header = lines[0]
        match = re.match(r':NCOR\s+([\d.]+)\s+(\d+)\s+(\d+)', header)
        if not match:
            continue
        
        nf = float(match.group(1))    # 轨道总占据
        block_id = int(match.group(2))  # 块ID
        n_orbit = int(match.group(3)) # 轨道总数
        
        # 提取数据行
        data_lines = lines[1:n_orbit+1]
        if len(data_lines) != n_orbit:
            print(f"警告：块{block_id}数据行数不匹配，跳过")
            continue
        
        vector = []
        for row_idx, line in enumerate(data_lines):
            # 分割数据列
            cols = line.split()
            # 计算需要提取的列索引（每行提取第2*row_idx列）
            target_col = 2 * row_idx
            if target_col >= len(cols):
                print(f"错误：块{block_id}第{row_idx}行数据不足")
                vector = []
                break
            
            try:
                value = float(cols[target_col])
                vector.append(value / nf)  # 计算占据比例
            except (ValueError, IndexError):
                print(f"格式错误：块{block_id}第{row_idx}行数据异常")
                vector = []
                break
        
        if len(vector) == n_orbit:
            result_dict[block_id] = vector
    
    return result_dict


def process_case(case_dir: str, datasave_dir: str, case_name: str) -> Dict[str, List[float]]:
    """处理单个case目录"""
    dlt_result = {}
    
    # 查找xw553结尾的目录
    xw553_dirs = glob.glob(os.path.join(case_dir, '*xw553'))
    if not xw553_dirs:
        logging.error(f"Case {case_name}: Missing xw553 directory")
        return {}
    
    xw553_dir = xw553_dirs[0]

    data_path = os.path.join(datasave_dir, f"{case_name}.pt")
    struct_file =os.path.join(case_dir, f"{case_name}.struct")
    moment_file =os.path.join(case_dir, 'mminfo.json')
    ncor_file =os.path.join(xw553_dir, f"{case_name}.outputdmf1")

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
        ncor_result = process_case_outputdmf1(ncor_file)
        #print("处理结果：", ncor_result)
        for a, b in pairs:
            key = f"{a}-{b}"
            #print("keys:", key)
            try:
                # 获取a的新编码
                new_code = id_map[a]
                tar_code = id_map[b]
                num_dir = os.path.join(xw553_dir, str(new_code))

                ncorblk_idx = orig_ids.index(b) + 1
                occp = ncor_result[ncorblk_idx]
                scaled_occp = [x * len(occp) for x in occp]
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
                dltinfowithocc = [a * b for a, b in zip(scaled_occp, hybrid_values)]
                dlt_result[key] = dltinfowithocc
                
            except Exception as e:
                logging.error(f"Case {case_name} pair {key}: {str(e)}")
                continue
    
        data = struct_to_Hgraph(struct_file, magion_indices=sorted_m_ids, magion_edge_attrs=dlt_result, magmom=magnetic_moments)
        save_heterograph(data, data_path)
    except Exception as e:
        logging.error(f"Case {case_name}: {str(e)}")
    
    # print(dlt_result)

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
        save_path = os.path.join(datasave_dir, f"{case_name}.pt")
        # 检查是否已存在处理结果
        if os.path.exists(save_path):
            logging.info(f"Skipping processed case: {case_name}")
            continue
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
        print("Usage: python datacreate.py <target_directory>")
        sys.exit(1)
    
    datasave_dir = '/home/xw553/mnt/w/Hdata'
    main(sys.argv[1], datasave_dir)

    # with open("dlt.output", "w") as f:
    #     for case, data in results.items():
            
    #         print(f"\nCase: {case}", file=f)  
    #         for pair, values in data.items():
    #             print(f"  {pair}: {values}", file=f)  

    
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

