import re
from collections import defaultdict
import os
import glob
import itertools
from typing import Dict, List, Tuple
from indmffile import Indmfl
import json

def parse_struc_eledict(struct_file):
    element_sequence = []
    with open(struct_file, 'r') as file:
        lines = file.readlines()

    current_element = None
    current_count = 1

    for line in lines:
        line = line.strip()

        if 'MULT' in line:
            match = re.search(r'MULT=\s*(\d+)', line)

            if match:
                current_count = int(match.group(1))

        elif 'NPT' in line:
            match = re.search(r'([A-Za-z0-9:.]+(?:\s*[0-9]+)?)\s*NPT=', line)
            if match:
                current_element = match.group(1).strip()
                current_element = re.sub(r'\d+', '', current_element)
                if current_element:
                    element_sequence.extend([current_element]*current_count)
                current_count = 1
    
    element_dict = {idx-1:element for idx, element in enumerate(element_sequence, 1)}
    return element_dict


def parse_struct(struct_file, log):
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

def parse_magnetic_moments(log_file, coord_to_id, target_ids, log):
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
            if line.startswith('angles:'):
                # 提取角度值
                parts = line.strip().split()
                if len(parts) != 4:
                    print(f"Invalid format: {line.strip()}", file=log)
                    continue
                
                try:
                    alpha, beta, gamma = map(float, parts[1:4])
                except ValueError:
                    print(f" Non-numeric values in: {line.strip()}", file=log)
                    continue

                # 检查正交性 (允许1e-6的浮点误差)
                if not all(abs(angle - 90.0) < 2.0 for angle in (alpha, beta, gamma)):
                    print("ATTENTION!!! non-orthogonal lattice detected", alpha, beta, gamma, file=log)

            if "indx_by_element" in line:
                break
            
            matches = pattern.findall(line)

            if not matches:
                continue
  
            for r_idx, r_coord_str, m_str in matches:
                if len(found_ids) == len(target_set):
                    sorted_moments = {k: moments[k] for k in sorted(moments)}
                    return dict(sorted_moments)
                
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
    #print(sorted(moments), file=log)
    sorted_moments = {k: moments[k] for k in sorted(moments)}
    # for aid in target_ids:
    #     if aid not in moments:
    #         moments[aid] = [0.0, 0.0, 0.0]

    return dict(sorted_moments)


def get_cor(case, dir):
    indmf = Indmfl(case)
    file = os.path.join(dir, f'{case}.indmfl')
    indmf.read(file)
    return list(indmf.atoms.keys())


def process_case(case_dir: str, case_name: str, log) -> Dict[str, List[float]]:
    """处理单个case目录"""
    
    
    struct_file =os.path.join(case_dir, f"{case_name}.struct")
    log_file =os.path.join(case_dir, 'cif2struct.log')
    mm_file =os.path.join(case_dir, "mminfo.json")

    try:
        # 获取强关联原子ID列表
        elelist = parse_struc_eledict(struct_file)
        coord_map, _ = parse_struct(struct_file, log)
        cor_ids = get_cor(case_name, case_dir)
        cor_ids = [x - 1 for x in cor_ids]

        magnetic_moments = parse_magnetic_moments(log_file, coord_map, cor_ids, log)

        mminfo = {"moments": magnetic_moments, "elements list": elelist}
        with open(mm_file, "w") as f:
            json.dump(mminfo, f, indent=4)

        print('elelist:', elelist, file=log)
        print('magnetic_moments ID:', magnetic_moments, file=log)

                
    except Exception as e:
        print(f"Case {case_name}: {str(e)}", file=log)
    
    return None

def main(target_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """主处理函数"""
    
    getmmlog = os.path.join(target_dir, 'getmm.log')
    # 遍历所有一级子目录
    with open (getmmlog, 'w') as log:
        for case_name in os.listdir(target_dir):
            case_path = os.path.join(target_dir, case_name)
            if not os.path.isdir(case_path):
                continue
            
            print(f"Processing case: {case_name}", file=log)
            process_case(case_path, case_name, log)
        
    
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python processdlt.py <target_directory>")
        sys.exit(1)
    
    main(sys.argv[1])

