import re
import numpy as np
from indmffile import Indmfl
import os
import shutil
import subprocess


def parse_indmfi(file_path):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    blocks = []
    idx = 0
    while idx < len(lines):
        
        line = lines[idx].strip()
        if '# dim' in line:  # new block
            dimension = int(line.split()[0])  # get dim
            idx += 1  # next
            
            # get num in sigind
            diagonal_numbers = []
            for i in range(dimension):
                row = list(map(int, lines[idx + i].strip().split()))
                diagonal_numbers.append(row[i])  
            idx += dimension  # go next block

            # create position tuple
            half_size = len(diagonal_numbers) // 2
            tuples = [(num, 1) for num in diagonal_numbers[:half_size]] + [(num, -1) for num in diagonal_numbers[half_size:]]
            blocks.append(tuples)
        else:
            idx += 1  # next

    return blocks


def modify_sig_inp(input_file, output_file, positions):
    """
    Modifies the value at the specified position in the `s_oo` list in the sig.inp file, which can be increased or decreased respectively.

    :param input_file: input file name, e.g. 'sig.inp'
    :param output_file: name of the output file, e.g. 'sig_modified.inp'
    :param positions: the index to be modified and the corresponding increment operation, in the form [(index, increment)].
    """
    with open(input_file, 'r') as file:
        content = file.read()

    # s_oo
    s_oo_match = re.search(r"# s_oo=\s*\[([^\]]+)\]", content)
    if not s_oo_match:
        raise ValueError("Not found s_oo")

    # to float
    s_oo_str = s_oo_match.group(1)
    s_oo_list = list(map(float, s_oo_str.split(',')))

    # change energy
    for pos, increment in positions:
        pos -= 1
        if 0 <= pos < len(s_oo_list):
            s_oo_list[pos] += increment
        else:
            print(f"Warning: index {pos} over range, skip it!")

    # to str
    modified_s_oo_str = ', '.join(f"{x:.1f}" for x in s_oo_list)

    # replace
    modified_content = re.sub(
        r"# s_oo=\s*\[[^\]]+\]",
        f"# s_oo= [{modified_s_oo_str}]",
        content
    )

    # write
    with open(output_file, 'w') as file:
        file.write(modified_content)

    print(f"Successfuly get new file: {output_file}")


def execute_script(script_name):

    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")


def create_subfolders(target_path, coratom, blocks):

    files = [f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
    
    for i in range(1, coratom + 1):
        subfolder = os.path.join(target_path, str(i))
        os.makedirs(subfolder, exist_ok=True)  

        # 复制文件到子文件夹
        for file in files:
            shutil.copy(os.path.join(target_path, file), subfolder)
        
        # 在每个子文件夹中先重命名 sig.inp 为 sig.inp_o
        sig_inp_path = os.path.join(subfolder, 'sig.inp')
        if os.path.exists(sig_inp_path):
            os.rename(sig_inp_path, os.path.join(subfolder, 'sig.inp_o'))
            print(f"Renamed sig.inp to sig.inp_o in folder {subfolder}")

        # 在每个子文件夹中执行 modify_sig_inp
        os.chdir(subfolder)  # 切换到子文件夹
        modify_sig_inp('sig.inp_o', 'sig.inp', blocks[i-1])  # 执行 modify_sig_inp 函数
        os.chdir(target_path)  # 切换回原文件夹

if __name__ == '__main__':
    # 示例用法
    # input_file = "sig.inp"
    # output_file = "sig_modified.inp"
    # positions_to_modify = [(0, 1), (10, -1), (20, 1)]  # 格式为 (索引, 增量)
    # modify_sig_inp(input_file, output_file, positions_to_modify)
    target_path = os.getcwd()
    indmfi_file = [f for f in os.listdir(target_path) if f.endswith('.indmfi') and os.path.isfile(os.path.join(target_path, f))]
    indmfif_path = os.path.join(target_path, indmfi_file[0])
    blocks = parse_indmfi(indmfif_path)
    cornum = len(blocks)
    create_subfolders(target_path, cornum, blocks)
    
