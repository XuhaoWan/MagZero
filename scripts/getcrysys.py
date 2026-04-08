import os
import glob
import re
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



def find_crystal_system(mcif_path):
    """使用 pymatgen 获取晶系"""
    try:
        # 解析 CIF 文件
        parser = CifParser(str(mcif_path))
        structure = parser.get_structures()[0]
        
        # 获取晶系
        spg_analyzer = SpacegroupAnalyzer(structure)
        return spg_analyzer.get_crystal_system()
    
    except Exception as e:
        print(f"Error processing {mcif_path}: {str(e)}")
        return "Unknown"

def main():
    hdata_dir = 'Hdata'
    mag_dirs = [f'magzero{i}' for i in range(1, 10)]
    results = {}

    # 获取所有.pt文件
    pt_files = glob.glob(os.path.join(hdata_dir, '*.pt'))
    if not pt_files:
        print(f"错误：目录 {hdata_dir} 中没有找到.pt文件")
        return

    for pt_path in pt_files:
        base = os.path.splitext(os.path.basename(pt_path))[0]
        found = False
        
        # 在magzero目录中查找对应的子文件夹
        for mag_dir in mag_dirs:
            subfolder = os.path.join(mag_dir, base)
            if os.path.isdir(subfolder):
                # 查找.mcif文件
                mcif_files = glob.glob(os.path.join(subfolder, '*.mcif'))
                if not mcif_files:
                    print(f"警告：在 {subfolder} 中未找到.mcif文件")
                    continue
                
                # 读取第一个找到的.mcif文件
                crystal_system = find_crystal_system(mcif_files[0])
                if crystal_system:
                    results[os.path.basename(pt_path)] = crystal_system
                else:
                    print(f"警告：在 {mcif_files[0]} 中未找到晶系信息")
                    results[os.path.basename(pt_path)] = '未知'
                
                found = True
                break
        
        if not found:
            print(f"错误：未找到 {base} 对应的子文件夹")
            results[os.path.basename(pt_path)] = '未找到'

    # 输出结果
    print("\n统计结果：")
    print("=" * 40)
    for filename, system in results.items():
        print(f"{filename}: {system}")
    csv_path = "crystal_systems.csv"
    with open(csv_path, "w") as f:
        f.write("PT File,Crystal System\n")
        for file, system in results.items():
            f.write(f"{file},{system}\n")
    print(f"\n结果已保存至: {csv_path}")
if __name__ == "__main__":
    main()

