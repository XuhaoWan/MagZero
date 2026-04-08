import yaml
from collections import namedtuple
from time import time
import os
import torch
from torch_geometric.data import HeteroData


    
def timer(func):
    def wrapper(*args,**kw):
        start = time()
        res = func(*args,**kw)
        duration = time() - start
        print(f"run {func.__name__} in {duration:.1f} seconds")
        return res
    return wrapper
    
def yaml2dict(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    return res

def dict2namedtuple(dic):
    return namedtuple('Config', dic.keys())(**dic)

def load_yaml(path):
    res = yaml2dict(path)
    config = dict2namedtuple(res)
    print(config)
    return config

def load_and_filter_heterodata(folder_path):
    """加载并过滤heterodata数据"""
    dataset = []
    
    # 遍历文件夹中的所有.pt文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.pt'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        data = torch.load(filepath)
        
        
        # 检查是否为HeteroData类型
        if not isinstance(data, HeteroData):            
            continue
            
        # 检查magion节点的特征维度

        magion_x = data['magion'].x
        if magion_x.dim() == 1 or magion_x.size(0) <= 1:
            continue  # 跳过维度<=1的情况
            
        dataset.append(data)
    
    return dataset



def load_and_check_heterodata(folder_path):
    """加载并过滤heterodata数据"""
    dataset = []
    
    # 遍历文件夹中的所有.pt文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.pt'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        data = torch.load(filepath)
        
        
        # 检查是否为HeteroData类型
        if not isinstance(data, HeteroData):            
            continue
            
        # 检查magion节点的特征维度

        #data_dict = convert_hetero_to_global(data)
        #gdata = data_dict['global']
        #magiondata = data_dict['magion']
        magion_x = data['magion'].x
        if magion_x.dim() == 1 or magion_x.size(0) <= 1:
        #if magion_x.size(0) == 2:
            print(filename)
            continue  # 跳过维度<=1的情况
        magion_edge_attr = data['magion', 'magion_edge', 'magion'].edge_attr
        if magion_edge_attr.size(1) <= 1:
            print(filename, magion_edge_attr.size())

        #if gdata.edge_index.numel() == 0:
                  
    
    return None


def load_heterodata_and_comparetime(folder_path):
    convtime = 0
    magzerotime = 0
    
    # 遍历文件夹中的所有.pt文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.pt'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        data = torch.load(filepath)
        
        
        # 检查是否为HeteroData类型
        if not isinstance(data, HeteroData):            
            continue
            
        # 检查magion节点的特征维度

        #data_dict = convert_hetero_to_global(data)
        #gdata = data_dict['global']
        #magiondata = data_dict['magion']
        atom_x = data['atom'].x.size(0)
        magion_x = data['magion'].x.size(0)
        total_num = atom_x + magion_x
        convtime += (magion_x*30*0.02 + atom_x*30*0.005)*(5**magion_x)
        magzerotime += magion_x*30*0.02 + atom_x*30*0.005 + 30*0.01

                  
    print(convtime, magzerotime)
    return None


# def build_edge_centric_graph(data):
#     """
#     Convert raw atom graphs to edge centered graphs.
#     - Edge node features: [original edge properties, atom i features, atom j features, magnetic moment i, magnetic moment j]
#     - Edge graph connectivity: If two edges share an atom, connect the corresponding edge nodes.
#     - Target labeling: relative orientation of the magnetic moments corresponding to each edge [cosθ, sinθ]
#     """
#     # 确保无向边处理
#     edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
#     num_edges = edge_index.size(1)

#     # 步骤1: 构建边节点特征
#     edge_nodes = []
#     edge_targets = []
#     for idx in range(num_edges):
#         i, j = edge_index[:, idx]
        
#         # 边节点特征 = 边属性 + 原子i特征 + 原子j特征 + 磁矩i + 磁矩j
#         # edge_feat = torch.cat([
#         #     edge_attr[idx],          # 原始边属性 [7]
#         #     data.x[i],                # 原子i特征 [164]
#         #     data.x[j],                # 原子j特征 [164]
#         #     data.magmom[i],           # 磁矩i [3]
#         #     data.magmom[j]           # 磁矩j [3]
#         # ], dim=-1)                   # 总维度 7 + 164*2 + 3*2 = 341
#         edge_feat = edge_attr[idx].clone()
#         # 计算磁矩相对方向标签 [cosθ, sinθ]
#         mi, mj = data.magmom[i], data.magmom[j]
#         cos_theta = (mi.dot(mj)) / (torch.norm(mi)*torch.norm(mj) + 1e-6)
#         sin_theta = torch.sqrt(1 - cos_theta**2)
        
#         edge_nodes.append(edge_feat)
#         edge_targets.append(torch.stack([cos_theta, sin_theta]))

#     # 步骤2: 构建边图的连接关系
#     new_edge_index = []
#     new_edge_attr = []
#     for (e1, e2) in combinations(range(num_edges), 2):
        
#         # 获取两条边连接的原子
#         atoms_e1 = set(edge_index[:, e1].tolist())
#         atoms_e2 = set(edge_index[:, e2].tolist())
#         shared_atoms = list(atoms_e1 & atoms_e2)

#         # 为每个共享原子创建一条边
#         for atom in shared_atoms:
#             new_edge_index.append([e1, e2])
#             new_edge_index.append([e2, e1])  # 无向边双向连接
#             new_edge_attr.append(data.x[atom].clone())  # 边属性为共享原子特征
#             new_edge_attr.append(data.x[atom].clone())  # 双向边属性相同

#         # 关键修复：处理孤立边节点
#     if num_edges > 0 and len(new_edge_index) == 0:
#         # 情况1: 所有边节点均为孤立节点（如原图仅有两个节点）
#         for e in range(num_edges):
#             # 添加自环边
#             new_edge_index.append([e, e])
#             # 虚拟边属性：全零或共享原子特征的平均（若无共享原子）
#             if data.x.size(0) > 0:
#                 virtual_attr = data.x.mean(dim=0)  # 平均原子特征
#             else:
#                 virtual_attr = torch.zeros(data.x.size(1))
#             new_edge_attr.append(virtual_attr)

#     # 转换为Tensor
#     edge_node_features = torch.stack(edge_nodes).contiguous()    # [num_edges, 341]
#     edge_graph_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
#     edge_labels = torch.stack(edge_targets)         # [num_edges, 2]

#     # 构建新的Data对象
#     return Data(
#         x=edge_node_features,
#         edge_index=edge_graph_index.to(edge_node_features.device),
#         edge_attr = torch.stack(new_edge_attr).float() if new_edge_attr else torch.empty(0, dtype=torch.float32),
#         y=edge_labels,
#         original_edge_index=edge_index,
#         batch=get_edge_batch(data.batch, edge_index)  # 处理batch信息
#     )

# def to_undirected(edge_index, edge_attr):
#     """将有向边转换为无向边，合并属性"""
#     undirected_edges = {}
#     for i in range(edge_index.size(1)):
#         src, dst = edge_index[:, i].tolist()
        
#         # 关键修复：确保边以升序存储 (src <= dst)
#         if src > dst:
#             src, dst = dst, src  # 交换顺序
        
#         key = (src, dst)
#         if key in undirected_edges:
#             # 如果已存在相同无向边，合并属性（取平均）
#             undirected_edges[key] = (undirected_edges[key] + edge_attr[i]) / 2
#         else:
#             # 否则添加新边
#             undirected_edges[key] = edge_attr[i]
    
#     # 重建无向边索引和属性
#     new_edge_index = []
#     new_edge_attr = []
#     for (src, dst), attr in undirected_edges.items():
#         new_edge_index.append([src, dst])
#         new_edge_attr.append(attr)
    
#     return torch.tensor(new_edge_index).t().contiguous(), torch.stack(new_edge_attr)

# def get_edge_batch(atom_batch, edge_index):
#     """将原子级batch信息转换为边级batch信息"""
#     edge_batch = []
#     for i in range(edge_index.size(1)):
#         src, _ = edge_index[:, i]
#         edge_batch.append(atom_batch[src])
#     return torch.tensor(edge_batch)

def scale_to_negative_one(tensor, batch):
    # 将batch转换为连续索引 (例如 [2,2,5,5] -> [0,0,1,1])
    _, batch_indices = torch.unique(batch, return_inverse=True)
    
    # 计算每个batch的最小值和最大值
    min_vals = torch.zeros(len(torch.unique(batch_indices)), tensor.shape[1], 
                          dtype=tensor.dtype, device=tensor.device)
    max_vals = torch.zeros_like(min_vals)
    
    for i in torch.unique(batch_indices):
        mask = (batch_indices == i)
        batch_data = tensor[mask]
        min_vals[i] = torch.min(batch_data, dim=0).values
        max_vals[i] = torch.max(batch_data, dim=0).values
    
    # 扩展统计量到每个样本
    min_expanded = min_vals[batch_indices]
    max_expanded = max_vals[batch_indices]
    
    epsilon = 1e-8
    scaled = 2 * (tensor - min_expanded) / (max_expanded - min_expanded + epsilon) - 1
    return scaled

def zscore_standardize(tensor, batch):
    _, batch_indices = torch.unique(batch, return_inverse=True)
    
    # 计算每个batch的均值和标准差
    means = torch.zeros(len(torch.unique(batch_indices)), tensor.shape[1], 
                       dtype=tensor.dtype, device=tensor.device)
    stds = torch.zeros_like(means)
    
    for i in torch.unique(batch_indices):
        mask = (batch_indices == i)
        batch_data = tensor[mask]
        means[i] = torch.mean(batch_data, dim=0)
        centered = batch_data - means[i]
        stds[i] = torch.sqrt(torch.mean(centered ** 2, dim=0) + 1e-8)
    
    # 扩展统计量到每个样本
    mean_expanded = means[batch_indices]
    std_expanded = stds[batch_indices]
    
    standardized = (tensor - mean_expanded) / std_expanded
    return standardized

def min_max_normalize(tensor, batch):
    _, batch_indices = torch.unique(batch, return_inverse=True)
    
    # 计算每个batch的最小值和最大值
    min_vals = torch.zeros(len(torch.unique(batch_indices)), tensor.shape[1], 
                          dtype=tensor.dtype, device=tensor.device)
    max_vals = torch.zeros_like(min_vals)
    
    for i in torch.unique(batch_indices):
        mask = (batch_indices == i)
        batch_data = tensor[mask]
        min_vals[i] = torch.min(batch_data, dim=0).values
        max_vals[i] = torch.max(batch_data, dim=0).values
    
    # 扩展统计量到每个样本
    min_expanded = min_vals[batch_indices]
    max_expanded = max_vals[batch_indices]
    
    epsilon = 1e-8
    normalized = (tensor - min_expanded) / (max_expanded - min_expanded + epsilon)
    return normalized


if __name__ == "__main__":
    data_folder = r'D:\Ai Spin\data\datawith9' 
    #load_and_check_heterodata(data_folder)
    load_heterodata_and_comparetime(data_folder)
    #with open ("ds.dat", 'w') as f:
    #    print(ds, file=f)