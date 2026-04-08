import math
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict, deque

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

def determine_magnetic_moments(cos_list):
    n = int((1 + math.sqrt(1 + 8 * len(cos_list))) // 2)
    cos_matrix = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if idx < len(cos_list):
                cos_matrix[i][j] = cos_matrix[j][i] = float(cos_list[idx])
                idx += 1

    # Step 1: 精确分组（基于图遍历）
    graph = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            if abs(cos_matrix[i][j]) >= 0.85:
                graph[i].append(j)
                graph[j].append(i)

    visited = set()
    groups = []
    for node in range(n):
        if node not in visited:
            queue = deque([node])
            visited.add(node)
            current_group = []
            while queue:
                u = queue.popleft()
                current_group.append(u)
                for v in graph[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            groups.append(sorted(current_group))
    print(groups)
    # Step 2: 计算组间平均余弦值
    group_pairs = defaultdict(list)
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            for a in groups[i]:
                for b in groups[j]:
                    #if a < b:
                    group_pairs[(i,j)].append(abs(cos_matrix[a][b]))

    avg_cos = {}
    for (i,j), values in group_pairs.items():
        avg_cos[(i,j)] = np.mean(values) if values else 0.0
    print(group_pairs)

    # Step 3: 符号分配（基于BFS）
    sign = np.ones(n, dtype=int)
    for group in groups:
        if 0 in group:
            root = 0
            break
    else:
        root = groups[0][0]

    visited = set()
    queue = deque([root])
    visited.add(root)
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                s = 1 if cos_matrix[u][v] > 0 else -1
                sign[v] = sign[u] * s
                visited.add(v)
                queue.append(v)

    # Step 4: 方向优化（考虑所有原子对）
    group_dirs = {}
    fixed_group = next(i for i, g in enumerate(groups) if 0 in g)
    group_dirs[fixed_group] = np.array([0.0, 0.0, 1.0], dtype=float)

    params = []
    bounds = []
    for idx, group in enumerate(groups):
        if idx == fixed_group:
            continue
        theta = np.arccos(np.clip(np.random.uniform(-1,1), -1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        params.extend([theta, phi])
        bounds.extend([(0, np.pi), (0, 2*np.pi)])

    def objective(x):
        error = 0.0
        dirs = {fixed_group: np.array([0,0,1], dtype=float)}
        ptr = 0
        for idx, group in enumerate(groups):
            if idx == fixed_group:
                continue
            theta, phi = x[ptr], x[ptr+1]
            ptr += 2
            dx = np.sin(theta)*np.cos(phi)
            dy = np.sin(theta)*np.sin(phi)
            dz = np.cos(theta)
            dirs[idx] = np.array([dx, dy, dz], dtype=float)

        # 计算所有原子对的误差
        for i in range(n):
            gi = next(idx for idx, g in enumerate(groups) if i in g)
            di = dirs[gi] * sign[i]
            for j in range(i+1, n):
                gj = next(idx for idx, g in enumerate(groups) if j in g)
                dj = dirs[gj] * sign[j]
                pred = np.dot(di, dj)
                actual = cos_matrix[i][j]
                error += (pred - actual)**2
        return error

    if params:
        res = minimize(objective, params, method='L-BFGS-B', bounds=bounds)
        opt_params = res.x
    else:
        opt_params = []

    # 重建最终方向
    dirs = {fixed_group: np.array([0,0,1], dtype=float)}
    ptr = 0
    for idx, group in enumerate(groups):
        if idx == fixed_group:
            continue
        if opt_params.any():
            theta, phi = opt_params[ptr], opt_params[ptr+1]
            ptr += 2
        else:
            theta, phi = np.pi/2, 0
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)
        dz = np.cos(theta)
        dirs[idx] = np.array([dx, dy, dz], dtype=float)

    # 生成原子方向
    atom_dirs = []
    for atom in range(n):
        group_idx = next(i for i, g in enumerate(groups) if atom in g)
        direction = dirs[group_idx].copy()
        direction *= sign[atom]
        direction /= np.linalg.norm(direction)
        atom_dirs.append(tuple(direction))

    # 误差分析
    error_count = 0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            pred = np.dot(atom_dirs[i], atom_dirs[j])
            if abs(pred - cos_matrix[i][j]) > 0.2:
                error_count += 1
            total += 1
    error_ratio = error_count / total if total > 0 else 0

    return atom_dirs, error_ratio > 0.2, error_ratio, avg_cos

# 示例使用
if __name__ == "__main__":
#     # 测试案例1: 4个原子
#     n = 4
#     cos_list = [
#         0.7706,   
#         -0.9281,   
#         -0.8642, 
#         -0.8642,  
#         -0.9281,   
#         0.9074  
#     ]
    
#     directions, is_unsatisfied, error, avg_cos = determine_magnetic_moments(cos_list)
    
#     print("最优磁矩方向：")
#     for i, dir in enumerate(directions):
#         print(f"原子{i}: ({dir[0]:.4f}, {dir[1]:.4f}, {dir[2]:.4f})")
#     print(f"总误差：{error:.4f}")

#     # 测试案例2: 6
#     n = 6
#     cos_list = [
# 0.47, -0.5, -0.91, -0.49, 0.48, 0.48, -0.48, -0.77, -0.38, 0.48, -0.3586, -0.9445, 0.4551, -0.47, 0.47
#     ]
    
#     directions, is_unsatisfied, error, avg_cos = determine_magnetic_moments(cos_list)
#     print("\n简单共线案例：")
#     for i, dir in enumerate(directions):
#         print(f"原子{i}: ({dir[0]:.4f}, {dir[1]:.4f}, {dir[2]:.4f})")
#     print(f"总误差：{error:.4f}")
#     print("\n共线组间平均余弦值：")
#     for (g1, g2), val in avg_cos.items():
#         print(f"组{g1}与组{g2}: {val:.4f}")

    import numpy as np
    from scipy.optimize import minimize

    # 给定的目标余弦值
    target = {
        (0, 1): 0.4750,
        (0, 2): 0.4825,
        (0, 3): 0.4726,
        (1, 2): 0.4300,
        (1, 3): 0.7700,
        (2, 3): 0.4143
    }

    # 组0固定为 (0,0,1)
    v0 = np.array([0.0, 0.0, 1.0])
    # 参数化：对每个向量使用球坐标 (phi, theta)
    # 对组0, phi=0；其他组的变量为 phi_i 和 theta_i, i=1,2,3, 共6个参数
    # 优化变量顺序：[phi1, theta1, phi2, theta2, phi3, theta3]
    # 单位向量转换函数
    def spherical_to_cartesian(phi, theta):
        return np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

    # 构造所有向量
    def get_vectors(x):
        # x = [phi1, theta1, phi2, theta2, phi3, theta3]
        v1 = spherical_to_cartesian(x[0], x[1])
        v2 = spherical_to_cartesian(x[2], x[3])
        v3 = spherical_to_cartesian(x[4], x[5])
        return [v0, v1, v2, v3]

    # 共面惩罚项：我们希望所有向量的y分量尽量接近0 (即在 x-z 平面)
    def coplanar_penalty(vectors):
        penalty = 0.0
        for v in vectors:
            penalty += v[1]**2  # y 分量平方
        return penalty

    # 目标函数：余弦误差 + lambda * 共面惩罚
    def objective(x, lam=10.0):
        vectors = get_vectors(x)
        error = 0.0
        # 遍历所有组合对
        indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for i, j in indices:
            dot_val = np.dot(vectors[i], vectors[j])
            key = (min(i,j), max(i,j))
            # 对组0已经固定，其他对对应target中的值
            error += (dot_val - target[key])**2
        # 加入共面惩罚项
        error += lam * coplanar_penalty(vectors)
        return error

    # 初始猜测：利用组0与其他组的余弦值估计phi, 并随机theta
    # 例如，对于组i，与组0的内积为 cos(phi_i) 近似等于目标
    def initial_guess():
        init = []
        for i in range(1, 4):
            # 由于 cos(phi) = dot(v0, vi) = cos(phi)，则 phi = arccos(target)
            phi = np.arccos(target[(0, i)])
            theta = np.random.uniform(0, 2*np.pi)
            init.extend([phi, theta])
        return np.array(init)

    init_x = initial_guess()

    # 调整惩罚项权重 lam 以平衡余弦拟合和共面要求
    lam = 10

    res = minimize(objective, init_x, args=(lam,), method='l-bfgs-b', options={'disp': True})

    print("优化结果：")
    print("参数:", res.x)
    vectors = get_vectors(res.x)
    for i, v in enumerate(vectors):
        print("组{} 向量: {}".format(i, v))
        
    print("\n组间余弦值：")
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, j in pairs:
        cos_val = np.dot(vectors[i], vectors[j])
        print("组{}与组{}: {:.4f}".format(i, j, cos_val))