import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 网格参数
R = 0.01                  # 外半径 [m]
r_inner = 0.001       # 内半径 [m]
r_outer = R               # 外半径 [m]
Nr = 50                   # 径向网格数
Ntheta = 200              # 圆周向网格数

# 坐标生成：结构化极坐标网格
r = np.linspace(r_inner, r_outer, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta + 1, endpoint=True)
R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')

# 转换为笛卡尔坐标
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# 展平输出坐标点 (结构化拓扑仍保留)
x_flat = X.flatten()
y_flat = Y.flatten()

# 保存为 CSV 文件（笛卡尔坐标）
df = pd.DataFrame({'x': x_flat, 'y': y_flat})
df.to_csv("O_grid_structured.csv", index=False)

# 可选：保存结构化拓扑为 npz
np.savez("O_grid_structured.npz", X=X, Y=Y, r=r, theta=theta)
print(f"O型结构化网格生成完成，共 {Nr} × {Ntheta} 个点")

plt.figure(figsize=(6,6))
plt.plot(X, Y, 'k-', linewidth=0.5)  # 径向线
plt.plot(X.T, Y.T, 'k-', linewidth=0.5)  # 圆周线
plt.gca().set_aspect('equal')
plt.title("O型结构化网格")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.show()