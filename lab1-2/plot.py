import json
import matplotlib.pyplot as plt

# 从 output.json 文件中读取数据
with open('lab1-2\data\output.json', 'r') as f:
    output = json.load(f)

lam_range = output['lam_range']
ndcg_lam = output['ndcg_lam']
mse_lam = output['mse_lam']
k_range = output['k_range']
ndcg_k = output['ndcg_k']
mse_k = output['mse_k']

# 绘制折线图
fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=1000)

# mse_lam 和 ndcg_lam 随 lam 变换的折线图
axs[0].plot(lam_range, mse_lam, label='MSE', marker='o')
axs[0].plot(lam_range, ndcg_lam, label='NDCG', marker='o')
axs[0].set_title('MSE and NDCG vs Lambda')
axs[0].set_xlabel('Lambda')
axs[0].set_ylabel('Score')
axs[0].legend()
axs[0].grid(True)

# mse_k 和 ndcg_k 随 k 变换的折线图
axs[1].plot(k_range, mse_k, label='MSE', marker='o')
axs[1].plot(k_range, ndcg_k, label='NDCG', marker='o')
axs[1].set_title('MSE and NDCG vs K')
axs[1].set_xlabel('K')
axs[1].set_ylabel('Score')
axs[1].legend()
axs[1].grid(True)

plt.savefig('lab1-2/data/output.png')