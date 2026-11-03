import matplotlib.pyplot as plt
import numpy as np

# ========== 1. 数据准备 ==========
metrics = ['ARI', 'Homogeneity', 'Completeness', 'V-Measure']

scores = {
    'MSDiffAug': [0.4952, 0.8791, 0.8857, 0.8813],
    'MSBERT': [0.4379, 0.8827, 0.8878, 0.8794],
    'Spec2Vec': [0.2589, 0.7792, 0.8170, 0.7914],
    'MS/MS Bin': [0.2267, 0.7329, 0.7662, 0.7437]
}

stds = {
    'MSDiffAug': [0.0406, 0.0054, 0.0083, 0.0057],
    'MSBERT': [0.0284, 0.0021, 0.0123, 0.0087],
    'Spec2Vec': [0.0314, 0.0091, 0.0126, 0.0101],
    'MS/MS Bin': [0.0141, 0.0100, 0.0133, 0.0109]
}

# ========== 2. 顶刊配色方案 (填充色 & 对应的深色边框) ==========
# 填充色 (你提供的配色)
face_colors = ['#B5AED5', '#B2E6FD', '#B8D2CC', '#E8B2A7']
# 对应加深的边框色 (手动微调，更具高级感)
edge_colors = ['#8E84B8', '#7BC8E8', '#8BA8A1', '#C68678']
# 误差棒颜色 (在边框色基础上再加深一档)
err_colors = ['#635C81', '#568CA2', '#617671', '#8B5E54']

# ========== 3. 图形属性设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.0

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics))
# width 设为 0.18，间距通过偏移量控制，使同一组内产生微小缝隙
width = 0.18
# 组内间距微调系数
gap = 0.02

# ========== 4. 绘制条形图 ==========
# 计算偏移量：每个柱子之间保留 gap 距离
rects1 = ax.bar(x - 1.5*(width + gap), scores['MSDiffAug'], width, label='MSDiffAug',
                color=face_colors[0], edgecolor=edge_colors[0], lw=1.2,
                yerr=stds['MSDiffAug'], error_kw=dict(ecolor=err_colors[0], capsize=2.5, lw=1.0))
rects2 = ax.bar(x - 0.5*(width + gap), scores['MSBERT'], width, label='MSBERT',
                color=face_colors[1], edgecolor=edge_colors[1], lw=1.2,
                yerr=stds['MSBERT'], error_kw=dict(ecolor=err_colors[1], capsize=2.5, lw=1.0))
rects3 = ax.bar(x + 0.5*(width + gap), scores['Spec2Vec'], width, label='Spec2Vec',
                color=face_colors[2], edgecolor=edge_colors[2], lw=1.2,
                yerr=stds['Spec2Vec'], error_kw=dict(ecolor=err_colors[2], capsize=2.5, lw=1.0))
rects4 = ax.bar(x + 1.5*(width + gap), scores['MS/MS Bin'], width, label='MS/MS Bin',
                color=face_colors[3], edgecolor=edge_colors[3], lw=1.2,
                yerr=stds['MS/MS Bin'], error_kw=dict(ecolor=err_colors[3], capsize=2.5, lw=1.0))

# ========== 5. 坐标轴与细节微调 ==========
# 1. 字体大小设置
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# 2. 移除上方和右侧边框 (Modern scholarly style)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3. 设置刻度线向内
ax.tick_params(direction='in', top=False, right=False)

# 4. 设置范围
ax.set_ylim(0, 1.05)

# 5. 图例设置 (字体14，无阴影，边框精细)
ax.legend(fontsize=14, loc='upper left', frameon=True,
          edgecolor='black', fancybox=False, framealpha=1.0)

# 6. 背景辅助网格 (仅纵向)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#999999')
ax.set_axisbelow(True) # 确保网格在柱子后面

# ========== 6. 保存与输出 ==========
plt.tight_layout()
# 建议保存为 PDF 或 DPI 600 的 PNG 以满足投稿要求
plt.savefig('Clustering.png', dpi=600, bbox_inches='tight')
plt.savefig('Clustering.pdf', dpi=600, bbox_inches='tight')
# plt.savefig('similarity_vs_recall_publication.svg', format='svg', bbox_inches='tight')
plt.show()