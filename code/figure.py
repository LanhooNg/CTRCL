import matplotlib.pyplot as plt
import numpy as np

# 提供的图像数据
categories = ['MobileNetV2', 'ResNet-50', 'MiT-B0', 'MiT-B2']
baseline = [65, 80, 85, 85]
baseline_pcp = [90, 90, 90, 90]

# 条形宽度
bar_width = 0.15

# X轴位置
index = np.array([0.25, 0.55, 1, 1.3])

# 绘制条形图
fig, ax = plt.subplots()
bar1 = ax.bar(index - bar_width/2, baseline, bar_width, label='baseline')
bar2 = ax.bar(index + bar_width/2, baseline_pcp, bar_width, label='baseline+PCP')

# 添加标签和标题
ax.set_xlabel('Category')
ax.set_ylabel('Dice')
ax.set_xticks(index)
ax.set_xticklabels(categories)
ax.legend(loc='lower right')  # 调整图例位置
ax.legend()

# 保存图形
plt.savefig('../dataset/Synapse/test_vol_h5_sne/picture_1.png', dpi=300)

# Displaying the plot
plt.show()
