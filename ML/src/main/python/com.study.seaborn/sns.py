import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
 画直方图
"""
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col='time')
g.map(plt.hist, "tip")

"""
 画散点图
"""
g = sns.FacetGrid(tips,col='sex',hue='smoker') # 设置参数hue，分类显示
g.map(plt.scatter,"total_bill","tip", alpha=0.7) # 参数alpha，设置点的大小
g.add_legend()  # 加注释