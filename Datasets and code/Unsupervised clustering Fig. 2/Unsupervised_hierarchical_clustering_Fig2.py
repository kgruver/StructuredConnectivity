import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors

cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgrey","crimson"])    

cell_connected_to_lobules_demo = pd.read_csv ('Cell_connected_to_lobules_boolean_excluding_lobule2andNonecells.csv')

sn.clustermap(cell_connected_to_lobules,figsize=(6,6), tree_kws=dict(linewidths=3, color="black"), method='average', annot_kws={"fontsize":16}, metric='hamming', col_cluster=True, row_cluster=False, annot=False, linewidths=1, cmap=cmap2)
plt.show()







