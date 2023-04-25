import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors

cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgrey","crimson"])    

cell_connected_to_lobules_boolean_excluding_lobule2andNonecells = pd.read_csv ('Cell_connected_to_lobules_boolean_excluding_lobule2andNonecells.csv')

sn.clustermap(cell_connected_to_lobules_boolean_excluding_lobule2andNonecells,figsize=(6,6), method='average', annot_kws={"fontsize":16}, metric='hamming', col_cluster=True, row_cluster=False, annot=False, cmap=cmap2)
plt.show()







