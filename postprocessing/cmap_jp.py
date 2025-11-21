import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# shifted hot colormap to better capture details
def new_cmap(colors, nodes, name:str=None):
    nodes[...] -= nodes[0]
    nodes /= nodes[-1]
    # print(nodes)

    if name:
        try:
            my_cmap = LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))
            mpl.colormaps.register(cmap=my_cmap)
        except:
            my_cmap = LinearSegmentedColormap.from_list("dummy", list(zip(nodes, colors)))
            print("Already defined")

# shifted hot colormap to better capture details
name = "jp_temperature"
colors = ["white", "darkblue", "darkred", "orange", "white"]
nodes = np.array([10.6, 11.7, 12., 13.5, 15.6])
new_cmap(colors, nodes, name)

# shifted hot colormap to better capture details
name = "jp_linear"
colors = ["white", "darkblue", "darkred", "orange", "white"]
nodes = np.array([0., 1., 2., 3., 4.])
new_cmap(colors, nodes, name)

## hex-codes
# darkblue : "#00008B"
# darkred : "#8B0000"
# orange   : "#FFA500" 