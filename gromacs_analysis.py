# aiida packages
from aiida.orm import SinglefileData, List
from aiida.engine import calcfunction

# plotting packages
import re
import numpy as np
#from IPython.display import display, IFrame
import os

import matplotlib.pyplot as plt
# Set global figure properties using rcParams
plt.rcParams.update({
    'figure.figsize': (8, 5),       # Set figure size
    'axes.titlesize': 16,           # Title font size
    'axes.labelsize': 14,           # X & Y label size
    'xtick.labelsize': 12,          # X-tick label size
    'ytick.labelsize': 12,          # Y-tick label size
    'xtick.major.size': 6,          # Major tick size
    'ytick.major.size': 6,          # Major tick size
    'xtick.minor.size': 4,          # Minor tick size
    'ytick.minor.size': 4,          # Minor tick size
    'xtick.major.width': 1.5,       # Major tick width
    'ytick.major.width': 1.5,       # Major tick width
    'xtick.minor.width': 1.0,       # Minor tick width
    'ytick.minor.width': 1.0,       # Minor tick width
    'xtick.direction': 'in',        # Tick direction (inward)
    'ytick.direction': 'in',        # Tick direction (inward)
    'axes.grid': True,              # Enable grid by default
    'grid.linestyle': '--',         # Dashed grid lines
    'grid.alpha': 0.7,              # Grid transparency
    'legend.fontsize': 12,          # Legend font size
    'axes.spines.top': True,       # Remove top border
    'axes.spines.right': True,     # Remove right border
    'font.family': 'serif',         # Font family
    'font.size': 13,                # Default font size
    'axes.edgecolor': 'black',      # Axis edge color
    'axes.linewidth': 1.2,          # Axis border thickness
})

@calcfunction
def create_time_plot(xvg: SinglefileData) -> List:
    xvg_lines = xvg.get_content().split('\n')

    head_lines = [line for line in xvg_lines if line.startswith('@')]
    plt.title('GROMACS Properties')
    
    xaxis_label_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if line.split()[1] == 'xaxis']
    xaxis_label_list = [word for word in xaxis_label_str[0].split(', ')]
    plt.xlabel(f'{xaxis_label_list[0]}')
    
    yaxis_label_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if line.split()[1] == 'yaxis']
    yaxis_label_list = [word for word in yaxis_label_str[0].split(', ')]

    yaxis_legend_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if len(line.split()) > 2 and line.split()[2] == 'legend']

    data = np.loadtxt(xvg_lines, comments=['#', '@']).T
    
    if len(yaxis_label_list) != len(data) - 1:
        raise ValueError('Data Error.')
        
    plot = List([])
    for iplot in range(len(data)-1):
        #print('plotting -> ', iplot)
        plt.ylabel(f'{yaxis_legend_str[iplot]} {yaxis_label_list[iplot]}')
        plt.plot(data[0], data[iplot+1])
        plt.tight_layout()
        plt.savefig(f'{os.getcwd()}/time_plot_{iplot}.png', format='png', bbox_inches='tight', dpi=150)
        plt.clf()
        plot.append(f'{os.getcwd()}/time_plot_{iplot}.png')
    return plot