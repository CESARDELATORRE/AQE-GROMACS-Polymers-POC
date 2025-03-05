# aiida packages
from aiida.orm import SinglefileData, List, Int, Dict, Str
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
def get_average_property(result: SinglefileData, property_list: List) -> List:
    result_lines = result.get_content().split('\n')

    property_line_list = [line for line in result_lines if len(line.split()) > 0 and line.split()[0] in property_list.get_list()]
    
    average_property_list = List([])
    for line in property_line_list:
        wordlist = line.split()
        average_property_list.append([wordlist[0], wordlist[1], wordlist[5]])
    return average_property_list

@calcfunction
def calc_average_property(xvg: SinglefileData) -> List:
    xvg_lines = xvg.get_content().split('\n')

    head_lines = [line for line in xvg_lines if line.startswith('@')]
    
    data = np.loadtxt(xvg_lines, comments=['#', '@']).T

    yaxis_legend_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if len(line.split()) > 2 and line.split()[2] == 'legend']

    yaxis_label_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if line.split()[1] == 'yaxis']
    yaxis_label_list = [word for word in yaxis_label_str[0].split(', ')]

    n_data = len(data[0])
    last_n_data = max(1, int(n_data * 0.5))

    average_prop = List([])
    for iprop in range(len(data)-1):
        last_data = data[iprop+1][-last_n_data:]
        average_prop.append([np.mean(last_data, axis=0), 
        np.std(last_data, axis=0),
        yaxis_label_list[iprop],
        yaxis_legend_str[iprop]])
    return average_prop

@calcfunction
def create_time_plot(xvg: SinglefileData) -> List:
    xvg_lines = xvg.get_content().split('\n')

    head_lines = [line for line in xvg_lines if line.startswith('@')]
    
    xaxis_label_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if line.split()[1] == 'xaxis']
    xaxis_label_list = [word for word in xaxis_label_str[0].split(', ')]
    
    yaxis_label_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if line.split()[1] == 'yaxis']
    yaxis_label_list = [word for word in yaxis_label_str[0].split(', ')]

    yaxis_legend_str = [re.search(r'"(.*?)"', line).group(1) for line in head_lines if len(line.split()) > 2 and line.split()[2] == 'legend']

    data = np.loadtxt(xvg_lines, comments=['#', '@']).T
    
    if len(yaxis_label_list) != len(data) - 1:
        raise ValueError('Data Error.')
        
    plot = List([])
    for iplot in range(len(data)-1):
        print('plotting -> ', iplot)
        plt.title('GROMACS Properties')
        plt.xlabel(f'{xaxis_label_list[0]}')
        plt.ylabel(f'{yaxis_legend_str[iplot]} {yaxis_label_list[iplot]}')
        plt.plot(data[0], data[iplot+1])
        plt.tight_layout()
        plt.savefig(f'{os.getcwd()}/time_plot_{iplot}.png', format='png', bbox_inches='tight', dpi=150)
        plt.clf()
        plot.append(f'{os.getcwd()}/time_plot_{iplot}.png')
    return plot

def linear_regression(x, y, sigma):
    """
    Function to perform linear regression with error estimation using chi-square.
    Args:
        x: Independent variable (temperature)
        y: Dependent variable (density)
        sigma: Uncertainty (standard deviation) of the y values
    Returns:
        slope: The slope of the linear fit
        intercept: The intercept of the linear fit
        fit_line: The fitted line (predicted values for y)
        chi_square: Chi-square statistic
        reduced_chi_square: Reduced chi-square statistic
        slope_error: Standard error of the slope
        intercept_error: Standard error of the intercept
    """
    # Calculate the weights
    w = 1 / (sigma * sigma)

    # Weighted sums
    S_w = np.sum(w)
    S_wx = np.sum(w * x)
    S_wy = np.sum(w * y)
    S_wxy = np.sum(w * x * y)
    S_wx2 = np.sum(w * x * x)

    # Slope and intercept
    m = (S_w * S_wxy - S_wx * S_wy) / (S_w * S_wx2 - S_wx * S_wx)
    b = (S_wx2 * S_wy - S_wx * S_wxy) / (S_w * S_wx2 - S_wx * S_wx)
    
    slope = m
    intercept = b
    
    # Predicted values from the linear fit
    fit_line = slope * x + intercept
    
    # Residuals (errors in the data points)
    residuals = y - fit_line
    
    # Chi-square statistic
    chi_square = np.sum((residuals / sigma) * (residuals / sigma))
    
    return slope, intercept, fit_line, chi_square

@calcfunction
def create_tg_plot(thermo_T_list: List, average_property_list: List, icol: Int) -> Str:

    average_list = []
    std_list = []
    for average_prop in average_property_list:
        average_list.append(average_prop[icol.value][0])
        std_list.append(average_prop[icol.value][1])

    average_list = np.array(average_list)
    std_list = np.array(std_list)
    thermo_T_list = np.array(thermo_T_list)

    best_n = None
    best_chi_square = float('inf')  # Start with a very high value
    best_fit_params = None
    best_fit_lines = None
    
    for n in range(2, len(thermo_T_list)-1):
        # Fit the first n points
        x_fit1 = thermo_T_list[:n]
        y_fit1 = average_list[:n]
        sigma_fit1 = std_list[:n]
        m1, b1, fit_line1, chi1 = linear_regression(x_fit1, y_fit1, sigma_fit1)
        
        # Fit the remaining points (len(thermo_T_list)-n)
        x_fit2 = thermo_T_list[n-1:]
        y_fit2 = average_list[n-1:]
        sigma_fit2 = std_list[n-1:]
        m2, b2, fit_line2, chi2 = linear_regression(x_fit2, y_fit2, sigma_fit2)
        
        # Compute the total chi-square for this value of n
        total_chi_square = chi1 + chi2

        # If this is the best chi-square, update the best n and store the parameters
        if total_chi_square < best_chi_square:
            best_chi_square = total_chi_square
            best_n = n
            best_fit_params = ((m1, b1), (m2, b2))
            best_fit_lines = (fit_line1, fit_line2)

    # Output the best n and corresponding chi-square value
    print(f"Best value of n: {best_n}")
    print(f"Best total \u03C7²: {best_chi_square}")

    # Now plot the best fit lines
    m1, b1 = best_fit_params[0]
    m2, b2 = best_fit_params[1]
    
    # Create full-range fitted lines
    full_range_fit1 = m1 * thermo_T_list + b1
    full_range_fit2 = m2 * thermo_T_list + b2

    # Calculate the intersection point (Tg)
    Tg = (b2 - b1) / (m1 - m2)
    print(f'Calculated Tg (intersection of the two lines): {Tg:.2f} K')

    plt.title('Density vs Temperature')
    plt.errorbar(thermo_T_list, average_list, yerr = std_list, fmt = 'o', color = "black",
                 label = None, ecolor = 'red', elinewidth = 2, capsize = 4, capthick = 2)
    
    plt.plot(thermo_T_list, full_range_fit1, label = "First few points fit",
             linestyle = "--", color = "green")

    plt.plot(thermo_T_list, full_range_fit2, label = "Last few points fit",
             linestyle = "--", color = "yellow")
    
    plt.axvline(x = Tg, color = 'blue', linestyle = ':', label = f'Tg = {Tg:.2f} K')
    
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average Density (kg/m³)")
    plt.legend(fontsize=12)
    plt.savefig(f'{os.getcwd()}/tg_plot.png', format = 'png', bbox_inches = 'tight', dpi = 150)
    plt.clf()
    return(Str(f'{os.getcwd()}/tg_plot.png'))