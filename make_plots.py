from utils import read_data
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from ptable_trends import ptable_plotter
import os
import numpy as np

import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

import datashader as ds
from datashader.mpl_ext import dsshow

import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


#Utils
def dircheck():
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('plots/defects'):
        os.mkdir('plots/defects')
    if not os.path.exists('plots/parity'):
        os.mkdir('plots/parity')
    if not os.path.exists('plots/ptables'):
        os.mkdir('plots/ptables')
    if not os.path.exists('plots/delta_ptables'):
        os.mkdir('plots/delta_ptables')
    if not os.path.exists('plots/delta_orbital_line_plots'):
        os.mkdir('plots/delta_orbital_line_plots')
    if not os.path.exists('plots/saturation'):
        os.mkdir('plots/saturation')

def save_parity_plot(x, y, title, xlabel, ylabel, fname, colors=None):
    #     if xlim_bottom is not None:
    #       plt.xlim(xlim_bottom, max(x))

    plt.plot(x,y, 'r.', alpha=0.1) # x vs y
    plt.plot(x,x, 'k-') # identity line

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    
    plt.savefig(f'plots/{fname}.png')
    
    # clear plot
    plt.clf()

def save_hist_plot(x, y, title, xlabel, ylabel, fname, bins=(200,200)):
    # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    
    #     if xlim_bottom is not None:
    #       plt.xlim(xlim_bottom, max(x))

    plt.hist2d(x, y, bins, cmap=plt.cm.jet)
    plt.colorbar()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    
    plt.savefig(f'plots/{fname}.png')
    
    # clear plot
    plt.clf()

def save_scatter_plot(x, y, title, xlabel, ylabel, fname, rmse_unit=None):
    def using_datashader(ax, x, y):
        df = DataFrame(dict(x=x, y=y))
        dsartist = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            vmin=0,
            vmax=15,
            norm="linear",
            aspect="equal",
            ax=ax,
        )
        cbar = plt.colorbar(dsartist)
        cbar.set_label("Counts", rotation=270, labelpad=20)

    fig, ax = plt.subplots()

    using_datashader(ax, x, y)
    ax.plot([min(x), max(x)], [min(x), (max(x))], linestyle='--', dashes=(5, 10), color='black') # identity line length of 5, space of 10 
    
    # Make text
    if rmse_unit is not None:
        rmse = np.sqrt(np.mean((np.array(x) - np.array(y))**2))
        ax.text(0.05, 0.95, f'RMSD: {rmse:.4f} {rmse_unit}', transform=ax.transAxes, fontsize=20, verticalalignment='top')
    

    # fix ticks
    plt.locator_params(axis='both', nbins=5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('scaled')

    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f'plots/{fname}.png')
    
    # clear plot
    plt.clf()


def save_inset_plot(x, y, title, xlabel, ylabel, fname, rmse_unit=None, zoom=None):
    def using_datashader(ax, x, y, plot_cbar=True):
        df = DataFrame(dict(x=x, y=y))
        dsartist = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            vmin=0,
            vmax=15,
            norm="linear",
            aspect="equal",
            ax=ax,
        )
        if plot_cbar:
            cbar = plt.colorbar(dsartist, shrink=0.8)
            cbar.set_label("Counts", rotation=270, labelpad=20)

    fig, ax = plt.subplots()

    # Plot main data
    using_datashader(ax, x, y)
    ax.plot([min(x), max(x)], [min(x), (max(x))], linestyle='--', dashes=(5, 10), color='black') # identity line length of 5, space of 10 

    # cut_coords = (200,250)

    ##### MAKE AND PLOT INSET
    axin = ax.inset_axes([0.5, 0.02, 0.4, 0.4])
    # Plot the data on the inset axis and zoom in on the important part
    using_datashader(axin, x, y, plot_cbar=False)
    axin.set_xlim(*zoom)
    axin.set_ylim(*zoom)
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)
    # Remove ticks
    axin.xaxis.set_ticks([])
    axin.yaxis.set_ticks([])
    
    # Make text
    if rmse_unit is not None:
        rmse = np.sqrt(np.mean((np.array(x) - np.array(y))**2))
        ax.text(0.05, 0.95, f'RMSD: {rmse:.4f} {rmse_unit}', transform=ax.transAxes, fontsize=20, verticalalignment='top')

    # fix ticks
    plt.locator_params(axis='both', nbins=5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('scaled')

    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f'plots/{fname}.png')
    
    # clear plot
    plt.clf()



# Parity plots
def make_parity_plots(font_size=None):
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})
    
    #kvrh
    pred_kvrh, actual_kvrh = read_data('res/kvrh_parity.pickle')
    save_scatter_plot(actual_kvrh, pred_kvrh, None, 'Measured $K_{VRH}$ (GPa)', 'Predicted $K_{VRH}$ (GPa)', f'parity/kvrh_parity_{font_size}')

    #gvrh
    pred_gvrh, actual_gvrh = read_data('res/gvrh_parity.pickle')
    loged_pred_gvrh, loged_actual_gvrh = np.log10(pred_gvrh), np.log10(actual_gvrh)
    f = lambda x: np.isnan(x) or np.isinf(x)
    fil = filter(lambda x: not(f(x[0]) or f(x[1])), zip(loged_pred_gvrh, loged_actual_gvrh))
    loged_pred_gvrh, loged_actual_gvrh = list(zip(*fil))
    
    save_scatter_plot(loged_actual_gvrh, loged_pred_gvrh, None, 'Measured $G_{VRH}$ log(GPa)', 'Predicted $G_{VRH}$ log(GPa)', f'parity/gvrh_parity_{font_size}')

    #eform
    pred_eform, actual_eform = read_data('res/eform_parity.pickle')
    # remove outliers
    pure_actual_eform, pure_pred_eform = [], []
    for i, x in enumerate(actual_eform):
        if x >= -5: 
            pure_actual_eform.append(x)
            pure_pred_eform.append(pred_eform[i])
    save_scatter_plot(pure_actual_eform, pure_pred_eform, None, 'Measured $E_{form}$ (eV/atom)', 'Predicted $E_{form}$ (eV/atom)', f'parity/eform_parity_{font_size}')


# Defects
def make_kvrh_defect_plots():
    #kvrh
    defect_data = read_data('res/kvrh_defect_3.pickle')
    baseline_data, _ = read_data('res/kvrh_parity.pickle')
    inset_zooms = {'H':(50,100), 'Mn':(150,200), 'Rb':(200,250)}
    
    for k in defect_data:
        save_inset_plot(baseline_data, defect_data[k], f'$K_{{VRH}}$ {k} defect', 'Predicted $K_{VRH}$ (GPa)', 'Predicted $K_{VRH}$ (GPa) with defect', f'defects/kvrh_{k}_defect_3', rmse_unit="(GPa)", zoom=inset_zooms[k])

def make_gvrh_defect_plots():
    #gvrh
    defect_data = read_data('res/gvrh_defect_3.pickle')
    baseline_data, _ = read_data('res/gvrh_parity.pickle')
    inset_zooms = {'H':(1.7,2.2), 'Mn':(1.75,2.25), 'Rb':(1.75,2.25)}
    
    for k in defect_data:
        save_inset_plot(np.log10(baseline_data), np.log10(defect_data[k]), f'$G_{{VRH}}$ {k} defect', 'Predicted $G_{VRH}$ log(GPa)', 'Predicted $G_{VRH}$ log(GPa) with defect', f'defects/gvrh_{k}_defect_3', rmse_unit="log(GPa)", zoom=inset_zooms[k])

        # GEN NO LOGED
        save_inset_plot(baseline_data, defect_data[k], f'$G_{{VRH}}$ {k} defect', 'Predicted $G_{VRH}$ (GPa)', 'Predicted $G_{VRH}$ (GPa) with defect', f'defects/gvrh_{k}_defect_3_NOLOG', rmse_unit="(GPa)", zoom=inset_zooms[k])
        

def make_eform_defect_plots():
    #eform
    defect_data = read_data('res/eform_defect_3_20k.pickle')
    baseline_data, _ = read_data('res/eform_parity.pickle')
    inset_zooms = {'H':(-0.25,0.5), 'Mn':(0.0,0.75), 'Rb':(-0.25,0.5)}
    
    for k in defect_data:
        save_inset_plot(baseline_data[:len(defect_data[k])], defect_data[k], f'$E_{{form}}$ {k} defect', 'Predicted $E_{form}$ (eV/atom)', 'Predicted $E_{form}$ (eV/atom) with defect', f'defects/eform_{k}_defect_3', rmse_unit="(eV/atom)", zoom=inset_zooms[k])

# Ptables
def make_ptable_plots(delta=True):
    if delta:
        data = read_data('res/ptable_delta_defects.pickle')
        out_dir = 'delta_ptables'
        prefix = 'deltaptable'
    else:
        data = read_data('res/ptable_defect_3.pickle')
        out_dir = 'ptables'
        prefix = 'ptable'

    for enlargement in data:
        d = data[enlargement]
        for struct_name in d:
            for prop in d[struct_name]:

                # Skip when key is elem
                if prop == 'elem':
                    continue
                
                df = DataFrame(d[struct_name][prop], index=d[struct_name]['elem'])
                df.to_csv('tmp.csv', header=False)
                ptable_plotter('tmp.csv', show=False, png_fname=f'plots/{out_dir}/{prefix}_{struct_name}_{prop}_{enlargement}.png')
                
    os.remove('tmp.csv')

def just_colored_plot():
    data = read_data('res/ptable_delta_defect_3.pickle')
    elems = data[3]['ni_fcc']['elem']
    vals = [0 for _ in range(len(elems))]
    for x in ['Ni', 'Mo', 'Au', 'Al']:
        idx = elems.index(x)
        vals[idx] = 1
    df = DataFrame(vals, index=elems)
    df.to_csv('tmp.csv', header=False)
    ptable_plotter('tmp.csv', show=False, png_fname=f'plots/colered_ptable.png')
    os.remove('tmp.csv')
    
def save_orbital_line_plot(elem_val_dict, prop, struct_name):
    out_dir = 'plots/delta_orbital_line_plots/'
    fname = f'{out_dir}{prop}_{struct_name}_orbital.png'

    elems_3d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    elems_4d = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
    elems_5d = ['Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

    colors = ['r', 'g', 'b']

    for i, group in enumerate([elems_3d, elems_4d, elems_5d]):
        vals = [elem_val_dict[e] for e in group]
        plt.plot(vals, '-o', color=colors[i], label=f'{i+3}d')
    
    # Add line at y=0
    y0_vals = [0 for _ in range(len(elems_3d))]
    plt.plot(y0_vals, color='black', label='', linestyle='--', linewidth=3)
   
    if prop == 'g_vrh':
        plt.ylabel("Predicted $G'_{VRH}$ (GPa)")

    elif prop == 'k_vrh':
        plt.ylabel("Predicted $K'_{VRH}$ (GPa)")

    elif prop == 'eform':
        plt.ylabel("Predicted $E'_{form}$ (eV/atom)")
    
    # make xticks labels
    labels = ['\n'.join([x,y,z]) for x,y,z in zip(elems_3d, elems_4d, elems_5d)]
    
    # plt.xticks(range(len(elems_3d)))
    plt.xticks(range(len(elems_3d)),labels=labels)
    
    plt.legend()
    plt.tight_layout(pad=3, w_pad=3, h_pad=3)
    plt.savefig(fname)
    plt.clf()
    
def make_orbital_line_plots():
    data = read_data('res/ptable_delta_defects.pickle')

    for enlargement in data:
        # make these plots only for enlargement (3,3,3)
        if enlargement == 3:
            d = data[enlargement]
            for struct_name in d:
                for prop in d[struct_name]:

                    if prop == 'elem':
                        continue

                    X = d[struct_name][prop]
                    elems = d[struct_name]['elem']

                    # merge 
                    merged = dict()
                    for x, elem in zip(X, elems):
                        merged[elem] = x
                    
                    save_orbital_line_plot(merged, prop, struct_name)
                
                    
def save_saturation_plot(data, prop_name):
    prop_dict = {'eform': ("$E_{form}$", '(eV/atom)'),
                 'k_vrh': ("$K_{VRH}$", '(GPa)'),
                 'g_vrh': ("$G_{VRH}$", '(GPa)')
                }

    tex_prop, unit = prop_dict[prop_name]
    
    fig, ax = plt.subplots()
    ticks = None
    markers = ['.', 'o', '*']
    
    for k, m in zip(data, markers):
        x = [e[0] for e in data[k]]
        y = [e[1] for e in data[k]]
        plt.plot(x, y, label=k, marker=m, zorder=1.0, linestyle='--')
        ticks = x
    
    # add dashed vertical line at (3,3,3) enlargement which is the second tick here so
    vtick = ticks[1]
    plt.axvline(x=vtick, color = 'black', linestyle='--', zorder=-1.0)
    
    plt.xticks(ticks)
    plt.xlabel('Number of atoms')
    plt.ylabel(f'Predicted {tex_prop} {unit}')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=13)

    # plt.title(f'Single defects in Mo BCC: Saturation plot of {tex_prop}', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/saturation/{prop_name}_saturation_plot.png')
    
    # clear plot
    plt.clf()

def make_saturation_plots():
    data = read_data('res/saturation.pickle')
    for prop in data:
        save_saturation_plot(data[prop], prop)
    
    
if __name__ == '__main__':
    dircheck()
    
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["figure.figsize"] = (8,8)

    make_parity_plots(font_size=21)
    make_kvrh_defect_plots()
    make_gvrh_defect_plots()
    make_eform_defect_plots()
    make_ptable_plots(delta=False)
    make_ptable_plots(delta=True)
    make_orbital_line_plots()
    make_saturation_plots()
