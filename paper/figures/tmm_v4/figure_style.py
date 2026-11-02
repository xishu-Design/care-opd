"""Shared experimental-figure style aligned with Method diagnostics v21.

Formatting only: no source values, statistics, or axes ranges are modified.
"""
import matplotlib.pyplot as plt

INK = '#1F2D40'
MUTED = '#596779'
AXIS = '#6C747C'
GRID = '#E4E9ED'
BLUE = '#5B8FD9'
ORANGE = '#F2A64A'
GREEN = '#4FB28F'
CLEAN = '#7D8795'
GKD = '#9D75B2'
CARE = '#0072B2'
COUNT_INK = '#172638'


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8.0, 'axes.labelsize': 8.0, 'axes.titlesize': 8.2,
        'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6, 'legend.fontsize': 7.5,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': AXIS,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'grid.color': GRID, 'grid.linewidth': .45,
        'legend.frameon': False,
        'lines.solid_joinstyle': 'round', 'lines.solid_capstyle': 'round',
        'lines.dash_joinstyle': 'round', 'lines.dash_capstyle': 'round',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
        'hatch.linewidth': .45,
    })
