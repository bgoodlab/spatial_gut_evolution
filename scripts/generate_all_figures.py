import sys
import os

# Figure 1
# (all schematic?)

# Figure 2: toy model figure

# Figure 3: selection figure
# (have to make 2 pieces separately and glue them together later)
# Make panel a (schematic of established mutation)
os.system('python plot_establishment_figure.py')
# Make other panels (fixation profiles and net fixation probabilities)
os.system('python plot_selection_figure.py')

# Figure 4: human figure
os.system('python plot_human_figure.py')
