from numpy import *

# Information about lattices in the synthetic data set.
lattices = array(['fcc','bcc','hcp'])
n_lattices = len(lattices)
y_lattices = array([0,1,2,3,4,5])

# Information about lattices in MD simulations.
md_lattices = array(['fcc','bcc','hcp'])
n_md_lattices = len(md_lattices)
y_md_lattices = array([0,1,2,3,4,5,-1,-2,-3,-4])

# Number of features employed.
N_feat = 120