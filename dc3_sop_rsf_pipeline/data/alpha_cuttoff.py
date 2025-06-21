import numpy as np

alpha_fcc = np.loadtxt('data/alpha_files/alpha_fcc.dat')
alpha_bcc = np.loadtxt('data/alpha_files/alpha_bcc.dat')
alpha_hcp = np.loadtxt('data/alpha_files/alpha_hcp.dat')
alpha_liquid = np.loadtxt('data/alpha_files/alpha_liquid.dat')

solid_min = min(alpha_fcc.min(), alpha_bcc.min(), alpha_hcp.min())
liquid_max = max(alpha_liquid)

alpha_cut = (solid_min + liquid_max)/2
print("Computed alpha cut: ", alpha_cut)

np.savetxt('data/alpha_cutoff.dat', [alpha_cut], fmt='%.6f')