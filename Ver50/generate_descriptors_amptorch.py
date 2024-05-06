import numpy as np
import itertools
from collections import Counter
from dscribe.descriptors import ACSF, SOAP
from ase.io import Trajectory
from ase import Atoms

def Generate_acsf_descriptor(atoms):
    epsilon = [1.0]
    g2_eta = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    g4_eta = [0.005]
    R_s = [0] * 4
    lamb = [-1, 1]
    r_cut = 6.5

    g2_eta =[a / r_cut**2 for a in g2_eta]
    g4_eta = [a / r_cut**2 for a in g4_eta]

    g2_params = [list(item) for item in itertools.product(g2_eta, R_s)]
    g4_params = [list(item) for item in itertools.product(g4_eta, epsilon, lamb)]


    # Set up: Instantiating an ACSF descripto
    species = set()
    species.update(atoms.get_chemical_symbols())
    acsf = ACSF(
        species=species,
        r_cut=r_cut,
        g2_params=g2_params,
        g4_params=g4_params,
        )

        # Create ACSF output for the system
    acsf_fp = acsf.create(atoms, n_jobs=-1)

    return acsf_fp

def Generate_soap_descriptor(atoms):
    r_cut = 6.5
    n_max = 9
    l_max = 10

    # Set up: Instantiating an ACSF descripto
    species = set()
    species.update(atoms.get_chemical_symbols())

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,    )


    soap_fp  = soap.create(atoms, n_jobs=-1)
    #print(soap_traj.shape)

    return soap_fp

