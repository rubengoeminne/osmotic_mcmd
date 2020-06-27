

import numpy as np
import tempfile
import os
import pkg_resources

from molmod import angstrom, nanometer, kjmol

from yaff import *

__all__ = [
    'get_system_cof300', 'get_ff_cof300', 'get_system_argon'
]



def get_system_cof300():
    s = System.from_file(pkg_resources.resource_filename(__name__, '../data/lp_avg.chk'))
    return s

def get_ff_cof300(system):
    ff = ForceField.generate(system, pkg_resources.resource_filename(__name__, '../data/pars.txt'))
    return ff

def get_system_argon():
    s = System.from_file(pkg_resources.resource_filename(__name__, '../data/argon.chk'))
    return s
