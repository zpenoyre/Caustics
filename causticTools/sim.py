import Cython
import pyximport
pyximport.install()
from . import cythonSim

#this allows line profiler (via ipython magic %lprun) to profile cython functions
from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults()['linetrace'] = True
get_directive_defaults()['binding'] = True

def runSim(nShells,nPhase,nEcc,rho,T,dt,rMin,rMax,M,m,name,nOutput,G=4.96e-15):
    cythonSim.updateGlobal(G,M,m)
    cythonSim.runSim(nShells,nPhase,nEcc,T,dt,rMin,rMax,rho,nOutput,name)