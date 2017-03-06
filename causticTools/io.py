import bokeh
import bokeh.plotting as blt
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from . import cythonMisc

def plotDens(fig,whichOutput,nOutput,simName):
    # 0_shellIndex, 1_shellMass, 2_shellDens, 3_r1, 4_r2, 5_vr1, 6_vr2, 7_M1, 8_M2
    shellData=np.genfromtxt('output/'+simName+'.'+str(int(whichOutput))+'_'+str(int(nOutput))+'.txt')
    #nasty little hack to get points just either side of the steps in density
    rads=np.hstack((shellData[:,3],shellData[:,4]))
    rads=np.hstack((rads-1e-6,rads+1e-6)) 
    rads=np.sort(rads)
    dens=np.zeros_like(rads)
    
    for ind,rad in enumerate(rads):
        overlapShells=np.argwhere((shellData[:,3] < rad) & (shellData[:,4] > rad))
        dens[ind]=np.sum(shellData[overlapShells,2])
    
    cols=bokeh.palettes.Viridis11
    fig.line(rads,dens,color=cols[int(11*(whichOutput/nOutput))],line_width=2)
    return fig

def plotSmoothDens(fig,whichOutput,nOutput,simName,smoothLength,minR,maxR):
    # 0_shellIndex, 1_shellMass, 2_shellDens, 3_r1, 4_r2, 5_vr1, 6_vr2, 7_M1, 8_M2
    shellData=np.genfromtxt('output/'+simName+'.'+str(int(whichOutput))+'_'+str(int(nOutput))+'.txt')
    dr=0.25
    nBins=int((maxR-minR)/dr)
    rads=np.linspace(minR+dr,maxR-dr,nBins)
    dens=cythonMisc.findSmoothDens(rads,nBins,shellData,smoothLength)
    
    cols=bokeh.palettes.Viridis11
    fig.line(rads,dens,color=cols[int(11*(whichOutput/nOutput))],line_width=2)
    return fig
    
def plotDensProfile(fig,Rs,Rhos,minR,maxR,colour): #if alread have radii and densities plot directly
    cols=bokeh.palettes.Viridis11
    fig.line(Rs,Rhos,color=cols[int(11*colour)],line_width=2)
    return fig