import numpy as np
cimport numpy as np

cpdef np.ndarray findSmoothDens(np.ndarray[double,ndim=1] rads,int nBins,np.ndarray shellData,double smoothLength):
    cdef:
        np.ndarray[np.double_t,ndim=1] dens=np.zeros(nBins)
        np.ndarray[np.double_t,ndim=1] sigma=np.sqrt(smoothLength**2+np.power(0.5*(shellData[:,4]-shellData[:,3]),2))
        np.ndarray[np.double_t,ndim=1] mid=0.5*(shellData[:,3] + shellData[:,4])
        np.ndarray[np.double_t,ndim=1] rho0=shellData[:,1]/ \
            (4* (2*np.pi)**1.5 * np.power(sigma,3) * (1+0.5*np.power(mid/sigma,2)+(4/np.sqrt(2*np.pi))*(mid/sigma)))
        np.ndarray[np.double_t,ndim=1] one_twoSigmaSq=1/(2*np.power(sigma,2))
        int i
    for i in range(nBins):
        dens[i]=np.sum(rho0*np.exp(-(rads[i]-mid)**2 *one_twoSigmaSq))
    return dens