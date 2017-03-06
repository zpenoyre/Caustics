import scipy.optimize
import numpy as np
#analytic soln
G=1
m=1
M=1
mu=1
t=1
eps=1
turn=np.zeros(1)
def densProfile(minR,maxR,nR,M,m,t,nTurn=100000,G=4.96e-15): #updates parameters and finds density profile
    setGlobals(m,M,t,nTurn,G)
    Rs=genRads(minR,maxR,nR)
    Rhos=np.zeros(nR)
    for i in range(nR):   
        Rhos[i]=findDens(Rs[i])
    return Rs,Rhos

def setGlobals(new_m,new_M,new_t,nTurn,new_G): #change parameters of calculation (global in this file)
    global G
    G=new_G
    global m
    m=new_m
    global M
    M=new_M
    global mu
    mu=(2-(M/m))**-1
    global eps
    eps=1-(mu**-1)
    global t
    t=new_t
    global turn
    if (turn.size!=nTurn):
        turn=findTurns(nTurn)

def findRad(eta):
    return ((G*m*(t**2))*((1-eps*np.cos(eta))**3)/((eta-eps*np.sin(eta))**2))**(1/3)

def findR0(eta):
    return ((G*m*(t**2)/(mu**3))*((eta-eps*np.sin(eta))**-2))**(1/3)

def F(eta,R):
    return findRad(eta)-R

def theta(eta): #flipped compared to notebook!
    return 1-eps*np.cos(eta)

def omega(eta):
    return 1-eps*(np.cos(eta)+(3*np.sin(eta)/2)*((eta-eps*np.sin(eta))/(1-eps*np.cos(eta))))

def rho(R):
    return 1
def etaMax(R): #maximum possible (but not probable) value of eta for a given R
    return (G*m*(t**2)*((1+eps)**3)/(R**3))**(1/2)
def etaMin(R): #minimum possible (but not probable) value of eta for a given R
    return (G*m*(t**2)*((1-eps)**3)/(R**3))**(1/2)
def etaTurn(eta):
    return (3*eps*eta*np.sin(eta))+(4*eps*np.cos(eta))-2*(1+(eps**2))-(eps**2)*(np.sin(eta)**2)
    
def findTurns(n): #finds the eta corresponding to turning points in r(eta)
    turnPt=np.zeros(n+1)
    for i in range(1,n+1):
        lowerBound=(0.5+i)*np.pi
        upperBound=lowerBound+np.pi
        turnPt[i]=scipy.optimize.brentq(etaTurn,lowerBound,upperBound)
    return turnPt
    
def findInf(rMin,rMax): #finds the r0s between rMin and rMax corresponding to singularities in the density (omega(eta)=0) at time t
    lowEta=etaMin(rMax)
    #print("lower eta: ",lowEta)
    highEta=etaMax(rMin)
    #print("high eta: ",highEta)
    lowIndex=int((lowEta/np.pi)-1.5)
    if (lowIndex<0):
        lowIndex=0
    #print("lower index: ",lowIndex)
    highIndex=1+int((highEta/np.pi)-1.5)
    if (highIndex<1):
        return np.zeros(1)
    #print("high index: ",highIndex)
    nInf=highIndex-lowIndex
    rInf=np.zeros(nInf)
    for i in range(0,nInf):
        lowerBound=(1.5+i+lowIndex)*np.pi
        upperBound=lowerBound+np.pi
        etaInf=scipy.optimize.brentq(omega,lowerBound,upperBound)
        #print("root ",i," is ",etaInf)
        rInf[i]=findRad(etaInf)
    return rInf[((rInf>rMin) & (rInf<rMax))]
    
def genRads(rMin,rMax,nRads): #generates a list of radii irregularly spaced such as to be more dense around singularities (at a certain t)
    rads=np.linspace(np.sqrt(rMin),np.sqrt(rMax),nRads)**2
    rInf=findInf(rMin,rMax)
    nInf=rInf.size
    if (rInf[0]==0):
        return rads
    for i in range(0,nRads):
        #print("original rad: ",rads[i])
        near=np.argmin(np.abs(rads[i]-rInf))
        dist=rInf[near]-rads[i]
        #print(dist," from nearest point")
        #print("factor: ",np.exp(-(dist/(0.01*rads[i]))**2))
        rads[i]=rads[i]+dist*np.exp(-(dist/(0.01*rads[i]))**2)
        #print("new rad: ",rads[i])
    return rads
    
def findEtas(R): #finds etas for which r(eta)=R
    value,lowInt = min((b,a) for a,b in enumerate(turn-etaMin(R)) if (b>0 and a%2==1))
    lowInt-=1
    if (findRad(turn[lowInt+1])-R>0):
        lowInt+=2
    value,highInt = min((b,a) for a,b in enumerate(turn-etaMax(R)) if (b>0 and a%2==0))
    highInt-=1
    if (R-findRad(turn[highInt-1])>0):
        highInt-=2
    nInt=highInt-lowInt
    etas=np.zeros(nInt)
    diff=0
    for i in range(0,nInt):
        if (i==0):
            if (findRad(turn[lowInt])<R):
                etas=etas[:-1]
                diff=-1
                continue
        if (i==nInt-1):
            if (findRad(turn[highInt])<R and findRad(turn[highInt-1])<R):
                etas=etas[:-1]
                break
        lowerBound=turn[lowInt+i+diff]
        upperBound=turn[lowInt+i+1+diff]
        etas[i+diff]=scipy.optimize.brentq(F,lowerBound,upperBound,args=(R))
    return etas
    
def findDens(R): #finds the analytic density at one radii
    etas=findEtas(R)
    th_mu=(theta(etas)*mu)**-1
    dens=rho(R*th_mu)
    om_mu=(omega(etas)*mu)**-1
    return np.sum(dens*om_mu*(th_mu**2))