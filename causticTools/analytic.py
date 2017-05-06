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
def densProfile(minR,maxR,nR,M,m,t,nTurn=100000,G=4.96e-15,returnGradient=0,evenSpaced=1): #updates parameters and finds density profile (and optionally it's gradient)
    setGlobals(m,M,t,nTurn,G)
    if evenSpaced==0:
        Rs=genRads(minR,maxR,nR)
    else:
        Rs=np.linspace(np.sqrt(minR),np.sqrt(maxR),nR)**2
    Rhos=np.zeros(nR)
    if (returnGradient==1):
        Grads=np.zeros(nR)
    for i in range(nR):   
        if (returnGradient==1):
            Rhos[i],Grads[i]=findDens(Rs[i],returnGrad=1)
        else:
            Rhos[i]=findDens(Rs[i])
    if (returnGradient==1):
        return Rs,Rhos,Grads
    else:
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
    
def alpha(eta): #could drop this in to a few other places too?
    return (eta-eps*np.sin(eta))/(1-eps*np.cos(eta))
    
def dRho_dr(eta): #finds dRho_dr for a given eta. Non-sensical over singularities but defined elsewhere
    print('eta: ',eta)
    R=findRad(eta)
    print('R: ',R)
    rho=findDens(R)
    om=omega(eta)
    al=alpha(eta)
    num=(1-eps*(3/2))*np.sin(eta)+al*eps*(3/2)*((eps**2)*(np.sin(eta)**2) - np.cos(eta))
    denom=3*(al**2)*eps*np.sin(eta)-2*(eta-eps*np.sin(eta))
    return -(rho/om)*(3*(al**4)*(R**2)*eps)*(num/denom)
    
    
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
    if (rInf.size==0):
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
    
def findDens(R,printWorking=0,returnGrad=0): #finds the analytic density at one radii
    etas=findEtas(R)
    one_thmu=(theta(etas)*mu)**-1
    dens=rho(R*one_thmu)
    #WHY IS THIS ABSOLUTE VALUE!?!?!
    one_ommu=(omega(etas)*mu)**-1
    Rho=np.sum(dens*np.abs(one_ommu)*(one_thmu**2))
    if(printWorking==1):
        print('______at R: ',R)
        print('etas: ',etas)
        print('1/mu*theta: ',one_thmu)
        print('dens(r0): ',dens)
        print('1/mu*omega: ',one_ommu)
        print('dens(R): ',np.sum(dens*one_ommu*(one_thmu**2)))
    if(returnGrad==1):
        al=alpha(etas)
        om=omega(etas)
        th=theta(etas)
        num=(1-eps*(3/2))*np.sin(etas)+al*eps*(3/2)*((eps**2)*(np.sin(etas)**2) - np.cos(etas))
        denom=3*(al**2)*eps*np.sin(etas)-2*(etas-eps*np.sin(etas))
        rho_i=dens*one_ommu*(one_thmu**2)
        dr_deta=(1/(3*(R**2)*(al**3)))*(3*eps*al*np.sin(etas)-2*(1-eps*np.cos(etas)))
        otherTerm=(2*eps*np.sin(etas)/th)+(eps/(2*om))*((3*al*(eps-np.cos(etas))/(1-eps*np.cos(etas))) - np.sin(etas))
        grad=np.sum(-rho_i*otherTerm/dr_deta)
        return Rho,grad
    return Rho