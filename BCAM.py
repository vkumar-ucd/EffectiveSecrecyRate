# On the secrecy rate under statistical QoS provisioning for RIS-assisted MISO wiretap channel
# Authors: (*)Vaibhav Kumar, (*)Mark F. Flanagan, (^)Derrick Wing Kwan Ng, and (*)Le-Nam Tran
# DOI: 10.1109/GLOBECOM46510.2021.9685957
# Conference: IEEE Globecom 2021, Madrid, Spain
# (*): School of Electrical and Electronic Engineering, University College Dublin, Belfield, Dublin 4, Ireland
# (^): School of Electrical Engineering and Telecommunications, University of New South Wales, NSW 2052, Australia
# email: vaibhav.kumar@ucdconnect.ie / vaibhav.kumar@ucd.ie / vaibhav.kumar@ieee.org

import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(1)


# Function for dB to power conversion
def db2pow(x):
    return 10**(0.1*x)

# Function for power to dB conversion
def pow2db(x):
    return 10*np.log10(x) 

# Hermitian function
def Herm(x):
    return x.conj().T    

# Function to generate Rayleigh fading channel coefficients
def chanGen(zeta,d,dim1,dim2):
    pl0dB = -30                                     # pathloss (in dB) at reference distance
    pl = db2pow(pl0dB-10*zeta*np.log10(d))          # pathloss (in dB) at the given distance
    y = np.sqrt(pl/2)*(np.random.randn(dim1,dim2)\
            +1j*np.random.randn(dim1,dim2))         # Rayleigh distributed random variables 
    return y    

# Function to compute f(phi_l)
def fCalc(phi):
    numLocal = r_bl*np.cos(phi-phi_bl)+beta_bl    
    denLocal = r_el*np.cos(phi-phi_el)+beta_el
    return numLocal/denLocal




# System parameters
P = db2pow(15)                          # transmit power at Alice
N = 8                                   # number of antennas at Alice
Nris = 64                               # number of tiles at RIS
sigmaSquare = db2pow(-75)               # noise power at Bob/Eve
zetaAI = 2.2                            # pathloss exponent for Alice-RIS link
zetaIB = 2.5                            # pathloss exponent for RIS-Bob link
zetaIE = 2.5                            # pathloss exponent for RIS-Eve link
zetaAB = 3.5                            # pathloss exponent for Alice-Bob link
zetaAE = 3.5                            # pathloss exponent for Alice-Eve link
dAI = 50                                # distance between Alice and RIS
dAEh = 44                               # horizontal distance between Alice and Eve
dv = 2                                  # vertical distance between the line joining Alice-RIS and Bob-Eve
dABh = 50                               # horizontal distance between Alice and Bob
dAB = np.sqrt(dABh**2 + dv**2)          # distance between Alice and Bob
dAE = np.sqrt(dAEh**2 + dv**2)          # distance between Alice and Eve
dIE = np.sqrt((dAI-dAEh)**2 + dv**2)    # distance between RIS and Eve
dIB = np.sqrt((dAI-dABh)**2 + dv**2)    # distance between RIS and Eve 
rateSeq = np.array([])                  # empty list to store the rate sequence 
maxIter = 100                           # maximum number of iterations

# Generating the channel coefficients
normFact = 1/np.sqrt(sigmaSquare)           # normalization factor
Hai = chanGen(zetaAI,dAI,Nris,N)            # Alice-RIS channel                                        
hib = normFact*chanGen(zetaIB,dIB,1,Nris)   # RIS-Bob channel
hie = normFact*chanGen(zetaIE,dIE,1,Nris)   # RIS-Eve channel
hab = normFact*chanGen(zetaAB,dAB,1,N)      # Alice-Bob channel
hae = normFact*chanGen(zetaAE,dAE,1,N)      # Alice-Eve channel

# Random initialization
thetaCurrent = np.ones((Nris,),dtype=complex)

# Effective channels
zb = hib@np.diag(thetaCurrent)@Hai+hab
ze = hie@np.diag(thetaCurrent)@Hai+hae

# Algorithm 1: Block-Coordinate Ascent Method
for iIter in range(maxIter):    
    Zb = Herm(zb)@zb
    Ze = Herm(ze)@ze
    tempMat = np.linalg.pinv(P*Ze+np.eye(N))@(P*Zb+np.eye(N))
    eigVals,eigVecs = np.linalg.eig(tempMat)
    maxCol = list(eigVals).index(max(eigVals))
    uMax = eigVecs[:,[maxCol]]
    uMax = uMax/np.linalg.norm(uMax,2)
    wUpdated = np.sqrt(P)*uMax
    ab = np.diag(np.ndarray.flatten(hib.conj()))@Hai.conj()@wUpdated.conj()
    bb = ab@(wUpdated.T)@(hab.T)
    cb = (abs(hab@wUpdated))**2
    ae = np.diag(np.ndarray.flatten(hie.conj()))@Hai.conj()@wUpdated.conj()
    be = ae@(wUpdated.T)@(hae.T)
    ce = (abs(hae@wUpdated))**2    
    
    for nris in range(Nris):
        theta_l = thetaCurrent[nris]
        theta_m = np.delete(thetaCurrent,nris)
        ab_l = ab[nris]
        ab_m = np.delete(ab,nris)
        bb_l = bb[nris]
        bb_m = np.delete(bb,nris)
        ae_l = ae[nris]
        ae_m = np.delete(ae,nris)
        be_l = be[nris]
        be_m = np.delete(be,nris)
        alpha_bl = 2*(ab_l*np.sum(ab_m.conj()*theta_m)+bb_l)
        beta_bl = abs(ab_l)**2+abs(np.sum(ab_m.conj()*theta_m))**2+2*np.real(np.sum(bb_m.conj()*theta_m))+cb+1
        alpha_el = 2*(ae_l*np.sum(ae_m.conj()*theta_m)+be_l)
        beta_el = abs(ae_l)**2+abs(np.sum(ae_m.conj()*theta_m))**2+2*np.real(np.sum(be_m.conj()*theta_m))+ce+1
        r_bl = abs(alpha_bl)
        phi_bl = np.angle(alpha_bl)
        r_el = abs(alpha_el)
        phi_el = np.angle(alpha_el)
        phi_l = np.angle(theta_l)
        r_l = np.sqrt((r_bl*beta_el)**2 + (r_el*beta_bl)**2\
             - 2*r_bl*r_el*beta_bl*beta_el*np.cos(phi_el-phi_bl))
        num = -r_bl*beta_el*np.sin(phi_bl)+r_el*beta_bl*np.sin(phi_el)
        den = r_bl*beta_el*np.cos(phi_bl)-r_el*beta_bl*np.cos(phi_el)
        varphi_l = np.arctan2(num,den)
        phi_l0 = np.array([0])
        phi_l1 = np.arcsin((r_bl*r_el/r_l)*np.sin(phi_bl-phi_el))-varphi_l
        phi_l2 = np.pi-np.arcsin((r_bl*r_el/r_l)*np.sin(phi_bl-phi_el))-varphi_l
        phi_l_Vec = np.concatenate((phi_l0,phi_l1.flatten(),phi_l2.flatten()),axis=0)
        maxIndex = np.argmax(fCalc(phi_l_Vec))
        phi_l_opt = phi_l_Vec[maxIndex]
        theta_l = np.exp(1j*phi_l_opt)
        thetaCurrent[nris] = theta_l
    zb = hib@np.diag(thetaCurrent)@Hai+hab
    ze = hie@np.diag(thetaCurrent)@Hai+hae
    rateSeq = np.append(rateSeq,np.log2(1+abs(zb@wUpdated)**2)-np.log2(1+abs(ze@wUpdated)**2))
plt.plot(rateSeq)
plt.show()
        



    


