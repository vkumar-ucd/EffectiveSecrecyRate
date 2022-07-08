# On the secrecy rate under statistical QoS provisioning for RIS-assisted MISO wiretap channel
# Authors: (*)Vaibhav Kumar, (*)Mark F. Flanagan, (^)Derrick Wing Kwan Ng, and (*)Le-Nam Tran
# DOI: 10.1109/GLOBECOM46510.2021.9685957
# Conference: IEEE Globecom 2021, Madrid, Spain
# (*): School of Electrical and Electronic Engineering, University College Dublin, Belfield, Dublin 4, Ireland
# (^): School of Electrical Engineering and Telecommunications, University of New South Wales, NSW 2052, Australia
# email: vaibhav.kumar@ucdconnect.ie / vaibhav.kumar@ucd.ie / vaibhav.kumar@ieee.org

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


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

# Function to compute the objective in (15)
def gCalc(phi,r_bl,phi_bl,beta_bl,r_el,phi_el,beta_el):
    numLocal = r_bl*np.cos(phi-phi_bl)+beta_bl    
    denLocal = r_el*np.cos(phi-phi_el)+beta_el
    return numLocal/denLocal

# Function to update w
def updateW(theta):
    zb = hib@np.diag(thetaCurrent)@Hai+hab          # effective Alice-Bob channel       
    ze = hie@np.diag(thetaCurrent)@Hai+hae          # effective Alice-Eve channel
    Zb = Herm(zb)@zb                                # see below (7)
    Ze = Herm(ze)@ze                                # see below (7)
    tempMat = np.linalg.pinv(P*Ze+np.eye(N))@\
                    (P*Zb+np.eye(N))                # see below (8)
    eigVals,eigVecs = np.linalg.eig(tempMat)        # EigenValue Decomposition
    maxCol = list(eigVals).index(max(eigVals))      # index of the maximum eigenvalue
    uMax = eigVecs[:,[maxCol]]                      # eigenvector corresponding to largest eigenvalue
    uMax = uMax/np.linalg.norm(uMax,2)              # normalizing the eigenvector
    w = np.sqrt(P)*uMax                             # updated transmit beamformer
    return w 

# Function to update RIS phase shifts
def updateTheta(w,theta):
    # computing paramters below (10)
    ab = np.diag(np.ndarray.flatten(hib.conj()))@Hai.conj()@w.conj()
    bb = ab@(w.T)@(hab.T)
    cb = (abs(hab@w))**2
    # computing parameters below (12)
    ae = np.diag(np.ndarray.flatten(hie.conj()))@Hai.conj()@w.conj()
    be = ae@(w.T)@(hae.T)
    ce = (abs(hae@w))**2    
    
    for l in range(Nris):
        theta_m = np.delete(theta,l)
        # parameters related to Bob
        ab_l = ab[l]
        ab_m = np.delete(ab,l)
        bb_l = bb[l]
        bb_m = np.delete(bb,l)        
        alpha_bl = 2*(ab_l*np.sum(ab_m.conj()*theta_m)+bb_l)
        beta_bl = abs(ab_l)**2+abs(np.sum(ab_m.conj()*theta_m))**2\
                            +2*np.real(np.sum(bb_m.conj()*theta_m))+cb+1
        r_bl = abs(alpha_bl)
        phi_bl = np.angle(alpha_bl)
        # parameters related to Eve
        ae_l = ae[l]
        ae_m = np.delete(ae,l)
        be_l = be[l]
        be_m = np.delete(be,l)
        alpha_el = 2*(ae_l*np.sum(ae_m.conj()*theta_m)+be_l)
        beta_el = abs(ae_l)**2+abs(np.sum(ae_m.conj()*theta_m))**2\
                            +2*np.real(np.sum(be_m.conj()*theta_m))+ce+1        
        r_el = abs(alpha_el)
        phi_el = np.angle(alpha_el)
        # obtaining optimal phase-shift for the l-th tile 
        r_l = np.sqrt((r_bl*beta_el)**2 + (r_el*beta_bl)**2\
             - 2*r_bl*r_el*beta_bl*beta_el*np.cos(phi_el-phi_bl))       # see below (16)
        num = -r_bl*beta_el*np.sin(phi_bl)+r_el*beta_bl*np.sin(phi_el)
        den = r_bl*beta_el*np.cos(phi_bl)-r_el*beta_bl*np.cos(phi_el)
        varphi_l = np.arctan2(num,den)                                  # see above (17)
        phi_l0 = np.array([0])
        phi_l1 = np.arcsin((r_bl*r_el/r_l)\
                    *np.sin(phi_bl-phi_el))-varphi_l                    # see (18)
        phi_l2 = np.pi-np.arcsin((r_bl*r_el/r_l)\
                    *np.sin(phi_bl-phi_el))-varphi_l                    # see (18)
        phi_l_Vec = np.concatenate((phi_l0,phi_l1.flatten(),\
                        phi_l2.flatten()),axis=0)                       # list of possible phase-shifts
        maxIndex = np.argmax(gCalc(phi_l_Vec,r_bl,phi_bl,\
                        beta_bl,r_el,phi_el,beta_el))                   # index of optimal phase-shift in the list
        phi_l_opt = phi_l_Vec[maxIndex]                                 # optimal phase-shift for the l-th tile
        theta[l] = np.exp(1j*phi_l_opt)                                 # optimal reflection-coefficient for the l-th tile
    return theta

# Function to calculate secrecy rate
def rateCalc(w,theta):
    zb = hib@np.diag(theta)@Hai+hab
    ze = hie@np.diag(theta)@Hai+hae
    y = np.log2(1+abs(zb@w)**2)-np.log2(1+abs(ze@w)**2)
    return y   

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
relChange = 1e3                         # arbitrary large number
epsilon = 1e-5                          # convergence tolerance

# Generating the channel coefficients
normFact = 1/np.sqrt(sigmaSquare)           # normalization factor
Hai = chanGen(zetaAI,dAI,Nris,N)            # Alice-RIS channel                                        
hib = normFact*chanGen(zetaIB,dIB,1,Nris)   # RIS-Bob channel
hie = normFact*chanGen(zetaIE,dIE,1,Nris)   # RIS-Eve channel
hab = normFact*chanGen(zetaAB,dAB,1,N)      # Alice-Bob channel
hae = normFact*chanGen(zetaAE,dAE,1,N)      # Alice-Eve channel

# Random initialization
thetaCurrent = np.ones((Nris,),dtype=complex)

# Algorithm 1: Block-Coordinate Ascent Method
while relChange > epsilon:  
    # update transmit beamformer 
    wCurrent = updateW(thetaCurrent) 
    # update RIS phase shifts
    thetaCurrent = updateTheta(wCurrent,thetaCurrent) 
    # calculate rate
    rateSeq = np.append(rateSeq,rateCalc(wCurrent,thetaCurrent))
    # check for convergence
    if len(rateSeq) > 3:
        relChange = (rateSeq[-1] - rateSeq[-2])/rateSeq[-2]

# Plot the iterates 
plt.plot(rateSeq)
plt.xlabel('Iteration number')
plt.ylabel('Secrecy rate (bps/Hz)')
# plt.show()
plt.savefig("./Convergence.pdf", format="pdf", bbox_inches="tight")