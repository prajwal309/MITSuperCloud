#importing libraries
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import batman
import emcee
import corner
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d, CubicSpline
from os import path


def log_prior(theta):
    '''
    Mostly the uniform priors for the variables
    '''

    #Unpack the values
    ARef, Mp, Offset, T0, rp_Rs, a_Rs, b, u1, u2 = theta
    Mp = Mp*JupMass

    if Mp<0.0:
        return -np.inf

    if np.abs(Offset)>20000:
        return -np.inf

    if np.abs(T0)>0.1:
        return -np.inf

    if (rp_Rs<0.0 or rp_Rs>0.4):
        return -np.inf

    if (b<0.0 or b>a_Rs):
        return -np.inf

    if np.abs(u1-u1_Read)>0.1:
        return -np.inf

    if np.abs(u2-u2_Read)>0.1:
        return -np.inf

    PSec =  Period_Read*Day2Sec
    CalcStDensity = 3.0*np.pi/(G*PSec**2.0)*(a_Rs)**3.0

    #Density has to be within three sigma
    if (CalcStDensity - StellarDensity)/ErrorDensity>5.0:
        return -np.inf
    #Time for inferior conjunction can be anywhere
    return 0


def log_likelihood(theta, Phase, Flux):
    '''likelihood function'''


    global ErrorSumLeast

    #unpacking the values
    ARef, Mp, Offset, T0, rp_Rs, a_Rs, b, u1, u2 = theta
    Mp = Mp*JupMass

    #Convert b in to inclination
    Inc = np.arccos(b/a_Rs)
    Inc_Deg = np.rad2deg(Inc)

    #Chunking data into transit fit
    PhaseCutOff = 0.035
    PhaseTransitIndex = np.abs(Phase)<PhaseCutOff
    TransitTime = Phase[PhaseTransitIndex]*Period_Read
    TransitFlux = Flux[PhaseTransitIndex]

    #Chunking data for phase fit
    Phase4PC = Phase[~PhaseTransitIndex] - T0
    Flux4PC = Flux[~PhaseTransitIndex]

    AEll = alpha_ell*Mp/MStar_Read*(1./a_Rs)**3.0*(np.sin(Inc))**2.0

    K =(2.*np.pi*G/(Period_Read*Day2Sec))**(1./3.)*Mp*np.sin(Inc)/(MStar_Read**(2./3.))
    ADop = alpha_dop*K/c

    Ref = ARef*(1.0+np.cos(2.*np.pi*Phase4PC+np.pi))*1.e6
    Ell = AEll*(1.0+np.cos(4.*np.pi*Phase4PC-np.pi))*1.e6
    Dop = ADop*(1.0+np.cos(2.*np.pi*Phase4PC-np.pi/2))*1.e6 - ADop*1.e6

    #Note the HS means highly sampled
    FluxPhaseCurve = (Ref + Ell + Dop)+Offset

    # Evaluate a batman model
    paramsBatman.t0 = T0*Period_Read            #time of conjunction
    paramsBatman.per = Period_Read              #orbital period
    paramsBatman.rp = rp_Rs                     #planet radius (in units of stellar radii)
    paramsBatman.a = a_Rs                       #semi-major axis (in units of stellar radii)


    paramsBatman.inc = Inc_Deg                  #orbital inclination (in degrees)
    paramsBatman.ecc = 0                        #eccentricity
    paramsBatman.w = 90.0                       #longitude of periastron (in degrees)
    paramsBatman.limb_dark = "quadratic"        #limb darkening model
    paramsBatman.u = [u1,u2]                    #limb darkening parameters

    #Generate the light curve for batman
    mTransit = batman.TransitModel(paramsBatman, TransitTime, supersample_factor = 15, exp_time = 0.0204166667)#initializes model
    ModelTransitFlux =  (mTransit.light_curve(paramsBatman)-1.0)*1.e6

    ResidualTransit = TransitFlux - ModelTransitFlux

    #Now model the secondary eclipse
    Fp = 2.0*ARef
    paramsBatman.fp = Fp
    paramsBatman.t_secondary = (0.5+T0)*Period_Read

    TSecondary = Phase4PC*Period_Read
    mSecondary = batman.TransitModel(paramsBatman, TSecondary,transittype="secondary", supersample_factor = 15, exp_time = 0.0204166667)#initializes model
    SecEclip =  (mSecondary.light_curve(paramsBatman)-1.0 - Fp)*1.e6


    ModelCombined = SecEclip + FluxPhaseCurve
    ResidualPC = Flux4PC -ModelCombined

    #Summing up the error
    ErrorSum = np.sum(ResidualPC**2.0)+np.sum(ResidualTransit**2.0)
    if ErrorSum<ErrorSumLeast:
        ErrorSumLeast = ErrorSum
        print "Phase curve residual %4.3e" %(np.sum(ResidualPC**2.0))
        print "Transit residual %4.3e" %(np.sum(ResidualTransit**2.0))
        print " Error sum is %4.3e" %(ErrorSum)
        ParamsDictionary = zip(["ARef", "Mp", "Offset", "T0","rp_Rs", "a_Rs", "b", "u1", "u2"], [ARef, Mp/JupMass, Offset, T0, rp_Rs, a_Rs, b, u1, u2])
        BestParamFile = open("Data/5_BestParam_NoThermal.txt",'w')
        BestParamFile.write("Error,"+str(ErrorSum)+"\n")
        for u,v in ParamsDictionary:
            print u,"::", v
            BestParamFile.write(u+","+str(v)+"\n")
        BestParamFile.close()
        print "*"*50
        if ErrorSum<0.0:
            plt.figure()
            plt.subplot(121)
            plt.plot(TransitTime, TransitFlux, "ko")
            plt.plot(TransitTime, ModelTransitFlux, "r-")
            plt.subplot(122)
            plt.plot(Phase4PC, Flux4PC, "ko")
            plt.plot(Phase4PC, ModelCombined, "gd")
            plt.show()

    STDVal = 43.75
    ChiSq = ErrorSum/(STDVal**2.0)
    return -(ChiSq)


def log_posterior(theta, Phase, Flux):
    prior = log_prior(theta)
    if np.isfinite(prior):
        #evaluate likelihood function only if passes the prior test
        return prior + log_likelihood(theta, Phase, Flux)
    else:
        return -np.inf



#define Constant
AU = 149597870700. # m, courtesy NASA JPL NEO
SolarMass = 1.989e30 # kg, courtesy Swinburne COSMOS
SolarRadius = 695990000. # m, courtesy NASA Marshall Space Flight Center Solar Physics
JupMass = 1898.3e24 # kg, courtesy NASA NSSDC Jupiter fact sheet
JupRad = 71492000. # m, courtesy NASA NSSDC Jupiter fact sheet
G = 6.672e-11 # m^3 kg^-1 s^-2, courtesy Wikipedia
c = 299792458 # m s^-1, courtesy Wikipedia
h = 6.626e-34 # J s, courtesy Wikipedia
Boltzmann = 1.38e-23 # J K^-1, courtesy Wikipedia
Day2Sec = 24.*60*60.

#Read the parameters from the file
Location = "Data/K2_31b.dat"
ParamDat = np.genfromtxt(Location, dtype="string", delimiter=":")
Param = OrderedDict(ParamDat)

global Period_Read, StellarDensity, ErrorDensity, MPlanet_Read, u1_Read, u2_Read

rp_Rs_Read = float(Param['rp_Rs'])
a_Rs_Read = float(Param['a_Rs'])
Err_a_Rs = float(Param['Err_a_Rs'])
Inc_Read  = float(Param['Inc'])
Err_Inc = float(Param['Err_Inc'])
TStar_Read = float(Param['TStar'])
Err_TStar = float(Param['Err_TStar'])
MStar_Read = float(Param['MStar'])*SolarMass
Err_MStar = float(Param['Err_MStar'])*SolarMass
RStar_Read = float(Param['RStar'])*SolarRadius
Err_RStar = float(Param['Err_RStar'])*SolarRadius
MPlanet_Read = float(Param['MPlanet'])
Err_MPlanet = float(Param['Err_MPlanet'])

StellarDensity = MStar_Read/(4./3.*np.pi*RStar_Read**3.0)
ErrorDensity = 1./(4./3.*np.pi)*np.sqrt(((Err_MStar)/RStar_Read**3.0)**2.0+(3.0*MStar_Read/RStar_Read**4.0*Err_RStar)**2.0)

Period_Read = float(Param['Period'])
e_Read = float(Param['Ecc'])
omega_Read = float(Param['Omega'])
u_Read = float(Param['u'])
g_Read = float(Param['g'])
u1_Read = float(Param['u1'])
u2_Read = float(Param['u2'])


#Calculate the coefficient
alpha_ell = 0.15*(15.0+u_Read)*(1.0+g_Read)/(3.-u_Read)
alpha_dop = 7.2 - (6.e-4)*TStar_Read

#Phase folded data
LightCurveLoc = "Data/K2-31b_Everest_ButterworthBandPass_Folded.txt" #Select a binned light curve
Data_Folded = np.genfromtxt(LightCurveLoc,delimiter=',',skip_header=0)
Phase = Data_Folded[:,0]
Flux = Data_Folded[:,1]

#Define the lowest Chisq to save the best fitting data
global ErrorSumLeast
nwalkers = 50
ndim = 9

#Read the values from the
if path.exists("Data/5_BestParam_NoThermal.txt"):
    print "Initializing the parameters from the previously best known parameters."
    #Initialize to the best parameters
    DataBestParams = np.loadtxt("Data/5_BestParam_NoThermal.txt", delimiter=',', dtype="string")

    ErrorSumLeast, ARef_Best, Mp_Best, Offset_Best, T0_Best, rp_Rs_Best, a_Rs_Best, b_Best, u1_Best, u2_Best = DataBestParams[:,1].astype(np.float)

    ARef_Init = np.abs(np.random.normal(ARef_Best,5e-6,nwalkers))
    MPlanet_Init = np.abs(np.random.normal(Mp_Best,1.0*Err_MPlanet,nwalkers))
    Offset = np.random.normal(Offset_Best,5,nwalkers)

    #Parameters for fitting the transits
    T0_Init = np.random.normal(T0_Best, 1e-5, nwalkers)         #time of conjunction of the star
    Rp_Init = np.random.normal(rp_Rs_Best, 1e-4, nwalkers)      #planet radius (in units of stellar radii)
    aR_Init = np.random.normal(a_Rs_Best, 1e-1, nwalkers)       #semi-major axis (in units of stellar radii)

    b_Init = np.random.normal(b_Best,1e-1,nwalkers)             #impact parameter
    u1 = np.random.normal(u1_Best,1e-2, nwalkers)               #u1
    u2 = np.random.normal(u2_Best, 1e-2, nwalkers)              #u2

else:

    print "Initializing values based on discovery paper fit."
    #random set of initializer

    ErrorSumLeast = np.inf
    np.random.seed(10)

    #Parameters for Phase curve fitting
    ARef_Init = np.abs(np.random.normal(2e-5,5e-6,nwalkers))
    MPlanet_Init = np.abs(np.random.normal(MPlanet_Read,5.0*Err_MPlanet,nwalkers))
    Offset = np.random.normal(-50,10,nwalkers)

    #Parameters for fitting the transits
    T0_Init = np.random.normal(0, 1e-3, nwalkers)               #planet radius (in units of stellar radii)
    Rp_Init = np.random.normal(rp_Rs_Read, 1e-3, nwalkers)      #planet radius (in units of stellar radii)
    aR_Init = np.random.normal(a_Rs_Read, 1e-1, nwalkers)       #semi-major axis (in units of stellar radii)

    b = np.cos(np.deg2rad(Inc_Read))*a_Rs_Read
    b_Init = np.random.normal(b,1e-1,nwalkers)                  #impact parameter

    #Parameter for secondary eclipse fitting
    u1 = np.random.normal(u1_Read,1e-1, nwalkers)               #u1
    u2 = np.random.normal(u2_Read, 1e-1, nwalkers)              #u2


StartingGuessSelected = np.column_stack((ARef_Init,MPlanet_Init, Offset, T0_Init,Rp_Init, aR_Init, b_Init, u1, u2))

NSteps = 100000

#Initialize the transit model with batman
global paramsBatman
paramsBatman = batman.TransitParams()     #object to store transit parameters

counter = 0
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[Phase,Flux], threads=8)

print "Initializing to a test run"
pos, prob, state = sampler.run_mcmc(StartingGuessSelected, 200)
sampler.reset()

print "Starting test run"
pos, prob, state = sampler.run_mcmc(pos, NSteps)
print "Run completed"

print "The acceptance fraction was:", (np.mean(sampler.acceptance_fraction))

BurnIn = int(1./2.*NSteps)
samples = sampler.flatchain[nwalkers*BurnIn:,:]


Parameters = ["ARef", "Mp", "Offset", "T0","rp_Rs", "a_Rs", "b", "u1", "u2"]

fig,ax = plt.subplots()
corner.corner(samples, labels=Parameters, plot_contours="True", title_fmt=".2e",quantiles=[0.158, 0.5, 0.842], show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('Figures/1_CornerPlot_NoThermal.png')

np.savetxt("Data/1_RunData.txt", samples,  delimiter=',', )
#

#Save in a file
ResultFile = open("Data/Run1Result.txt", "w+")

for i in range(ndim):
    Values = np.percentile(samples[:,i],[15.8, 50.0, 84.2])
    ValuesStr = str(Values[0])+","+str(Values[1])+","+str(Values[2])+"\n"
    print Parameters[i], "::",Values
    ResultFile.write(Parameters[i]+":"+ValuesStr)

ResultFile.close()
