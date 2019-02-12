"""
Tools for atmospheric stability analysis
"""
import numpy as np
import pandas as pd


#
# stability correction terms
#
def BusingerDyer(z_L):
    """References:

    Dyer AJ, Hicks BB. Flux-gradient relationships in the constant flux layer.
    Q J R Meteorolog Soc. 1970;96:715-721.

    Businger JA, Wyngaard JC, Izumi Y, Bradley EF. Flux-profile
    relationships in the atmospheric boundary layer. J Atmos Sci. 1971;
    28:181-189.

    Dyer AJ. A review of flux-profile relationships. Boundary-Layer Meteorol.
    1974; 7:363-372.
    """
    x = (1. - 16*z_L)**0.25
    unstable = 2*np.log((1.+ x)/2.) + np.log((1.+x**2)/2.) - 2*np.arctan(x) + np.pi/2
    stable = -5*z_L
    #for z,uns,sta in zip(z_L,unstable,stable):
    #    print(z,uns,sta)
    return np.where(z_L < 0, unstable, stable)


#
# stability estimators
#
class HybridWindEstimator(object):
    """The "H-W" estimator considers 3 levels of wind-speed measurements
    (or 2 levels of wind speeds and a reliable surface roughness).

    Ref: Basu, S. (2018). A simple recipe for estimating atmospheric
         stability solely based on surface-layer wind speed profile.
         Wind Energy, 21(10), 937–941. doi:10.1002/we.2203
    """
    def __init__(self,heights=None):
        if heights is None:
            self.heights = None
        else:
            assert(len(heights)==3)
            self.heights = np.sort(heights)
        

    def read_pandas(self,df,
            height_column='height',windspeed_column='wind_speed'):
        """Read pandas dataframe"""
        heights = np.sort(df[height_column].unique())
        if self.heights is None:
            self.heights = heights
            print('Found heights:',self.heights)
        else:
            # we already specified heights for the estimator
            if not np.all([heights.__contains__(h) for h in self.heights]):
                raise ValueError(
                    'Mismatch between specified heights',self.heights,
                    'and heights in dataframe',heights
                )
        self.U1 = df.loc[df[height_column]==self.heights[0],windspeed_column].values
        self.U2 = df.loc[df[height_column]==self.heights[1],windspeed_column].values
        self.U3 = df.loc[df[height_column]==self.heights[2],windspeed_column].values
        assert(len(self.U1) == len(self.U2) == len(self.U3))


    def calculate_opt(self,psi_m,error_output=False,**kwargs):
        """Obtain solution for general stability functions, psi_m(z/L),
        by solving an optimization problem. Optional kwargs are inputs
        to scipy.optimize.fsolve.
        """
        #from scipy.optimize import fsolve
        from scipy.optimize import brentq # can bracket results

        dU21 = self.U2 - self.U1
        dU31 = self.U3 - self.U1
        z1,z2,z3 = self.heights
        self.RN = np.log(z3/z1) / np.log(z2/z1)
        self.R = dU31 / dU21
        def fun(x,R):
            # x == 1/L
            return np.log(z3/z1) - R*np.log(z2/z1) \
                    - psi_m(z3*x) + R*psi_m(z2*x) + (1-R)*psi_m(z1*x)
        guess = np.sign(self.R - self.RN)
        invL = np.empty(guess.shape)
        invL.fill(np.nan)
        err = np.empty(guess.shape)
        fail_unstable, fail_stable = 0,0
        for i,(x0,Ri) in enumerate(zip(guess,self.R)):
            if x0 == 0:
                invL[i] = 0
            else:
                #soln,info,ierr,msg = fsolve(fun,x0,args=(Ri,),
                #                            full_output=True,**kwargs)
                #if ierr==1:
                #    invL[i] = soln[0]
                #else:
                #    if x0 > 0:
                #        fail_stable += 1
                #    else:
                #        fail_unstable += 1

                # need to have bounds because psi_m(z_L) may not be
                # continuously differentiable at z_L=0
                if x0 > 0:
                    bounds = (0,1)
                else:
                    bounds = (-1,0)
                try:
                    root,res = brentq(fun,bounds[0],bounds[1],args=(Ri,),full_output=True,**kwargs)
                except ValueError: 
                    pass
                else:
                    #print(res)
                    if res.converged:
                        invL[i] = root
                    else:
                        if x0 > 0:
                            fail_stable += 1
                        else:
                            fail_unstable += 1

                if error_output:
                    Rnum = np.log(z3/z1) - psi_m(z3*invL[i]) + psi_m(z1*invL[i])
                    Rden = np.log(z2/z1) - psi_m(z2*invL[i]) + psi_m(z1*invL[i])
                    err[i] = Ri - Rnum/Rden

        print('failed to find 1/L for # of stable/unstable cases:',fail_stable,fail_unstable)
        self.invL = invL
        self.error = err

        if error_output:
            return self.invL, self.error
        else:
            return self.invL

