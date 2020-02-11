"""
Surface Layer Similarity Functions
==================================

Library of similarity functions (psi) used in the log profile equations:

    U(z) = ustar/kappa * (log(z/z0) - psi_m(z/L) + psi_m(z0/L))
    theta(z) - theta(0) =
            thetastar/kappa * (log(z/z0) - psi_h(z/L) + psi_h(z0/L))

where z is the height a.g.l.,

    psi(z/L) = ((1 - phi(x)) / x) integrated from x= 0 to z/L

for psi_m and psi_h, and

    phi_m(z/L) = kappa*z/ustar * dU/dz
    phi_h(z/L) = kappa*z/thetastar * dtheta/dz

"""
import numpy as np


def Paulson_m(x):
    """Momentum similarity function for unstable conditions

    Ref: Paulson, C.A., 1970: The mathematical representation of wind
         speed and temperature in the unstable atmospheric surface layer.
         J. Appl. Meteor., 9, 857-861.
    """
    return np.pi/2 - 2*np.arctan(x) + np.log((1+x)**2 * (1 + x**2) / 8)

def Paulson_h(x):
    """Heat similarity function for unstable conditions

    Ref: Paulson, C.A., 1970: The mathematical representation of wind
         speed and temperature in the unstable atmospheric surface layer.
         J. Appl. Meteor., 9, 857-861.
    """
    return 2 * np.log((1 + x**2) / 2)


def Jimenez_m(z_L, a=6.1, b=2.5, alpha_m=10.0):
    """Momentum similarity function used by WRF

    Ref: Jimenez, P.A., J. Dudhia, J.F. Gonzalez-Rouco, J. Navarro, J.P.
         Montavez and E. Garcia-Bustamante, 2012: A Revised Scheme for
         the WRF Surface Layer Formulation. Mon. Weather Rev., 140, 898-918.
    """
    psi = np.zeros(z_L.shape)
    zeta = np.array(z_L)
    # Unstable conditions (Eqn. 17)
    uns = np.where(zeta < 0)
    x = (1 - 16*zeta[uns])**0.25
    paulson_func = Paulson_m(x)  # "Kansas-type" functions
    y = (1 - alpha_m*zeta[uns])**(1./3)
    conv_func = 3./2 * np.log(y**2 + y + 1./3) \
            - np.sqrt(3) * np.arctan(2*y + 1/np.sqrt(3)) \
            + np.pi/np.sqrt(3)  # convective contribution
    psi[uns] = (paulson_func + zeta[uns]**2 * conv_func) \
            / (1 + zeta[uns]**2)
    # Stable conditions (Eqn. 18)
    sta = np.where(zeta >= 0)
    psi[sta] = -a * np.log(zeta[sta] + (1 + zeta[sta]**b)**(1./b))
    return psi

