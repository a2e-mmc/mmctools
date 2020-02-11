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


def Paulson_m(z_L):
    """Momentum similarity function for unstable conditions

    Ref: Paulson, C.A., 1970: The mathematical representation of wind
         speed and temperature in the unstable atmospheric surface layer.
         J. Appl. Meteor., 9, 857-861.
    """
    return np.pi/2 - 2*np.arctan(z_L) + np.log((1+z_L)**2 * (1 + z_L**2) / 8)

def Paulson_h(z_L):
    """Heat similarity function for unstable conditions

    Ref: Paulson, C.A., 1970: The mathematical representation of wind
         speed and temperature in the unstable atmospheric surface layer.
         J. Appl. Meteor., 9, 857-861.
    """
    return 2 * np.log((1 + z_L**2) / 2)



