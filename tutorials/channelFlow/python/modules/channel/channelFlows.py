import numpy as np
import scipy.optimize as optimize
from scipy.optimize import bisect
from multipledispatch import dispatch # use for function overloading

@dispatch(object)
def poiseuille_scaled(y):
    '''
    Analytic Poiseuille-flow velocity field between two plates.
    velocity scale: G*R**2/(2*mu)
    length scale: R

    Parameters:

    y : array_like
        coordinate in corrs section of the channel, where the coordinate system
        is placed in the center of the channel
    '''
    return poiseuille_scaled(y,0)

@dispatch(object, object)
def poiseuille_scaled(y, S=0):
    '''
    Analytic Poiseuille-flow velocity field between two plates.
    velocity scale: G*R**2/(2*mu)
    length scale: R

    Parameters:

    y : array_like
        coordinate in corrs section of the channel, where the coordinate system
        is placed in the center of the channel
    S : optional
        dimensionless slip length (ratio between slip length and half the
        channel height R)
    '''
    # possibly faster alternative
    # return 2*S+1 - np.power(x,2)
    return poiseuille_scaled(y, S, S)


@dispatch(object, object, object)
def poiseuille_scaled(y, Sp, Sm):
    '''
    Analytic stationary Poiseuille-flow velocity field between two plates,
    where the coordinate system is placed in the center of the channel which
    has the diameter [-1,1].
    velocity scale: G*R**2/(2*mu)
    length scale: R (radius of the channel, where the channel has height h)

    Parameters:

    y : array_like
        Dimensionless length, coordinate system in center of the channel
    Sp : optional
         dimensionless slip length on the upper plate
         corresponds to l^+ in Matthews, 2012
    Sm : optional
         dimensionless slip length on the lower plate,
         corresponds to l^- in Matthews, 2012
    '''
    return (3.0*(Sm + Sp) + 4.0*Sm*Sp + 2.0) / (Sm + Sp + 2.0) \
           - 2.0*(Sm - Sp) / (Sp + Sm + 2.0)*y - np.power(y, 2)


class StartupCoefficients_scaled:
    '''Scaled coefficients for the scaled Navier slip startup solution'''

    def __init__(self):
        pass

    def _pairs(self, iterable):
        '''Helper function creating pairs

        Parameters:
        '''
        import itertools
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    @dispatch(object,object)
    def _kFunc(self, kn, S):
        '''Characteristic function for Navier slip startup solution for
        Poiseuille flow between two parallel plates from Matthews, 2012.

        Parameters:

        kn : dimensionless series coefficient from Matthews, 2012
        S : dimensionless slip length on both of the parallel plates,
            corresponds to l in Matthews, 2012
        '''
        return self._kFunc(kn, S, S)

    @dispatch(object,object,object)
    def _kFunc(self, kn, Sp, Sm):
        '''Characteristic function for Navier slip startup solution for
        Poiseuille flow between two parallel plates from Matthews, 2012.

        Parameters:

        kn : dimensionless series coefficient from Matthews, 2012
        Sp : dimensionless slip length on the upper plate
             corresponds to l^+ in Matthews, 2012
        Sm : dimensionless slip length on the lower plate,
             corresponds to l^- in Matthews, 2012
        '''
        return np.tan(2.0*kn) - (Sm+Sp)*kn / (Sm*Sp*np.power(kn, 2) - 1.0)


    @dispatch(object,object)
    def An(self, S, kn):
        '''A_n coefficient for Navier slip startup solution for
        Poiseuille flow between two parallel plates from Matthews, 2012.

        Parameters:

        S : dimensionless slip length on both plates, corresponds to l
            in Matthews, 2012
        kn : dimensionless series coefficient from Matthews, 2012
        '''
        return self.An(S, S, kn)

    @dispatch(object,object,object)
    def An(self, Sp, Sm, kn):
        '''A_n coefficient for Navier slip startup solution for
        Poiseuille flow between two parallel plates from Matthews, 2012.

        Parameters:

        Sp : dimensionless slip length on the upper plate
             corresponds to l^+ in Matthews, 2012
        Sm : dimensionless slip length on the lower plate,
             corresponds to l^- in Matthews, 2012
        kn : dimensionless series coefficient from Matthews, 2012
        '''
        term1 = 2 * np.power(Sm*Sp, 2) * np.power(kn, 4)
        term2 = (np.power(Sm, 2) *(Sp + 2.0) + np.power(Sp, 2)*(Sm + 2.0) ) \
                * np.power(kn, 2)
        term3 = Sm + Sp + 2.0
        denom = 2.0 * (np.power(kn*Sp, 2) + 1.0)
        lhs = (term1 + term2 + term3) / denom

        term4 = np.sin(2.0*kn) * ( 4.0*Sm*Sp*(Sm + 1.0)/(Sm + Sp + 2.0) \
                + (2*Sm*(Sm+Sp) - 4.0)/ (np.power(kn,2) * (Sm + Sp + 2.0)) )
        term5 = (4.0*Sm*(Sm + Sp + 1.0) + 4.0*Sp) / (Sm + Sp + 2.0) \
                + 2.0 / np.power(kn, 2)
        rhs = 2.0/np.power(kn,3) + term4 - np.cos(2*kn) / kn * term5
        # NOTE: for Sp = Sm =: S the above should reduce to:
        # S=Sp
        # factor= 2.0 / ( np.power(kn,3)*(np.power(S*kn,2) + S + 1.0) )
        # term1 = kn*np.sin(2.0*kn)*(np.power(S*kn,2) + S - 1.0)
        # term2 = - np.cos(2.0*kn)* (2*S*np.power(kn,2) + 1.0)
        # return factor*(1.0 + term1 + term2)
        return rhs/lhs


    @dispatch(object,int,float,float)
    def Kn(self, S, N, tol, eps):
        '''List of K_n coefficients in the Navier slip startup solution

        Parameters:

        S : ratio of slip length to radius (half diameter of the channel)
        N : number of coefficients to be computed
        tol : tolerance for the bisection method
        eps : offset from singularities for the bisection interval
        '''
        return self.Kn(S, S, N, tol, eps)

    @dispatch(object,object,int,float,float)
    def Kn(self, Sp, Sm, N, tol, eps):
        '''Computes list of K_n coefficients in the transient solution for the
        channel flow with Navier slip boundary conditions

        Parameters:

        Sp : dimensionless slip length on the upper plate
             corresponds to l^+ in Matthews, 2012
        Sm : dimensionless slip length on the lower plate,
             corresponds to l^- in Matthews, 2012
        N : number of coefficients to be computed
        tol : tolerance for the bisection method
        eps : offset from singularities for the bisection interval

        Returns: 
        
        Numpy array of nondimensional coefficients kn for the transient solution of 
        the channel flow between two parallel plates.
        '''

        # distinguish no slip case 0.0,
        # note hat 0.0 is represented exactly as a floating point number
        if Sp == 0.0 and Sm == 0.0:
            # analytic solution
            return np.arange(1, N+1) * np.pi/2.0

#        if (Sp == 0.0 and Sm != 0.0) or (Sp != 0.0 and Sm == 0.0):
#            errorMsg = "Sp==0 and Sm !=0 and vice versa are not covered"
#            raise ValueError(errorMsg)

        # set up different intervals for slip case
        n = np.arange(0, N+1, 1)
        Phi = list(np.array( np.pi*(2.0*n + 1.0) / 4.0 ))

        # phi_s not applicable if either Sp or Sm is zero as the nature of the
        # char. equation changes (k^2 term is gone in this case)
        if Sp > 0 and Sm > 0:
            phi_s = 1.0 / np.sqrt(Sm*Sp)
            # put phi_s into Phi if it is located within the first N intervals
            if phi_s < Phi[-1]: 
                n_s =  np.floor((4.0 / (np.pi * np.sqrt(Sm*Sp)) -1.0) / 2.0)  + 1.0
                Phi.insert(int(n_s), phi_s)
                Phi = np.delete(Phi, -1, 0)

        roots = np.array([])
        for (cur,nex) in self._pairs(Phi):
            new_root = bisect(self._kFunc, \
                        cur + eps, \
                        nex - eps, \
                        xtol=tol, \
                        rtol=tol, \
                        maxiter=1000, \
                        args=(Sp,Sm))
            roots = np.append(roots, new_root)
        return roots

@dispatch(object,object,float,int)
def navierSlip_scaled(t, y, S, N=10):
    '''Analytic start-up Poiseuille-flow velocity between two plates with
    Navier slip boundary condition on both walls. Zero velocity at t=0.

    Parameters:

    The coordinate system is located in the center of the channel.
    t : scaled time
    y : scaled coordinate in the channel (cross section)
    S : dimensionless slip length on both plates, corresponds to l
        in Matthews, 2012
    N (optional): number of terms for the series solution
    '''
    return navierSlip_scaled(t, y, S, S, N)

@dispatch(object,object,float,float)
def navierSlip_scaled(t, y, Sp, Sm):
    return navierSlip_scaled(t, y, Sp, Sm, 10, 1e-8, 1e-8)

@dispatch(object,object,float,float,int)
def navierSlip_scaled(t, y, Sp, Sm, N):
    return navierSlip_scaled(t, y, Sp, Sm, N, 1e-8, 1e-8)

@dispatch(object,object,float,float,int,float,float)
def navierSlip_scaled(t, y, Sp, Sm, N, tol, eps):
    '''Analytic start-up Poiseuille-flow velocity between two plates with
    Navier slip boundary condition on both walls. Zero velocity at t=0.

    Parameters:

    The coordinate system is located in the center of the channel.
    t : scaled time
    y : scaled coordinate in the channel (cross section)
    Sp : dimensionless slip length on the upper plate
         corresponds to l^+ in Matthews, 2012
    Sm : dimensionless slip length on the lower plate,
         corresponds to l^- in Matthews, 2012
    N (optional): number of terms for the series solution
    '''
    coeffs = StartupCoefficients_scaled()
    Kn = coeffs.Kn(Sp, Sm, N, tol, eps)
    An = coeffs.An(Sp, Sm, Kn) # depends on Kn

    series = 0
    for kn, an in zip(Kn, An):
        sinTerm = np.sin( kn*(y + 1.0) )
        cosTerm = Sm*kn * np.cos( kn*(y + 1.0) )
        expTerm = np.exp(-np.power(kn,2)*t)
        series += an * (sinTerm + cosTerm)*expTerm
    return poiseuille_scaled(y, Sp, Sm) - series


class StartupCoefficients:
    '''Class computing the coefficients for the Navier slip startup solution
    '''
    def __init__(self):
        pass

    def _pairs(self, iterable):
        '''Helper function creating pairs
        Parameters:

        '''
        import itertools
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def _kFunc(self, k, L, R):
        '''Function passed to the bisection method.'''
        return np.tan(2.0*k)-2.0*k*L*R/(L*L*np.power(k,2)-R*R)

    def An(self, L, R, kn):
        '''A_n coefficient in the Navier slip startup solution

        Parameters:

        L : slip length
        R : half of the channel height
        kn : K_n coefficients from the Navier slip startup solution
        '''
        l = L/R
        factor = 2.0 / ( np.power(kn,3)*(np.power(kn,2)*l*l + l + 1.0) )
        term1  = kn*np.sin(2.0*kn)*(l*l*np.power(kn,2) + l-1)
        term2  = - np.cos(2.0*kn)* (2.0*l*np.power(kn,2) + 1.0)
        return factor*(1.0 + term1 + term2)

    def Kn(self, L, R, N, eps=1e-8):
        '''K_n coefficient in the Navier slip startup solution

        Parameters:

        L : slip length
        R : half of the channel height
        eps : offset for the bisection method
        '''

        n=np.arange(0, N, 1)
        if L == 0:
            Phi=np.sort(np.array(np.pi*(2*n+1)/4.0))
        else:
            Phi=np.sort(np.concatenate((np.array([R/L]), \
                        np.array(np.pi*(2*n+1)/4.0))))

        roots=np.array([])
        for (cur,nex) in self._pairs(Phi):
            roots=np.append(roots, \
                    optimize.bisect(self._kFunc,cur+eps,nex-eps,args=(L,R)))
        return roots

# stationary flow between two plates (no slip and Navier slip)
def poiseuille(x, mu, H, G, L=0):
    '''Analytic Poiseuille-flow velocity field between two plates

    The coordinate system is located in the center of the channel
    x : coordinate in the channel in m (cross section)
    mu : dynamic viscosity in Pa s
    H : diameter (height) of the channel in m
    G : pressure difference in flow direction in Pa/m
    L : slip length (default is no slip with L=0)
    '''
    return G*H/(2*mu) * ( (4*L + H)/4 - np.power(y,2)/H)

# Startup flow between two plates
def noSlip(t, x, mu, H, G, N=10):
    '''Analytic start-up Poiseuille-flow velocity between two plates with
    no slip boundary condition on both walls. Zero velocity at t=0.

    The coordinate system is located in the center of the channel.
    t : time in s
    x : coordinate in the channel in m (cross section)
    mu : dynamic viscosity in Pa s
    H : diameter (height) of the channel in m
    G : pressure difference in flow direction in Pa/m
    N : number of terms for the series solution
    '''
    res = - 1.0/(2.0*mu) * G*H*H * x/H *(1.0-x/H) # cancel H?
    series=0
    for n in range(1,N):
        series += (1-np.power(-1,n))/ (n*n*n) *np.sin(n*np.pi*x/H) \
                  *np.exp(-n*n*np.pi*np.pi*mu/rho*t/(H*H))
    return res + 1.0/mu *G*H*H*2.0/(np.power(np.pi,3)) * series

def navierSlip(t, x, mu, rho, R, G, N=10, L=0):
    '''Analytic start-up Poiseuille-flow velocity between two plates with
    Navier slip boundary condition on both walls. Zero velocity at t=0.

    Parameters:

    t : time in s
    x : coordinate in the channel in m (cross section)
    mu : dynamic viscosity in Pa s
    H : height of the channel in m
    G : pressure difference in flow direction in Pa/m
    N : number of terms for the series solution
    L : slip length in m
    '''
    coeffs = StartupCoefficients()
    Kn = coeffs.Kn(L, R, N)
    An = coeffs.An(L, R, Kn) # depends on Kn
    series = 0
    for kn, an in zip(Kn, An):
        sinTerm = np.sin(kn*(x/R+1.0))
        cosTerm = L/R * kn * np.cos(kn*(x/R+1.0))
        expTerm = np.exp(-np.power(kn,2)*mu*t/(rho*R*R))
        series += an * (sinTerm + cosTerm)*expTerm
    return G*R*R/(2.0*mu) * (2.0*L/R +1.0 -np.power(x/R,2) - series)


if __name__ == "__main__":
    # example application for scaled problem
    t = 0.01
    y_ana = np.linspace(-1, 1, 100)
    s = 0.1

    N = 10
    vx = navierSlip_scaled(t, y_ana, s, N)
    print("vx: " + str(vx))
