from cython.cimports.libc.math cimport sin,cos,exp,sqrt,pi,INFINITY,abs
import cython
from scipy.integrate import quad
import numpy as np

cdef sinc(x:cython.float):
    if x==0:
        return 1
    return sin(pi*x)/(pi*x)

cdef extern from "complex.h":
    float complex I

cdef class bath_csolve:
    cdef cython.float T
    cdef public cython.float tol 
    cdef cython.float coupling 
    cdef cython.float cutoff
    cdef cython.str label
    def __init__(self,  T,tol,coupling,cutoff,label):
        self.T=T
        self.tol=tol
        self.coupling=coupling
        self.cutoff=cutoff
        self.label=label

    cpdef spectral_density(self, w:cython.float):
        if self.label=="ohmic":
            return self.coupling*w*exp(-abs(w)/self.cutoff)
        if self.label=="overdamped":
            return 2*w*self.coupling*self.cutoff/(self.cutoff**2 + w**2)
    cpdef bose(self, nu):
        if (self.T == 0.0) or (nu==0.0):
            return 0.0
        return exp(-nu / self.T) / (1-exp(-nu / self.T))
    cpdef gammafa(self, w:cython.float, w1:cython.float, t:cython.float):
        var = (2 * pi * t *(cos((w - w1) / 2 * t) + I * sin((w - w1) / 2 * t))            
               * sinc((w1 - w) * t / 2 )
               * sqrt(self.spectral_density(w1) * (self.bose(w1) + 1))
               * sqrt(self.spectral_density(w) * (self.bose(w) + 1)))
        return var
    cdef _gamma(self, nu:cython.float, w:cython.float, w1:cython.float, t:cython.float):
        expc=cos((w - w1) / 2 * t) + I *sin((w - w1) / 2 * t)         
        var = t*t*(expc
           * self.spectral_density(nu)
            * (sinc((w - nu) / (2*pi ) * t)
               * sinc((w1 - nu) / (2*pi) * t))
            * (self.bose(nu) + 1)
        )
        var += t*t*(expc
            * self.spectral_density(nu)
            * (sinc((w + nu) / (2*pi ) * t)
               * sinc((w1 + nu) / (2*pi) * t))
            * self.bose(nu)
        )
        return var
    cdef _gamma_int(self, w:cython.float, w1:cython.float, t,limit:int=500):
        # Gauss_kron Adapt might make this faster
        return [quad(self._gamma,0,INFINITY,args=(w, w1, i),epsabs=self.tol,
                limit=limit,epsrel=self.tol,complex_func=True)[0] for i in t]
    cpdef gamma(self, w:cython.float, w1:cython.float, t,limit:int=50):
        # Gauss_kron Adapt might make this faster
        return self._gamma_int(w, w1, t,limit)
    cpdef gamma_adapt(self, w:cython.float, w1:cython.float, t):
        # THis is broken but might provide a significant speed up 
        return quad_gauss(self._gamma,0,INFINITY,tol=self.tol,args=(w,w1,t))


import sys
import numpy as np






def _quadrature_gk(a, b, f, norm_func, x ,w, v):
    """
    Generic Gauss-Kronrod quadrature
    """

    fv = [0.0]*len(x)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Gauss-Kronrod
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        z=c + h*x[i]
        ff = f(z)
        fv[i] = ff

        vv = v[i]

        # \int f(x)
        s_k += vv * ff
        # \int |f(x)|
        s_k_abs += vv * abs(ff)

    # Gauss
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2*i + 1]

    # Quadrature of abs-deviation from average
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        # \int |f(x) - y0|
        s_k_dabs += v[i] * abs(fv[i] - y0)

    # Use similar error estimation as quadpack
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        zz=200 * err / dabs
        err = dabs * min(1.0, sqrt(zz)*zz)

    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))

    if round_err > sys.float_info.min:
        err = max(err, round_err)

    return h * s_k, err, round_err


def _quadrature_gk15(a, b, f, norm_func):
    """
    Gauss-Kronrod 15 quadrature with error estimate
    """
    # Gauss-Kronrod points
    x = (0.991455371120812639206854697526329,
         0.949107912342758524526189684047851,
         0.864864423359769072789712788640926,
         0.741531185599394439863864773280788,
         0.586087235467691130294144838258730,
         0.405845151377397166906606412076961,
         0.207784955007898467600689403773245,
         0.000000000000000000000000000000000,
         -0.207784955007898467600689403773245,
         -0.405845151377397166906606412076961,
         -0.586087235467691130294144838258730,
         -0.741531185599394439863864773280788,
         -0.864864423359769072789712788640926,
         -0.949107912342758524526189684047851,
         -0.991455371120812639206854697526329)

    # 7-point weights
    w = (0.129484966168869693270611432679082,
         0.279705391489276667901467771423780,
         0.381830050505118944950369775488975,
         0.417959183673469387755102040816327,
         0.381830050505118944950369775488975,
         0.279705391489276667901467771423780,
         0.129484966168869693270611432679082)

    # 15-point weights
    v = (0.022935322010529224963732008058970,
         0.063092092629978553290700663189204,
         0.104790010322250183839876322541518,
         0.140653259715525918745189590510238,
         0.169004726639267902826583426598550,
         0.190350578064785409913256402421014,
         0.204432940075298892414161999234649,
         0.209482141084727828012999174891714,
         0.204432940075298892414161999234649,
         0.190350578064785409913256402421014,
         0.169004726639267902826583426598550,
         0.140653259715525918745189590510238,
         0.104790010322250183839876322541518,
         0.063092092629978553290700663189204,
         0.022935322010529224963732008058970)

    return _quadrature_gk(a, b, f, norm_func, x, w, v)


def _gauss_kron(f,a,b,tol):
    integral,err,rnd=_quadrature_gk15(a,b,f,np.linalg.norm)  
    if abs(err)<tol:
        q= integral
    else:
        c=(a+b)/2
        qa= _gauss_kron(f,a,c,tol)
        qb=_gauss_kron(f,c,b,tol)   
        q=qa+qb 
    return q
def gauss_kron(f,a,b,tol,args=()):
    func= lambda x: f(x,*args)
    if (a==0) and (b ==INFINITY):
        func= lambda x: f((1-x)/x,*args)
        b=1
    return _gauss_kron(func,a,b,tol)
def quad_gauss(f,a,b,tol,args=()):
    return gauss_kron(f,a,b,tol,args=args)
    