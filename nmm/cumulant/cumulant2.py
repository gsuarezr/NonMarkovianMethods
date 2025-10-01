import numpy as np
from scipy.integrate import quad_vec
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from qutip import spre as qutip_spre
from qutip import spost as qutip_spost
from qutip import Qobj as qutip_Qobj
from nmm.utils.utils import spre as jax_spre
from nmm.utils.utils import spost as jax_spost
from nmm.utils.utils import Qobj as jax_Qobj
import itertools
from collections import defaultdict
from nmm.cumulant.cum import bath_csolve
from multipledispatch import dispatch
import warnings
from jax import tree_util


@dispatch(qutip_Qobj)
def spre(op):
    return qutip_spre(op)


@dispatch(qutip_Qobj)
def spost(op):
    return qutip_spost(op)


@dispatch(jax_Qobj)
def spre(op):
    return jax_spre(op)


@dispatch(jax_Qobj)
def spost(op):
    return jax_spost(op)


class csolve2:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, 
                 matsubara=False,ls=False):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.dtype = Hsys.dtype
        self.ls=ls
        self.baths=baths

        if isinstance(Hsys, qutip_Qobj):
            self._qutip = True
        else:
            self._qutip = False

        self.Qs = Qs
        self.matsubara = matsubara

    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.eps,
                    self.limit, self.baths, self.dtype)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    def bose(self,w,bath):
        r"""
        It computes the Bose-Einstein distribution

        $$ n(\omega)=\frac{1}{e^{\beta \omega}-1} $$

        Parameters:
        ----------
        nu: float
            The mode at which to compute the thermal population

        Returns:
        -------
        float
            The thermal population of mode nu
        """
        if bath.T == 0:
            return 0
        if np.isclose(w, 0).all():
            return 0
        return np.exp(-w / bath.T) / (1-np.exp(-w / bath.T))
    

    def _gamma_(self, nu, bath, w, w1, t):
        r"""
        It describes the Integrand of the decay rates of the cumulant equation
        for bosonic baths

        $$\Gamma(w,w',t)=\int_{0}^{t} dt_1 \int_{0}^{t} dt_2
        e^{i (w t_1 - w' t_2)} \mathcal{C}(t_{1},t_{2})$$

        Parameters:
        ----------

        w: float or numpy.ndarray

        w1: float or numpy.ndarray

        t: float or numpy.ndarray

        Returns:
        --------
        float or numpy.ndarray
            It returns a value or array describing the decay between the levels
            with energies w and w1 at time t

        """

        self._mul = 1/np.pi
        var = (
            np.exp(1j * (w - w1) / 2 * t)
            * bath.spectral_density(nu)
            * (np.sinc((w - nu) / (2 * np.pi) * t)
               * np.sinc((w1 - nu) / (2 * np.pi) * t))
            * (self.bose(nu,bath) + 1)
        )
        var += (
            np.exp(1j * (w - w1) / 2 * t)
            * bath.spectral_density(nu)
            * (np.sinc((w + nu) / (2 * np.pi) * t)
               * np.sinc((w1 + nu) / (2 * np.pi) * t))
            * self.bose(nu,bath)
        )

        var = var*self._mul
    
        return var

    def gamma_gen(self, bath, w, w1, t, approximated=False):
        r"""
        It describes the the decay rates of the cumulant equation
        for bosonic baths

        $$\Gamma(\omega,\omega',t) = t^{2}\int_{0}^{\infty} d\omega 
        e^{i\frac{\omega-\omega'}{2} t} J(\omega) \left[ (n(\omega)+1) 
        sinc\left(\frac{(\omega-\omega)t}{2}\right)
        sinc\left(\frac{(\omega'-\omega)t}{2}\right)+ n(\omega) 
        sinc\left(\frac{(\omega+\omega)t}{2}\right) 
        sinc\left(\frac{(\omega'+\omega)t}{2}\right)   \right]$$

        Parameters
        ----------

        w : float or numpy.ndarray
        w1 : float or numpy.ndarray
        t : float or numpy.ndarray

        Returns
        -------
        float or numpy.ndarray
            It returns a value or array describing the decay between the levels
            with energies w and w1 at time t

        """
        if isinstance(t, type(jnp.array([2]))):
            t = np.array(t.tolist())
        if approximated:
            return self.gamma_fa(bath, w, w1, t)
        if self.matsubara:
            if w == w1:
                return self.decayww(bath, w, t)
            else:
                return self.decayww2(bath, w, w1, t)
        else:
            integrals = lambda t1: quad_vec(self._gamma_,0,t,
                                            args=(bath,w, w1, t1),
                                            epsabs=self.eps,
                                            epsrel=self.eps,
                                            quadrature="gk15")[0]
            integrals2=quad_vec(integrals,0,t,epsabs=self.eps,epsrel=self.eps,
                                quadrature="gk15")[0]
            return integrals2
        
    def _gamma_2(self, nu,bath, w, w1, t):
        r"""
        It describes the Integrand of the decay rates of the cumulant equation
        for bosonic baths

        $$\Gamma(w,w',t)=\int_{0}^{t} dt_1 \int_{0}^{t} dt_2
        e^{i (w t_1 - w' t_2)} \mathcal{C}(t_{1},t_{2})$$

        Parameters:
        ----------

        w: float or numpy.ndarray

        w1: float or numpy.ndarray

        t: float or numpy.ndarray

        Returns:
        --------
        float or numpy.ndarray
            It returns a value or array describing the decay between the levels
            with energies w and w1 at time t

        """
        var = bath.correlation_function(t-nu) *np.exp(2j*(np.sign(w)*self.f(nu)+ np.sign(w1)*self.f(t))) 
        return var
    def sparsify(self, vectors, tol=10):
        dims = vectors[0].dims
        new = []
        for vector in vectors:
            top = np.max(np.abs(vector.full()))
            vector = (vector/top).full().round(tol)*top
            vector = qutip_Qobj(vector).to("CSR")
            vector.dims = dims

            new.append(vector)
        return new


    def jump_operators(self, Q,t=None):
        try:
            evals, all_state = self.Hsys(t).eigenstates()
        except:
            evals, all_state = self.Hsys.eigenstates()

        N = len(all_state)
        collapse_list = []
        ws = []
        for j in range(N):
            for k in range(j + 1, N):
                Deltajk = evals[k] - evals[j]
                ws.append(Deltajk)
                collapse_list.append(
                    (
                        all_state[j]
                        * all_state[j].dag()
                        * Q
                        * all_state[k]
                        * all_state[k].dag()
                    )
                )  # emission
                ws.append(-Deltajk)
                collapse_list.append(
                    (
                        all_state[k]
                        * all_state[k].dag()
                        * Q
                        * all_state[j]
                        * all_state[j].dag()
                    )
                )  # absorption
        collapse_list.append(Q - sum(collapse_list))  # Dephasing
        ws.append(0)
        output = defaultdict(list)
        for k, key in enumerate(ws):
            output[jnp.round(key, 12).item()].append(collapse_list[k])
        eldict = {x: sum(y) for x, y in output.items()}
        dictrem = {}
        empty = 0*self.Hsys
        for keys, values in eldict.items():
            if not (values == empty):
                dictrem[keys] = values.to("CSR")
        return dictrem

    def decays(self, combinations, bath, approximated,t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.gamma_gen(bath, i[0], i[1], t,
                                          approximated)
        return rates

    def matrix_form(self, jumps, combinations):
        matrixform = {}
        lsform={}
        for i in combinations:
            ada=jumps[i[0]].dag() * jumps[i[1]]
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                 (spre(ada) +spost(ada))))
            lsform[i]= -1j*(spre(ada)-spost(ada))

        return matrixform,lsform

    def generator(self,t):
        generators = []
        for Q, bath in zip(self.Qs, self.baths):
            jumps = self.jump_operators(Q,t)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            rates = self.decays(combinations, bath, False,t)
            matrices,lsform = self.matrix_form(jumps, combinations)
            if self.ls is False:
                superop = sum(
                    (rates[i] * matrices[i]
                    for i in 
                        combinations))
            else:
                LS= self.LS(combinations,bath,t)            
                superop = sum(
                    (LS[i]*lsform[i]+rates[i] * matrices[i]
                    for i in 
                        combinations))
            generators.append(superop)
        try:
            return sum(generators)
        except:
            return sum(generators)

    def _reformat(self, generators):
        if len(generators) == len(self.t):
            return generators
        else:
            one_list_for_each_bath = [
                generators
                [i * len(self.t): (i + 1) * len(self.t)]
                for i in range(
                    0, int(
                        len(generators) / len(self.t)))]
            composed = list(map(sum, zip(*one_list_for_each_bath)))
            return composed

    def evolution(self, rho0):
        r"""
        This function computes the evolution of the state $\rho(0)$

        Parameters
        ----------

        rho0 : numpy.ndarray or qutip.Qobj
            The initial state of the quantum system under consideration.

        approximated : bool
            When False the full cumulant equation/refined weak coupling is
            computed, when True the Filtered Approximation (FA is computed),
            this greatly reduces computational time, at the expense of
            diminishing accuracy particularly for the populations of the system
            at early times.

        Returns
        -------
        list
            a list containing all of the density matrices, at all timesteps of
            the evolution
        """
        states = [
            ( self.generator(i)).expm()(rho0)
            for i in tqdm(self.t,
                desc='Computing Exponential of Generators . . . .',total=len(self.t))]  # this counts time incorrectly
        return states
    def _decayww(self,bath, w, t):
        cks=np.array([i.coefficient for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        result=[]
        for i in range(len(cks)):
            term1 =(vks[i]*t-1j*w*t-1)+np.exp(-(vks[i]-1j*w)*t)
            term1=term1*cks[i]/(vks[i]-1j*w)**2
            result.append(term1)
        return 2*np.real(sum(result))

    def _decayww2(self,bath ,w, w1, t):
        cks=np.array([i.coefficient for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        result=[]
        for i in range(len(cks)):
            a=(vks[i]-1j*w1)
            b=(vks[i]-1j*w)
            term1=cks[i]*np.exp(-b*t)/(a*b)
            term2=np.conjugate(cks[i])*np.exp(-np.conjugate(a)*t)/(np.conjugate(a)*np.conjugate(b))
            term3=cks[i]*((1/b)-(np.exp(1j*(w-w1)*t)/a))
            term4=np.conjugate(cks[i])*((1/np.conjugate(a))-(np.exp(1j*(w-w1)*t)/np.conjugate(b)))
            actual=term1+term2+(1j*(term3+term4)/(w-w1))
            result.append(actual)
        return sum(result)


    def decayww2(self, bath, w, w1, t):
        return self._decayww2(bath, w, w1, t)

    def decayww(self, bath, w, t):
        return self._decayww(bath, w, t)

    def _LS(self, bath, w,w1, t):
        if w!=w1:
            cks=np.array([i.coefficient for i in bath.exponents])
            vks=np.array([i.exponent for i in bath.exponents])
            mask = np.imag(cks) >= 0
            cks=cks[mask]
            vks=vks[mask]
            result=[]
            for i in range(len(cks)):
                a=(vks[i]-1j*w1)
                b=(vks[i]-1j*w)
                term1=cks[i]*np.exp(-b*t)/(a*b)
                term2=np.conjugate(cks[i])*np.exp(-np.conjugate(a)*t)/(np.conjugate(a)*np.conjugate(b))
                term3=cks[i]*((1/b)-(np.exp(1j*(w-w1)*t)/a))
                term4=np.conjugate(cks[i])*((1/np.conjugate(a))-(np.exp(1j*(w-w1)*t)/np.conjugate(b)))
                actual=term1-term2+(1j*(term3-term4)/(w-w1))
                result.append(actual)
            return sum(result)/2j 
        else:
            cks=np.array([i.coefficient for i in bath.exponents])
            vks=np.array([i.exponent for i in bath.exponents])
            result=[]
            for i in range(len(cks)):
                term1 =(vks[i]*t-1j*w*t-1)+np.exp(-(vks[i]-1j*w)*t)
                term1=term1*cks[i]/(vks[i]-1j*w)**2
                result.append(term1)
            return np.imag(sum(result))/2 
    def LS(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._LS(bath, i[0], i[1], t)
        return rates
tree_util.register_pytree_node(
    csolve2,
    csolve2._tree_flatten,
    csolve2._tree_unflatten)
# TODO Add Lamb-shift
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO support Tensor Networks
# Benchmark with the QuatumToolbox,jl based version
# Habilitate double precision (Maybe single is good for now)
# TODO Diffrax does not work unless one makes a pytree for Qobj apparently