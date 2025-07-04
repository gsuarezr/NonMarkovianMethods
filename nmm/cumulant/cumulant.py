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


class csolve:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, cython=False, limit=50,
                 matsubara=True,ls=False):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.limit = limit
        self.dtype = Hsys.dtype
        self.ls=ls

        if isinstance(Hsys, qutip_Qobj):
            self._qutip = True
        else:
            self._qutip = False
        if cython:
            self.baths = [bath_csolve(b.T, eps, b.coupling, b.cutoff, b.label)
                          for b in baths]
        else:
            self.baths = baths
        self.Qs = Qs
        self.cython = cython
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
    
    def gamma_fa(self, bath, w, w1, t):
        r"""
        It describes the decay rates for the Filtered Approximation of the
        cumulant equation

        $$\gamma(\omega,\omega^\prime,t)= 2\pi t e^{i \frac{\omega^\prime
        -\omega}{2}t}\mathrm{sinc} \left(\frac{\omega^\prime-\omega}{2}t\right)
         \left(J(\omega^\prime) (n(\omega^\prime)+1)J(\omega) (n(\omega)+1)
         \right)^{\frac{1}{2}}$$

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
        var = (2 * t * np.exp(1j * (w1 - w) * t / 2)
               * np.sinc((w1 - w) * t / (2 * np.pi))
               * np.sqrt(bath.spectral_density(w1) * (self.bose(w1,bath) + 1))
               * np.sqrt(bath.spectral_density(w) * (self.bose(w,bath) + 1)))
        return var

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
        if isinstance(t, list):
            t = np.array(t)
        if approximated:
            return self.gamma_fa(bath, w, w1, t)
        if self.matsubara:
            if w == w1:
                return self.decayww(bath, w, t)
            else:
                return self.decayww2(bath, w, w1, t)
        if self.cython:
            return bath.gamma(np.real(w), np.real(w1), t, limit=self.limit)

        else:
            integrals = quad_vec(
                self._gamma_,
                0,
                np.inf,
                args=(bath, w, w1, t),
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk21"
            )[0]
            return t*t*integrals

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

    def jump_operators(self, Q):
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
                if isinstance(values,qutip_Qobj):
                    dictrem[keys] = values.to("CSR")
                else:
                    dictrem[keys] = values
        return dictrem

    def decays(self, combinations, bath, approximated):
        rates = {}
        done = []
        for i in tqdm(combinations, desc='Calculating Integrals ...',
                      dynamic_ncols=True):
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.gamma_gen(bath, i[0], i[1], self.t,
                                          approximated)
        return rates

    def matrix_form(self, jumps, combinations):
        matrixform = {}
        lsform={}
        for i in tqdm(
                combinations, desc='Calculating time independent matrices...',
                dynamic_ncols=True):
            ada=jumps[i[0]].dag() * jumps[i[1]]
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                 (spre(ada) +spost(ada))))
            lsform[i]= -1j*(spre(ada)-spost(ada))

        return matrixform,lsform

    def generator(self, approximated=False):
        generators = []
        for Q, bath in zip(self.Qs, self.baths):
            jumps = self.jump_operators(Q)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            rates = self.decays(combinations, bath, approximated)
            matrices,lsform = self.matrix_form(jumps, combinations)
            if self.ls is False:
                superop = sum(
                    (rates[i] * np.array(matrices[i])
                    for i in tqdm(
                        combinations,
                        desc="Calculating time dependent generators")))
            else:
                LS= self.LS(combinations,bath,self.t)            
                superop = sum(
                    (LS[i]*np.array(lsform[i])+rates[i] * np.array(matrices[i])
                    for i in tqdm(
                        combinations,
                        desc="Calculating time dependent generators")))
            generators.extend(superop)
            del superop
        self.generators = self._reformat(generators)

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

    def evolution(self, rho0, approximated=False):
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
        self.generator(approximated)
        states = [
            (i).expm()(rho0)
            for k,i in tqdm(
                enumerate(self.generators),
                desc='Computing Exponential of Generators . . . .')]  # this counts time incorrectly
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
        t_array = np.asarray(t)
        result = self._decayww2(bath, w,w1, t_array)
        zero_indices = np.where(t_array == 0)
        result[zero_indices] = 0
        return result

    def decayww(self, bath, w, t):
        t_array = np.asarray(t)
        result = self._decayww(bath, w, t_array)
        zero_indices = np.where(t_array == 0)
        result[zero_indices] = 0
        return result

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
    csolve,
    csolve._tree_flatten,
    csolve._tree_unflatten)
