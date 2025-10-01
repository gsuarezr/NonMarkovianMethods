import numpy as np
from scipy.integrate import quad_vec
import jax.numpy as jnp
from qutip import spre as qutip_spre
from qutip import spost as qutip_spost
from qutip import Qobj as qutip_Qobj
from nmm.utils.utils import spre as jax_spre
from nmm.utils.utils import spost as jax_spost
from nmm.utils.utils import Qobj as jax_Qobj
import itertools
from collections import defaultdict
from multipledispatch import dispatch
from scipy.integrate import solve_ivp
from jax import tree_util
from scipy.interpolate import interp1d
import time
from scipy.sparse import csr_matrix
from nmm.utils.generators import GKLS
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


class redfield(GKLS):
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, matsubara=True,
                 ls=True,picture="I"):
        super().__init__(Hsys,t,Qs)
        self.eps = eps
        self.ls=ls
        self.picture = picture
        if isinstance(Hsys, qutip_Qobj):
            self._qutip = True
        else:
            self._qutip = False
        self.baths = baths
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
        return 1 / (np.exp(w / bath.T)-1)
    def _gamma(self, nu, bath, w, w1, t):
        r"""
        It describes the Integrand of the decay rates of the Redfield equation
        for bosonic baths

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
        var = 1j*bath.spectral_density(nu)*self.bose(
            nu,bath)*(1-np.exp(1j*t*(w+nu)))/(w+nu)
        var += 1j*bath.spectral_density(nu)*(
            self.bose(nu,bath)+1)*(1-np.exp(1j*t*(w-nu)))/(w-nu)
        var2 = 1j*bath.spectral_density(nu)*self.bose(
            nu,bath)*(1-np.exp(1j*t*(w1+nu)))/(w1+nu)
        var2 += 1j*bath.spectral_density(nu)*(
            self.bose(nu,bath)+1)*(1-np.exp(1j*t*(w1-nu)))/(w1-nu)
        return (var2 + np.conjugate(var))*self._mul

    def _gamma_gen(self, bath, w, w1, t):
        r"""
        It describes the the decay rates of the Redfield equation
        for bosonic baths


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
        if self.matsubara:
            return self.decayww2(bath, w, w1, t)

        integrals = quad_vec(
            self._gamma,
            0,
            np.inf,
            args=(bath, w, w1, t),
            points=[-w, -w1, w, w1],
            epsrel=self.eps,
            epsabs=1e-9,              
            quadrature="gk21"
        )[0]
        if self.picture=="I":
            return integrals *np.exp(1j*(w-w1)*t) # remove rotation so that it is in schrodinger
        else:
            return integrals

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

    def decays(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._gamma_gen(bath, i[1], i[0], t)
        return rates

    def matrix_form(self, jumps, combinations):
        matrixform = {}
        lsform= {}
        for i in combinations:
            ada=jumps[i[0]].dag()*jumps[i[1]]
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                (spre(ada) +spost(ada))))
            lsform[i]= 1j*(spre(ada)-spost(ada))
        return matrixform,lsform

    def prepare_interpolated_generators(self):
        print("Started integration and Generator Calculations")
        start=time.time()
        generators = [self.generator(i) for i in self.t]
        print("Finished integration and Generator Calculations")
        end=time.time()
        print(f"Computation Time:{end-start}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        start=time.time()
        print("Started interpolation")
        self.interpolated_generators = [
            interp1d(
                self.t, generators[:, i],
                kind='linear', bounds_error=False,
                fill_value="extrapolate")
            for i in range(generators.shape[1])]
        self.generator_shape = self.generators[0].shape
        print("Finished interpolation")
        end=time.time()
        print(f"Computation Time:{end-start}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    def interpolated_generator(self, t):
        if self.interpolated_generators is None:
            raise ValueError(
                "Interpolated generators not prepared. Call prepare_interpolated_generators() first.")

        interpolated = np.array([interp(t)
                                for interp in self.interpolated_generators])
        return interpolated.reshape(self.generator_shape)

    def generator(self,t):
        generators = []
        for Q, bath in zip(self.Qs, self.baths):
            jumps = self.jump_operators(Q,t)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            matrices,lsform = self.matrix_form(jumps, combinations)
            decays = self.decays(combinations, bath, t)   
            superop = []
            if self._qutip:
                if self.ls is True:
                    LS= self.LS(combinations,bath,t)       
                    gen = (LS[i]*np.array(lsform[i]) + np.array(matrices[i])*decays[i] for i in combinations)
                else:
                    gen = (np.array(matrices[i])*decays[i] for i in combinations)
            else:
                gen = (matrices[i]*(decays[i]).item() for i in combinations)
            superop.append(sum(gen))
            generators.extend(superop)
            del gen
            del matrices
            del decays
        generate = sum(generators)
        if self.picture=="I":
            return generate
        else:
            try:
                return generate+1j*(spre(self.Hsys(t))-spost(self.Hsys(t))) 
            except:
                return generate+1j*(spre(self.Hsys)-spost(self.Hsys))  
    def evolution(self, rho0, method="RK45"):
        r"""
        This function computes the evolution of the state $\rho(0)$

        Parameters
        ----------

        rho0 : numpy.ndarray or qutip.Qobj
            The initial state of the quantum system under consideration.

        Method : string
            The method to be used by scipy's solve_ivp

        Returns
        -------
        list
            a list containing all of the density matrices, at all timesteps of
            the evolution
        """
        y0 = rho0.full().flatten()
        y0 = np.array(y0).astype(np.complex128)
        #self.prepare_interpolated_generators()

        def f(t, y):
            if np.isscalar(t):
                return (csr_matrix(self.generator(t).full()) @ y)
            else:
                return np.array([
                    csr_matrix(self.generator(ti).full()) @ yi 
                    for ti, yi in zip(t, y.T)
                ]).T
        start=time.time()
        print("Started Solving the differential equation")
        result = solve_ivp(f, [0, self.t[-1]],
                           y0,
                           t_eval=self.t, method=method,vectorized=True)
        n = self.Hsys.shape[0]
        states = [result.y[:, i].reshape(n, n)
                  for i in range(len(self.t))]
        print("Finished Solving the differential equation")
        end=time.time()
        print(f"Computation Time:{end-start}")
        return states

    def _decayww2(self,bath, w, w1, t):
        cks=np.array([i.coefficient for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        result=[]
        for i in range(len(cks)):
            term1=cks[i]/(vks[i]-1j*w1)
            term2=np.conjugate(cks[i])/(np.conjugate(vks[i])+1j*w)
            term1*=(1-np.exp(-(vks[i]-1j*w1)*t))
            term2*=(1-np.exp(-(np.conjugate(vks[i])+1j*w)*t))
            result.append(term1+term2)
        if self.picture=="I":
            return sum(result) *np.exp(1j*(w-w1)*t)
        else:
            return sum(result)#now in schrodinger

    def decayww2(self, bath, w, w1, t):
        return self._decayww2(bath, w1, w, t).conj()

    
    def _LS(self, bath, w,w1, t):
        cks=np.array([i.coefficient for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        result=[]
        for i in range(len(cks)):
            term1=cks[i]/(vks[i]-1j*w1)
            term2=np.conjugate(cks[i])/(np.conjugate(vks[i])+1j*w)
            term1*=(1-np.exp(-(vks[i]-1j*w1)*t))
            term2*=(1-np.exp(-(np.conjugate(vks[i])+1j*w)*t))
            result.append(term1-term2)
        if self.picture=="I":
            return sum(result)/2j *np.exp(1j*(w-w1)*t)
        else:
            return sum(result)/2j#now in schrodinger
    def LS(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._LS(bath, i[0], i[1], t).conj()
        return rates
tree_util.register_pytree_node(
    redfield,
    redfield._tree_flatten,
    redfield._tree_unflatten)


