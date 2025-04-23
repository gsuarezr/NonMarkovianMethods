import jax.numpy as jnp
from jax import jit,Array
from tqdm import tqdm
from nmm.utils.utils import spre 
from nmm.utils.utils import spost 
from nmm.utils.utils import Qobj 
import itertools
from collections import defaultdict
import warnings
from jax import tree_util


class jaxcsolve:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4,ls=False):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.dtype = Hsys.dtype
        self.ls=ls 
        self.baths = baths
        self.Qs = Qs


    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.eps, self.dtype,
                    self.ls,self.Qs)
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
        if jnp.isclose(w, 0).all():
            return 0
        return jnp.exp(-w / bath.T) / (1-jnp.exp(-w / bath.T))


    def gamma_gen(self, bath, w, w1, t):
        if w == w1:
            return self.decayww(bath, w, t)
        else:
            return self.decayww2(bath, w, w1, t)

    def jump_operators(self,Q):
        evals, evecs = self.Hsys.eigenstates()
        N = evecs.shape[1]
        zero=jnp.zeros_like(self.Hsys)
        jumps = []
        ws= []
        for j in range(N):
            for k in range(j + 1, N):
                omega = evals[k] - evals[j]
                state_j = evecs[:, j:j+1]  
                state_k = evecs[:, k:k+1]

                emission = state_j * (state_j.dag() * Q * state_k) * state_k.dag()
                jumps.append(emission)
                ws.append(omega)

                absorption = state_k * (state_k.dag()* Q * state_j) * state_j.dag()
                jumps.append(absorption)
                ws.append(-omega)
        result=Qobj(zero)
        for op in jumps:
            result+=op
        dephasing=Q-result
        jumps.append(dephasing)
        ws.append(0*omega) #to keep right type and not weak

        return jumps,ws

    def jump_calc(self,Q):
        jumps,ws =self.jump_operators(Q)
        output = defaultdict(list)
        for k, key in enumerate(ws):
            output[jnp.round(key, 12).item()].append(jumps[k])
            eldict = {x: sum(y) for x, y in output.items()}
            dictrem = {}
            empty = self.Hsys*0
            for keys, values in eldict.items():
                if not (values == empty):
                    dictrem[keys] = values
        return dictrem
    def decays(self, combinations, bath,t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = jnp.conjugate(rates[j])
            else:
                rates[i] = self.gamma_gen(bath, i[0], i[1], t)
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
        for Q, bath in zip(self.Qs, self.baths):
            jumps=self.jump_calc(Q)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            rates = self.decays(combinations, bath,t)
            matrices,lsform = self.matrix_form(jumps, combinations)
            if self.ls is False:
                superop = sum(
                    (matrices[i]*rates[i] 
                    for i in 
                        combinations),start=spre(Q)*0)
            else:
                LS= self.LS(combinations,bath,t)            
                superop = sum(
                    LS[i]*lsform[i]+rates[i] *matrices[i]
                    for i in combinations)
            return superop
  

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
            self.generator(i).expm()(rho0)
            for k,i in tqdm(
                enumerate(self.t),
                desc='Computing Exponential of Generators . . . .')]  # this counts time incorrectly
        return states
    
    def _decayww(self, bath, w, t):
        cks=jnp.array([i.coefficient for i in bath.exponents])
        vks=jnp.array([i.exponent for i in bath.exponents])

        result=[]
        for i in range(len(cks)):
            term1 =(vks[i]*t-1j*w*t-1)+jnp.exp(-(vks[i]-1j*w)*t)
            term1=term1*cks[i]/(vks[i]-1j*w)**2
            result.append(term1)
        return 2*jnp.real(sum(result))

    def _decayww2(self, bath ,w, w1, t):
        cks=jnp.array([i.coefficient for i in bath.exponents])
        vks=jnp.array([i.exponent for i in bath.exponents])
        result=[]
        for i in range(len(cks)):
            a=(vks[i]-1j*w1)
            b=(vks[i]-1j*w)
            term1=cks[i]*jnp.exp(-b*t)/(a*b)
            term2=jnp.conjugate(cks[i])*jnp.exp(-jnp.conjugate(a)*t)/(jnp.conjugate(a)*jnp.conjugate(b))
            term3=cks[i]*((1/b)-(jnp.exp(1j*(w-w1)*t)/a))
            term4=jnp.conjugate(cks[i])*((1/jnp.conjugate(a))-(jnp.exp(1j*(w-w1)*t)/jnp.conjugate(b)))
            actual=term1+term2+(1j*(term3+term4)/(w-w1))
            result.append(actual)
        return sum(result)


    def decayww2(self,bath, w, w1, t):
        return self._decayww2( bath, w, w1, t)

    def decayww(self, bath, w, t):
        return self._decayww(bath, w, t)

    def _LS(self,bath, w,w1, t):
        cks=jnp.array([i.coefficient for i in bath.exponents])
        vks=jnp.array([i.exponent for i in bath.exponents])
        if w!=w1:
            result=[]
            for i in range(len(cks)):
                a=(vks[i]-1j*w1)
                b=(vks[i]-1j*w)
                term1=cks[i]*jnp.exp(-b*t)/(a*b)
                term2=jnp.conjugate(cks[i])*jnp.exp(-jnp.conjugate(a)*t)/(jnp.conjugate(a)*jnp.conjugate(b))
                term3=cks[i]*((1/b)-(jnp.exp(1j*(w-w1)*t)/a))
                term4=jnp.conjugate(cks[i])*((1/jnp.conjugate(a))-(jnp.exp(1j*(w-w1)*t)/jnp.conjugate(b)))
                actual=term1-term2+(1j*(term3-term4)/(w-w1))
                result.append(actual)
            return sum(result)/2j
        else:
            result=[]
            for i in range(len(cks)):
                term1 =(vks[i]*t-1j*w*t-1)+jnp.exp(-(vks[i]-1j*w)*t)
                term1=term1*cks[i]/(vks[i]-1j*w)**2
                result.append(term1)
            return jnp.imag(sum(result))/2
    def LS(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = jnp.conjugate(rates[j])
            else:
                rates[i] = 2*self._LS(bath, i[0], i[1], t)
        return rates
tree_util.register_pytree_node(
    jaxcsolve,
    jaxcsolve._tree_flatten,
    jaxcsolve._tree_unflatten)
# TODO Add Lamb-shift
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO support Tensor Networks
# Benchmark with the QuatumToolbox,jl based version
# Habilitate double precision (Maybe single is good for now)
# TODO Diffrax does not work unless one makes a pytree for Qobj apparently
