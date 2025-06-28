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
from jax import  tree_util
from scipy.integrate import solve_ivp


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


class crsolve:
    def __init__(self, Hsys, t, baths,Qs, eps=1e-4,limit=50,
                 matsubara=False,ls=False):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.limit=limit
        self.dtype = Hsys.dtype
        self.ls =ls
        
        
        if isinstance(Hsys,qutip_Qobj):
            self._qutip=True
        else:
            self._qutip=False
 
        self.baths=baths
        self.Qs = Qs
        self.matsubara=matsubara
    def _tree_flatten(self):
        children=(self.Hsys,self.t,self.eps,self.limit,self.baths,self.dtype)
        aux_data={}
        return (children,aux_data)
    @classmethod
    def _tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    def _gamma_(self, nu,bath, w, w1, t):
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
        var = bath.correlation_function(t-nu) *np.exp(1j*w1*t-1j*w*nu) 
        return var

    def gamma_gen(self, bath ,w, w1, t, approximated=False):
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
        if isinstance(t,type(jnp.array([2]))):
            t=np.array(t.tolist())

        if self.matsubara:
                return self.decayww2(bath,w,w1,t)

        else:
            integrals = lambda t1: quad_vec(
                self._gamma_,
                0,
                t,
                args=(bath,w, w1, t1),
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk15"
            )[0]
            return integrals(t)
            
    def jump_operators(self,Q):
        #try:
            #evals, all_state = self.Hsys(t).eigenstates()
        #except:
        evals, all_state = self.Hsys.eigenstates()
        N=len(all_state)
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
        for k,key in enumerate(ws):
            output[jnp.round(key,12).item()].append(collapse_list[k])
        eldict={x:sum(y) for x, y in output.items()}
        dictrem = {}
        empty =0*self.Hsys
        for keys, values in eldict.items():
            if not (values == empty):
                dictrem[keys] = values
        return dictrem
        
    def decays(self,combinations,bath,t=None):
        if t is None:
            t=self.t
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.gamma_gen(bath,i[0], i[1], t)
        return rates
    def matrix_form(self,jumps,combinations):   
        matrixform={}  
        lsform={}     
        for i in combinations:
            # ada=jumps[i[0]].dag() * jumps[i[1]]
            # matrixform[i] = (
            #     spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
            #     (0.5 *
            #      (spre(ada) +spost(ada)))).full()
            #     # matrixform[i]= (jumps[i[1]].full() @x@jumps[i[0]].dag().full()
            #     #                 -jumps[i[0]].dag().full() @ jumps[i[1]].full()@x/2
            #     #                 -x/2@jumps[i[0]].dag().full() @ jumps[i[1]].full())
            # lsform[i]=(spre(ada)-spost(ada)).full()
            #matrixform[i] =-( spre(jumps[i[1]].dag()*jumps[i[0]]) 
            #                - spre(jumps[i[0]])* spost(jumps[i[1]].dag()) )
            #lsform[i] =   -( spost(jumps[i[0]].dag()*jumps[i[1]]) 
            #                - spre(jumps[i[1]])* spost(jumps[i[0]].dag()) )
            uno=jumps[i[1]].dag()*jumps[i[0]]
            dos=jumps[i[0]].dag()*jumps[i[1]]
            matrixform[i] =-( spre(uno)+spost(dos) 
                            - spre(jumps[i[0]])* spost(jumps[i[1]].dag()) 
                            - spre(jumps[i[1]])* spost(jumps[i[0]].dag()) )
            if self.ls==True:
                lsform[i]=-( spre(uno)-spost(dos) 
                            - spre(jumps[i[0]])* spost(jumps[i[1]].dag()) 
                            + spre(jumps[i[1]])* spost(jumps[i[0]].dag()) )
            else:
                lsform[i]=0
        return matrixform,lsform
    
    def generator(self,t):
        for Q,bath in zip(self.Qs,self.baths):
            jumps=self.jump_operators(Q)
            ws=list(jumps.keys())
            combinations=list(itertools.product(ws, ws))
            matrices,lsform=self.matrix_form(jumps,combinations)
            # uno,dos,tres=self.matrix_form(jumps,combinations)
            a=[]
            for i in combinations:
                gamma=self.gamma_gen(bath,i[0],i[1],t)
                # f=1j*(self.gamma_gen(bath,i[0],i[1],t)* matrices[i] + self.gamma_gen(bath,i[0],i[1],t).conj()* lsform[i])/2
                working=gamma.real* matrices[i] + 1j*gamma.imag* lsform[i]
                #working=self.gamma_gen(bath,i[0],i[1],t)* matrices[i]
                # ff=(self.gamma_gen(bath,i[0],i[1],t).conj()+self.gamma_gen(bath,i[1],i[0],t))*uno
                # pre=self.gamma_gen(bath,i[0],i[1],t).conj()*tres
                # post=self.gamma_gen(bath,i[1],i[0],t)*dos
                a.append( working )  
            kk=  sum(a)
            # change this for first plus HC, check if still correct results, in transport and damped jc
            # TODO: make tests 
            try:
                return 1j*(spre(self.Hsys(t))-spost(self.Hsys(t))) +kk.conj()
            except:
                return kk.conj()

    def evolution(self, rho0, method="RK45"):
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
        # be easility jitted
        # try:
        #     y0=rho0.data.flatten()
        #     y0=np.array(y0).astype(np.complex128)
        #     f=lambda t,y: np.array(self.generator(t).data)@np.array(y) #maybe this can
        # except:
        y0 = rho0.full().flatten()
        y0 = np.array(y0).astype(np.complex128)
        # def f(t, y): return self.generator(
        #     t).full()@np.array(y)  # maybe this can

        def f(t, y):
            return self.generator(t).full() @ y
        print("Started Solving the differential equation")
        result = solve_ivp(f, [0, self.t[-1]],
                           y0,
                           t_eval=self.t, method=method)
        n = self.Hsys.shape[0]
        states = [result.y[:, i].reshape(n, n)
                  for i in range(len(self.t))]
        print("Finished Solving the differential equation")
        return states
    
    def _decayww2(self,bath,w,w1,t):
        cks=np.array([i.coefficient for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        l=[]
        for i in range(len(cks)):
            mul=cks[i]/(vks[i]-1j*w)
            fd=np.exp(1j*(w1-w)*t)-np.exp(-t*(vks[i]-1j*w1))
            l.append(mul*fd)
        return sum(l)
    def decayww2(self,bath,w,w1,t):
        return self._decayww2(bath,w,w1,t) 
    def _ls(self,bath,w,w1,t):
        cks=np.array([i.coefficient.real for i in bath.exponents])
        vks=np.array([i.exponent for i in bath.exponents])
        l=[]
        for i in range(len(cks)):
            vkr=vks[i].real
            vki=vks[i].imag
            ckr=cks[i].real
            cki=cks[i].imag
            vb=(1j*w1-vkr)
            gamma = np.exp((1j*(w-w1))*t)/(vki**2 + vb**2)
            first = (ckr*vki + cki*vkr - 1j*cki *w1)
            second=-np.exp(1j*t*w1)*(ckr*vki+cki*vkr-1j*cki *w1)*np.cos(t*vki)*np.exp(-vkr*t)
            third=np.exp(1j*t*w1)*(cki*vki-ckr*vkr-1j*ckr *w1)*np.sin(t*vki)*np.exp(-vkr*t)
            l.append(gamma*(first+second+third))
        return sum(l)

    def LS(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.ls(bath, i[0], i[1], t)
        return rates
tree_util.register_pytree_node(
    crsolve,
    crsolve._tree_flatten,
    crsolve._tree_unflatten)
# TODO pictures
# TODO better naming
# TODO make result object
# TODO support Tensor Networks
#TODO Diffrax does not work unless one makes a pytree for Qobj apparently 
#TODO do the analytical integral