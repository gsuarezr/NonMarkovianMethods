import pytest
import jax.numpy as jnp
import numpy as np
import nmm
import qutip as qt
import jax
jax.config.update("jax_enable_x64", True)
sz=nmm.Qobj(jnp.array([[1, 0], [0, -1]],dtype=jnp.complex128))
sx=nmm.Qobj(jnp.array([[0, 1], [1, 0]],dtype=jnp.complex128))
sy=nmm.Qobj(jnp.array([[0, -1j], [0, 1j]],dtype=jnp.complex128))
H1   = qt.tensor(qt.sigmap()*qt.sigmam(), qt.identity(2))
H2   = qt.tensor(qt.identity(2), qt.sigmap()*qt.sigmam())
H12  = 0.25*(qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap()))
Hsys = H1 + H2 + H12
Q1 = qt.tensor(qt.sigmax(), qt.identity(2))
Hsys=nmm.Qobj(jnp.array(Hsys.full(),dtype=jnp.complex128))
Q1=nmm.Qobj(jnp.array(Q1.full(),dtype=jnp.complex128)) 

def commutator(A,B):
    com=A*B-B*A
    return com

@pytest.fixture
def init():
    Hsys = sz/2
    Q = sz
    alpha= 0.05
    gamma = 5
    t = np.linspace(0, 1000, 10)
    T=1
    bath=nmm.OhmicBath(T,alpha,gamma)
    obj = nmm.cumulant.csolve(Hsys,t ,[bath], [Q],cython=True)
    return obj
@pytest.fixture
def init_jax():
    Hsys = sz/2
    Q = sz
    alpha= 0.05
    gamma = 5
    t = np.linspace(0, 1000, 10)
    T=1
    bath=nmm.OhmicBath(T,alpha,gamma)
    obj = nmm.cumulant.csolve(Hsys,t ,[bath], [Q],cython=True)
    return obj

class TestCumulant:
    @pytest.mark.parametrize("ars,expected",
                             [((1, 2, 3), (0.0628+0.8859j)),
                              ((1, 2, 0), 0 + 0j),
                              ((1.05, 0.095, 2), (0.3599 - 0.5087j))])
    def test_γfa(self, init, ars, expected):
        assert np.isclose(init.γfa(init.baths[0],*ars), expected, atol=1e-3)

    @pytest.mark.parametrize("ars,expected",
                             [((1, 2, 3, 4), (0.0023 + 0.0051j)),
                              ((0, 0, 0, 5), 0 + 0j),
                              ((1.05, 0.095, 2, 1), (0.0433-0.0610j))])
    def test_γ(self, init, ars, expected):
        assert np.isclose(init._γ(ars[0],init.baths[0],*ars[1:]), expected, atol=1e-3)

    @pytest.mark.parametrize("ars,expected",
                             [((1, 1, [2]), (0.7490)),
                              ((1, -1, [2]), (-0.1119+0.2445j)),
                              ((-1, -1, [2]), (0.3618))])
    def test_Γgen(self, init, ars, expected):
        assert np.isclose(init.Γgen(init.baths[0],*ars), expected, atol=1e-3).all()
    @pytest.mark.parametrize("Hsys,Q",[(sz,sx),(sz,sz),(sz,sz+sx),(Hsys,Q1)])
    def test_jump_operators(self,init,Hsys,Q):
        init.Hsys=Hsys
        jumps=init.jump_operators(Q)
        if (commutator(init.Hsys,Q) == 0*init.Hsys):
            assert len(jumps)==1
            assert list(jumps.keys())[0]==0
        else:
            for key,value in jumps.items():
                assert jnp.isclose(commutator(init.Hsys,value).data,(-key*value).data).all()
                assert commutator(init.Hsys,value.dag()*value)==0*value
