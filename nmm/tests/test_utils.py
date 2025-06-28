import pytest
import jax.numpy as jnp
from jax import random
import jax
import numpy as np
import nmm
from nmm import spre, spost
key=random.key(42)
jax.config.update("jax_enable_x64", True)

class TestQobj:
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_create(self,dims):
        random_matrix=random.normal(key,[dims,dims])
        qobj=nmm.Qobj(random_matrix)
        assert jnp.isclose(qobj.data,random_matrix).all()
        assert qobj.shape == (dims,dims)
        assert qobj.dtype == random_matrix.dtype
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_aritmethic(self,dims):
        qobj1=nmm.Qobj(random.normal(key,[dims,dims]))
        qobj2=nmm.Qobj(random.normal(key,[dims,dims]))
        assert jnp.isclose((qobj1+qobj2).data,qobj2.data+qobj1.data).all()
        assert jnp.isclose((qobj1-qobj2).data,qobj1.data-qobj2.data).all()
        with pytest.raises(Exception) as e_info:
            qobj1/qobj2
        assert jnp.isclose((qobj1*jnp.pi).data,jnp.pi*qobj1.data).all()
        assert jnp.isclose((jnp.pi*qobj1).data,jnp.pi*qobj1.data).all()
        assert jnp.isclose((qobj1*qobj2).data,qobj2.data@qobj1.data).all()
        assert jnp.isclose((qobj2*qobj1).data,qobj1.data@qobj2.data).all()
        assert jnp.isclose((qobj2+0).data,qobj2.data).all()
        assert jnp.isclose((qobj2-0).data,qobj2.data).all()
        assert jnp.isclose((0+qobj2).data,qobj2.data).all()
        assert jnp.isclose((0-qobj2).data,-qobj2.data).all()
        assert jnp.isclose((qobj2/2).data,qobj2.data/2).all()
        with pytest.raises(TypeError) as e_info:
            qobj1+1
        with pytest.raises(TypeError) as e_info:
            1+qobj1
        with pytest.raises(TypeError) as e_info:
            qobj1-1
        with pytest.raises(TypeError) as e_info:
            1-qobj1
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_expm(self,dims):
        qobj1=nmm.Qobj(random.normal(key,[dims,dims]))
        assert jnp.isclose(qobj1.expm().data,jax.scipy.linalg.expm(qobj1.data)).all()
        
    
class TestSuperop:
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_spre(self,dims):
        qobj1=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        qobj2=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        assert jnp.isclose((spre(qobj1)(qobj2)).data,qobj1.data@qobj2.data).all()
        assert jnp.isclose(((2*spre(qobj1))(qobj2)).data,2*qobj1.data@qobj2.data).all()
        assert jnp.isclose(((spre(qobj1)*2)(qobj2)).data,2*qobj1.data@qobj2.data).all()
        assert jnp.isclose(((spre(qobj1)/2)(qobj2)).data,qobj1.data@qobj2.data/2).all()
        with pytest.raises(Exception) as e_info:
            (1+spre(qobj2))
        with pytest.raises(Exception) as e_info:
            (spre(qobj1)+2)
        with pytest.raises(Exception) as e_info:
            spre(qobj1)/qobj2
        op1=spre(qobj1)+spre(qobj2)
        assert jnp.isclose((op1(qobj2)).data,(qobj1.data+qobj2.data)@qobj2.data).all()  
        #assert (spre(qobj1)==spre(qobj2))==False
        assert (spre(qobj1)==spre(qobj1))==True
        
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_spost(self,dims):
        qobj1=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        qobj2=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        assert jnp.isclose((spost(qobj1)(qobj2)).data,qobj2.data@qobj1.data).all()
        assert jnp.isclose(((2*spost(qobj1))(qobj2)).data,2*qobj2.data@qobj1.data).all()
        assert jnp.isclose(((spost(qobj1)*2)(qobj2)).data,2*qobj2.data@qobj1.data).all()
        assert jnp.isclose(((spost(qobj1)/2)(qobj2)).data,qobj2.data@qobj1.data/2).all()
        assert jnp.isclose(((spost(qobj1)-spost(qobj2))(qobj2)).data,qobj2.data@(qobj1.data-qobj2.data)).all()
        assert jnp.isclose(((spost(qobj1)+spost(qobj2))(qobj2)).data,qobj2.data@(qobj1.data+qobj2.data)).all()
        with pytest.raises(Exception) as e_info:
            (1+spost(qobj2))
        with pytest.raises(Exception) as e_info:
            (spost(qobj1)+2)
        with pytest.raises(Exception) as e_info:
            spost(qobj1)/spost(qobj2)
        op1=spost(qobj1)+spost(qobj2)
        assert jnp.isclose((op1(qobj2)).data,qobj2.data@(qobj1.data+qobj2.data)).all()
    @pytest.mark.parametrize("dims",range(1,20,5))
    def test_commutator(self,dims):
        qobj1=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        qobj2=nmm.Qobj(random.normal(key,[dims,dims],dtype=jnp.float64))
        assert jnp.isclose(((spre(qobj1)-spost(qobj1))(qobj2)).data,
                           qobj1.data@qobj2.data - qobj2.data@qobj1.data ).all()