import pytest
import jax.numpy as jnp
from jax import random
import numpy as np
import nmm
    
key=random.key(42)


class TestBaths:
    @pytest.mark.parametrize("w,T",[(0,2),(1,2),(3,0)])
    def test_bose(self,w,T):
        bath=nmm.BosonicBath(T)
        if w==0 or T==0:
            assert bath.bose(w) == 0
        else:
            assert jnp.isclose(bath.bose(w),1/(jnp.exp(w/T)-1))
    @pytest.mark.parametrize("w,T",[(1,2)])
    def test_ohmic_spectral_density(self,w,T):
        bath=nmm.OhmicBath(T,1,1)
        assert bath.spectral_density(0) == 0
    @pytest.mark.parametrize("w,T",[(1,2)])
    def test_overdamped_spectral_density(self,w,T):
        bath=nmm.OverdampedBath(T,1,1)
        assert bath.spectral_density(0) == 0
    @pytest.mark.parametrize("w,T",[(1,2)])
    def test_power_spectrum(self,w,T):
        bath=nmm.OhmicBath(T,1,1)
        assert bath.power_spectrum(0) == 0
        
        
        