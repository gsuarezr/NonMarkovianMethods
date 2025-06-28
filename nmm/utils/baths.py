import jax.numpy as jnp
import numpy as np
from jax import jit


class BosonicBath:
    def __init__(self, T):
        self.T = T

    def bose(self, ν):
        r"""
        It computes the Bose-Einstein distribution

        $$ n(\omega)=\frac{1}{e^{\beta \omega}-1} $$

        Parameters:
        ----------
        ν: float
            The mode at which to compute the thermal population

        Returns:
        -------
        float
            The thermal population of mode ν
        """
        if self.T == 0:
            return 0
        if np.isclose(ν, 0).all():
            return 0
        return np.exp(-ν / self.T) / (1-np.exp(-ν / self.T))

    def spectral_density(self, w):
        return None

    def correlation_function(self, t):
        return None

    def power_spectrum(self, w):
        return 2*(self.bose(w)+1)*self.spectral_density(w)


class OhmicBath(BosonicBath):
    def __init__(self, T, coupling, cutoff):
        super().__init__(T)
        self.coupling = coupling
        self.cutoff = cutoff
        self.label = "ohmic"

    def spectral_density(self, w):
        r"""
        It describes the spectral density of an Ohmic spectral density given by

        $$ J(\omega)= \alpha \omega e^{-\frac{|\omega|}{\omega_{c}}} $$

        Parameters
        ----------
        """
        return self.coupling*w*np.exp(-abs(w)/self.cutoff)

    def correlation_function(self, t):
        return None


class OverdampedBath(BosonicBath):
    def __init__(self, T, coupling, cutoff):
        super().__init__(T)
        self.coupling = coupling
        self.cutoff = cutoff
        self.label = "overdamped"
        self.params = np.array([coupling, cutoff], dtype=np.float64)

    def spectral_density(self, w):
        return 2*self.coupling*self.cutoff*w/(self.cutoff**2 + w**2)

    def _vk(self, k):
        if k == 0:
            return self.cutoff
        else:
            return 2*np.pi*k*self.T

    def _ckr(self, k):
        c, d = self.coupling, self.cutoff
        if k == 0:
            return c*d/np.tan(d/(2*self.T))
        else:
            vk = self._vk(k)
            return 4*c*d*vk*self.T/(vk**2 - d**2)

    def vk(self, k):
        return np.array([self._vk(i) for i in range(k)])

    def _cki(self, k):
        if k == 0:
            return -self.coupling*self.cutoff
        else:
            return 0

    def ckr(self, k):
        return np.array([self._ckr(i) for i in range(k)])

    def cki(self, k):
        return np.array([self._cki(i) for i in range(k)])

    def correlation_function(self, t, n=1000):
        return (self.ckr(n)+1j*self.cki(n))*np.exp(-(self.vkr(n)
                                                     + 1j*self.vki(n))*t)


class UnderdampedBath(BosonicBath):
    def __init__(self, T):
        super().__init__(T)

    def spectral_density(self, w):
        return

    def correlation_function(self, t):
        return super().correlation_function(t)
