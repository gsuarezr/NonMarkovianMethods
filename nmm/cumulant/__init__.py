import sys, os
sys.path.insert(0, os.path.abspath(".."))
from nmm.cumulant.cumulant import csolve
from nmm.cumulant.cum import bath_csolve
from nmm.cumulant.refined import crsolve
from nmm.cumulant.cumulant2 import csolve2
from nmm.cumulant.cumulant_jax import jaxcsolve
from nmm.cumulant.cumulant_jax2 import jaxcsolve2
from nmm.cumulant.cumulant_jax2 import Qobj as Qobj2