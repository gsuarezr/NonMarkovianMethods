import jax.numpy as jnp
from jax.scipy.linalg import expm
from numbers import Number
from jax import tree_util
from jax import jit
import functools # Import functools for partial

class Qobj:
    def __init__(self, op: jnp.array):
        self.data = op
        # Store shape/dtype in aux_data if they are truly static
        # For now, deriving them on the fly is fine.
        self.shape = op.shape
        self.dtype = op.dtype

    def _tree_flatten(self):
        children = (self.data,)
        # aux_data could potentially store shape/dtype if they
        # are guaranteed *not* to change in ways JAX can't track.
        # For simple cases, leaving it empty is fine.
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --- JIT-able Methods ---

    @functools.partial(jit, static_argnums=(0,)) # 'self' is not static here
    def __add__(self, other):
        # Note: Jitting dunder methods can sometimes be tricky depending on
        # how Python dispatches them. Explicit external functions often safer.
        if isinstance(other, Qobj):
            return Qobj(self.data + other.data)
        if other == 0: # Be careful comparing JAX arrays with == inside JIT
            return self
        else:
            # Raising errors inside JIT can be problematic.
            # Consider validity checks outside JITted code if possible.
            raise TypeError(f"Not Implemented for {type(other)}")

    @functools.partial(jit, static_argnums=(0,))
    def __sub__(self, other):
         if isinstance(other, Qobj):
             return Qobj(self.data - other.data)
         if other == 0:
             return self
         else:
             raise TypeError(f"Not Implemented for {type(other)}")

    @jit # No static args needed if only operating on self.data
    def dag(self):
        return Qobj(jnp.conjugate(jnp.transpose(self.data)))

    # Jitting __mul__ can be complex due to the type check inside.
    # Often better to have separate functions for scalar/matrix mult.
    def __mul__(self, other):
        if isinstance(other, Number):
            return Qobj(self.data * other)
        elif isinstance(other, Qobj):
             return Qobj(self.data @ other.data)
        else:
             raise TypeError(f"Multiplication not defined for {type(other)}")


    @functools.partial(jit, static_argnums=(0,))
    def __truediv__(self, other):
         if isinstance(other, Number):
             return Qobj(self.data / other)
         else:
             raise NotImplementedError("Ill defined Operation")

    @jit
    def expm(self):
        return Qobj(expm(self.data))

    # --- Methods less suitable for JIT or needing care ---

    # 1. __eq__: Uses .item(), which forces synchronization and breaks tracing.
    #    If needed inside JIT, return the JAX boolean array directly.
    def __eq__(self, other):
         # This version is OK *outside* JIT, but not *inside* a JITted function.
         # return jnp.allclose(self.data, other.data) # JIT-friendly version
         return jnp.isclose(self.data, other.data).all().item()

    @jit
    def eigenstates(self):
         eigvals, eigvecs= jnp.linalg.eigh(self.data)
         #eigvecs = [Qobj(eigvecs_matrix[:, i:i+1]) for i in range(eigvecs_matrix.shape[1])]
         return eigvals, eigvecs

    # --- Non-computational methods (don't JIT) ---
    def __str__(self):
        s = f"Operator: \n {self.data}"
        return s

    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, key):
        indexed_data = self.data[key]
        return Qobj(indexed_data)
class spre:
    def __init__(self, op,kron=True):
        self.kron=kron
        if kron:
            op=op.data
            self.data = jnp.kron(op, jnp.eye(op.shape[0]))
            self.dim = op.shape[0]
        else:
            self.data=op
            self.dim = int(op.shape[0]**0.5)
        ## it may be worth using tensor contractions here (einsum)
        self.func = lambda x: Qobj((self.data@x.data.reshape(self.dim**2)).reshape(self.dim, self.dim))
    def _tree_flatten(self):
        children=(self.data,)
        aux_data={"kron":self.kron}
        return (children,aux_data)
    @classmethod
    def _tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    def __eq__(self,other):
        return jnp.isclose(self.data,other.data).all().item()
    def __str__(self):
        s=f"SuperOperator: \n {self.data}"
        return s
    def __repr__(self):
        return self.__str__()
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spre(data,kron=False)

    def __radd__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spre(data,kron=False)
        
    def __sub__(self, other):
        if isinstance(other,(spre,spost)):
            data =self.data - other.data
            return spre(data,kron=False)
        else:
            raise TypeError

    def __mul__(self, other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spre(data,kron=False)
        data = self.data @ other.data
        return spre(data,kron=False)

    def __rmul__(self,other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spre(data,kron=False)

    def __truediv__(self,other):
        if (isinstance(other,Number)):
            data=self.data/other
            return spost(data,kron=False)
        else:
            raise NotImplementedError("Ill defined Operation")
    def expm(self):
        return spre(expm(self.data),kron=False)
class spost:
    def __init__(self, op,kron=True):
        if kron:
            op=op.data
            self.data = jnp.kron(jnp.eye(op.shape[0]), op.T)
            self.dim = op.shape[0]
        else:
            self.data=op
            self.dim = int(op.shape[0]**0.5)
        ## it may be worth using tensor contractions here (einsum)
        self.func = lambda x: Qobj((self.data@x.data.reshape(self.dim**2)).reshape(self.dim, self.dim))  
    def _tree_flatten(self):
        children=(self.data,)
        aux_data={"kron":self.kron}
        return (children,aux_data)
    @classmethod
    def _tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    def __eq__(self,other):
        return jnp.isclose(self.data,other.data).all().item()
    def __str__(self):
        s=f"SuperOperator: \n {self.data}"
        return s
    def __repr__(self):
        return self.__str__()
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spost(data,kron=False)

    def __radd__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spost(data,kron=False)
        
    def __sub__(self, other):
        if isinstance(other,(spre,spost)):
            data =self.data - other.data
            return spost(data,kron=False)
        else:
            raise TypeError

    def __truediv__(self,other):
        if (isinstance(other,Number)):
            data=self.data/other
            return spost(data,kron=False)
        else:
            raise NotImplementedError("Ill defined Operation")
    def __mul__(self, other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spost(data,kron=False)
        data = self.data @ other.data
        return spre(data,kron=False)

    def __rmul__(self,other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spost(data,kron=False)
    def expm(self):
        return spost(expm(self.data),kron=False)
    
tree_util.register_pytree_node(
    Qobj,
    Qobj._tree_flatten,
    Qobj._tree_unflatten)
tree_util.register_pytree_node(
    spre,
    spre._tree_flatten,
    spre._tree_unflatten)
tree_util.register_pytree_node(
    spost,
    spost._tree_flatten,
    spost._tree_unflatten)