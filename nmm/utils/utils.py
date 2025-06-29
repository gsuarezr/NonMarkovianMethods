import jax.numpy as jnp
from jax.scipy.linalg import expm
from numbers import Number
from jax import tree_util
from jax import jit,Array

class Qobj:
    def __init__(self, op: jnp.array):
        self.data = op
        self.shape = op.shape
        self.dtype = op.dtype

    def _tree_flatten(self):
        children = (self.data,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __add__(self, other):

        if isinstance(other, Qobj):
            return Qobj(self.data + other.data)
        if other == 0: 
            return self
        else:
            raise TypeError(f"Not Implemented for {type(other)}")
        
    def __radd__(self, other):
        if isinstance(other, Qobj):
            return Qobj(self.data + other.data)
        if other == 0: 
            return self
        else:
            raise TypeError(f"Not Implemented for {type(other)}")

    def __sub__(self, other):
         if isinstance(other, Qobj):
             return Qobj(self.data - other.data)
         if other == 0:
             return self
         else:
             raise TypeError(f"Not Implemented for {type(other)}")
         
    def __rsub__(self, other):
         if isinstance(other, Qobj):
             return Qobj(self.data - other.data)
         if other == 0:
             return -1*self
         else:
             raise TypeError(f"Not Implemented for {type(other)}")

    @jit 
    def dag(self):
        return Qobj(jnp.conjugate(jnp.transpose(self.data)))

  
    def __mul__(self, other):
        if isinstance(other, Number):
            return Qobj(self.data * other)
        elif isinstance(other, Array):
            return Qobj(self.data * other)
        elif isinstance(other, Qobj):
            return Qobj(self.data @ other.data)

        else:
             raise TypeError(f"Multiplication not defined for {type(other)}")
        
    def __rmul__(self, other):
        if isinstance(other, Number):
            return Qobj(self.data * other)
        elif isinstance(other, Array):
            return Qobj(self.data * other)
        elif isinstance(other, Qobj):
            return Qobj(self.data @ other.data)

        else:
             raise TypeError(f"Multiplication not defined for {type(other)}")

    def __truediv__(self, other):
         if isinstance(other, Number):
             return Qobj(self.data / other)
         else:
             raise NotImplementedError("Ill defined Operation")

    @jit
    def expm(self):
        return Qobj(expm(self.data))
    @jit
    def eigenstates(self):
         eigvals, eigvecs_matrix= jnp.linalg.eigh(self.data)
         eigvecs = [Qobj(eigvecs_matrix[:, i:i+1]) for i in range(eigvecs_matrix.shape[1])]
         return eigvals, eigvecs

    def __eq__(self, other):
         return jnp.isclose(self.data, other.data).all().item()

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
        if type(other) in (int, float, complex):
            data =self.data* other
            return spre(data,kron=False)
        if isinstance(other,Array):
            data =self.data* other
            return spre(data,kron=False)
        data = self.data @ other.data
        return spre(data,kron=False)

    def __rmul__(self,other):
        if type(other) in (int, float, complex):
            data =self.data* other
            return spre(data,kron=False)
        if isinstance(other,Array):
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