import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.lax import cond, while_loop, fori_loop, map as jax_map
from numbers import Number
from jax import tree_util
from jax import jit, Array
import functools
from collections import defaultdict # Still needed for initial non-JAX part
import warnings
from jax.scipy.linalg import eigh # Use JAX's eigh
import itertools # Needed for combinations outside JAX trace

# Helper for Qobj, spre, spost equality checks - return JAX bool array
# Used for debugging/assertions outside JAX transformations
def jax_is_close_all(a, b):
    # Assumes a and b have a .data attribute
    return jnp.isclose(a.data, b.data).all()


class Qobj:
    def __init__(self, op):
        # Ensure op is a JAX array
        self.data = jnp.asarray(op)
        # Shape and dtype are attributes derived from the static data
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def _tree_flatten(self):
        children = (self.data,)
        # shape and dtype are static properties derivable from data
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Ensure children[0] is a JAX array before passing to init
        return cls(jnp.asarray(children[0]), **aux_data)

    # --- JAX-Compatible Methods ---
    # Arithmetic operations correctly delegate to jnp, assuming other is JAX compatible
    def __add__(self, other):
        if isinstance(other, Qobj):
            return Qobj(self.data + other.data)
        if isinstance(other, (Number, Array)):
             return Qobj(self.data + other)
        if other == 0: # Python int 0
             return self
        raise TypeError(f"Addition not defined for {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (Number, Array)):
            return Qobj(other + self.data) # Let JAX handle broadcasting
        raise TypeError(f"Addition not defined for {type(other)}")


    def __sub__(self, other):
         if isinstance(other, Qobj):
             return Qobj(self.data - other.data)
         if isinstance(other, (Number, Array)):
             return Qobj(self.data - other)
         if other == 0: # Python int 0
             return self
         raise TypeError(f"Subtraction not defined for {type(other)}")

    def __rsub__(self, other):
         if isinstance(other, (Number, Array)):
             return Qobj(other - self.data)
         raise TypeError(f"Subtraction not defined for {type(other)}")


    # dag uses jnp.conjugate and jnp.transpose - JAX compatible
    def dag(self):
        return Qobj(jnp.conjugate(jnp.transpose(self.data)))


    # Mul handles Array, Number, and Qobj multiplication via @ - JAX compatible
    def __mul__(self, other):
        if isinstance(other, (Number, Array)):
            return Qobj(self.data * other)
        elif isinstance(other, Qobj):
            return Qobj(self.data @ other.data)
        raise TypeError(f"Multiplication not defined for {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, (Number, Array)):
             return Qobj(other * self.data)
        raise TypeError(f"Multiplication not defined for {type(other)}")


    def __truediv__(self, other):
         if isinstance(other, (Number, Array)):
             return Qobj(self.data / other)
         raise NotImplementedError("Ill defined Operation")

    # expm uses jax.scipy.linalg.expm - JAX compatible
    def expm(self):
        return Qobj(expm(self.data))

    # eigenstates uses jnp.linalg.eigh - JAX compatible
    # Returns eigvals (JAX array) and eigvecs (JAX array) - static structure
    def eigenstates(self):
         eigvals, eigvecs_matrix = eigh(self.data)
         return eigvals, Qobj(eigvecs_matrix) # Return eigenvalues array and eigenvectors matrix Qobj

    # Modified __eq__ to return JAX boolean array, remove .item()
    # Note: Cannot use this in Python 'if' based on traced values
    def __eq__(self, other):
         if isinstance(other, Qobj):
              return jnp.isclose(self.data, other.data).all()
         raise TypeError("Equality comparison only defined for Qobj")

    def __str__(self):
        # Note: String conversion inside JIT/grad is problematic side effect
        s = f"Operator: \n {self.data}"
        return s

    def __repr__(self):
        return self.__str__()

    # __getitem__ is fine if 'key' is a static index or JAX array index/slice
    def __getitem__(self, key):
         indexed_data = self.data[key]
         if isinstance(indexed_data, Array) and indexed_data.ndim == 0:
             return indexed_data
         return Qobj(indexed_data)


class spre:
    def __init__(self, op, kron=True):
        op_data = op.data if isinstance(op, Qobj) else jnp.asarray(op)
        self.kron = kron
        if kron:
            if op_data.ndim != 2 or op_data.shape[0] != op_data.shape[1]: raise ValueError("Input operator for spre with kron=True must be a square matrix.")
            self.data = jnp.kron(op_data, jnp.eye(op_data.shape[0], dtype=op_data.dtype))
            self.dim = op_data.shape[0] # Store system dimension
        else:
            self.data = jnp.asarray(op_data)
            if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]: raise ValueError("Input operator for spre with kron=False must be a square superoperator matrix.")
            dim2 = self.data.shape[0] # Dimension of the superoperator matrix
            # Validate that the superoperator dimension is a perfect square
            dim = int(round(dim2**0.5)) # Use round to handle potential float inaccuracies
            if dim * dim != dim2:
                 raise ValueError(f"Superoperator matrix shape {self.data.shape} ({dim2}) is not from a square system operator (dim^2). Calculated dim: {dim}")
            self.dim = dim # Store system dimension


        # Functional application: superoperator matrix @ vectorized density matrix data
        # Use the stored system dimension self.dim
        self.func = lambda x: Qobj(jnp.dot(self.data, x.data.reshape(-1)).reshape(self.dim, self.dim))

    def _tree_flatten(self):
        # --- Add this print statement ---
        print(f"DEBUG: spre _tree_flatten called on instance {id(self)}. Type of self.data is {type(self.data)}. Shape: {getattr(self.data, 'shape', 'N/A')}")
        if not isinstance(self.data, (jnp.ndarray, Array)): # Check against JAX array types
             print("DEBUG: !!! Non-JAX array detected in spre.data before flattening !!!")
             # Optionally raise a specific error here to stop closer to the source
             # raise TypeError(f"Problematic spre.data type: {type(self.data)}")
        # -------------------------------
        children = (self.data,)
        # Make sure self.dim (system dimension) is correctly stored in aux_data
        aux_data = {"kron": self.kron, "dim": self.dim}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # *** MODIFIED UNFLATTEN METHOD ***
        # Create an uninitialized instance to bypass __init__ validation/side effects
        instance = cls.__new__(cls)

        # Set attributes directly from children and aux_data
        instance.data = jnp.asarray(children[0])
        instance.kron = aux_data["kron"]
        instance.dim = aux_data["dim"] # Restore the system dimension from aux_data

        # Recreate the functional application lambda using the restored dim
        instance.func = lambda x: Qobj(jnp.dot(instance.data, x.data.reshape(-1)).reshape(instance.dim, instance.dim))

        return instance
    
    
    # Modified __eq__ to return JAX boolean array, remove .item()
    def __eq__(self, other):
         if isinstance(other, spre):
              return jnp.isclose(self.data, other.data).all()
         raise TypeError("Equality comparison only defined for spre")

    def __str__(self):
        s = f"SuperOperator (spre): \n {self.data}"
        return s

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
         if not isinstance(x, Qobj):
             raise TypeError("spre can only be called with a Qobj.")
         return self.func(x)

    # Arithmetic operations modified to handle Number/Array and return spre with kron=False
    def __add__(self, other):
        if isinstance(other, (Number, Array)):
             return spre(self.data + other, kron=False)
        elif isinstance(other, (spre, spost)):
             return spre(self.data + other.data, kron=False)
        if other == 0:
             return self
        raise TypeError(f"Addition not defined for {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (Number, Array)):
             return spre(other + self.data, kron=False)
        raise TypeError(f"Addition not defined for {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (spre, spost)):
             data = self.data - other.data
             return spre(data, kron=False)
        if isinstance(other, (Number, Array)):
             return spre(self.data - other, kron=False)
        raise TypeError(f"Subtraction not defined for {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, (Number, Array)):
             return spre(other - self.data, kron=False)
        raise TypeError(f"Subtraction not defined for {type(other)}")


    def __mul__(self, other):
        if isinstance(other, (Number, Array)):
             data = self.data * other
             return spre(data, kron=False)
        elif isinstance(other, (spre, spost)):
             data = self.data @ other.data
             return spre(data, kron=False)
        raise TypeError(f"Multiplication not defined for {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, (Number, Array)):
             data = other * self.data
             return spre(data, kron=False)

    def __truediv__(self, other):
         if isinstance(other, (Number, Array)):
             data = self.data / other
             return spre(data, kron=False)
         raise NotImplementedError("Ill defined Operation")

    # expm uses jax.scipy.linalg.expm - JAX compatible
    def expm(self):
         return spre(expm(self.data), kron=False)


class spost:
    def __init__(self, op, kron=True):
        op_data = op.data if isinstance(op, Qobj) else jnp.asarray(op)
        self.kron = kron
        if kron:
            if op_data.ndim != 2 or op_data.shape[0] != op_data.shape[1]: raise ValueError("Input operator for spost with kron=True must be a square matrix.")
            self.data = jnp.kron(jnp.eye(op_data.shape[0], dtype=op_data.dtype), op_data.T)
            self.dim = op_data.shape[0] # Store system dimension
        else:
            self.data = jnp.asarray(op_data)
            if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]: raise ValueError("Input operator for spost with kron=False must be a square superoperator matrix.")
            dim2 = self.data.shape[0] # Dimension of the superoperator matrix
            # Validate that the superoperator dimension is a perfect square
            dim = int(round(dim2**0.5)) # Use round
            if dim * dim != dim2:
                raise ValueError(f"Superoperator matrix shape {self.data.shape} ({dim2}) is not from a square system operator (dim^2). Calculated dim: {dim}")
            self.dim = dim # Store system dimension

        # Functional application
        # Use the stored system dimension self.dim
        self.func = lambda x: Qobj(jnp.dot(self.data, x.data.reshape(-1)).reshape(self.dim, self.dim))

    def _tree_flatten(self):
        children = (self.data,)
        # Make sure self.dim (system dimension) is correctly stored in aux_data
        aux_data = {"kron": self.kron, "dim": self.dim}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # *** MODIFIED UNFLATTEN METHOD ***
        # Create an uninitialized instance
        instance = cls.__new__(cls)

        # Set attributes directly
        instance.data = jnp.asarray(children[0])
        instance.kron = aux_data["kron"]
        instance.dim = aux_data["dim"] # Restore system dimension

        # Recreate the functional application lambda using the restored dim
        instance.func = lambda x: Qobj(jnp.dot(instance.data, x.data.reshape(-1)).reshape(instance.dim, instance.dim))

        return instance

    # Modified __eq__ to return JAX boolean array, remove .item()
    def __eq__(self, other):
         if isinstance(other, spost):
              return jnp.isclose(self.data, other.data).all()
         raise TypeError("Equality comparison only defined for spost")


    def __str__(self):
        s = f"SuperOperator (spost): \n {self.data}"
        return s

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
         if not isinstance(x, Qobj):
             raise TypeError("spost can only be called with a Qobj.")
         return self.func(x)


    # Arithmetic operations modified to handle Number/Array and return spost with kron=False
    def __add__(self, other):
        if isinstance(other, (Number, Array)):
             return spost(self.data + other, kron=False)
        elif isinstance(other, (spre, spost)):
             return spost(self.data + other.data, kron=False)
        if other == 0:
             return self
        raise TypeError(f"Addition not defined for {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (Number, Array)):
             return spost(other + self.data, kron=False)
        raise TypeError(f"Addition not defined for {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (spre, spost)):
             data = self.data - other.data
             return spost(data, kron=False)
        if isinstance(other, (Number, Array)):
             return spost(self.data - other, kron=False)
        raise TypeError(f"Subtraction not defined for {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, (Number, Array)):
             return spost(other - self.data, kron=False)
        raise TypeError(f"Subtraction not defined for {type(other)}")


    def __mul__(self, other):
        if isinstance(other, (Number, Array)):
             data = self.data * other
             return spost(data, kron=False)
        elif isinstance(other, (spre, spost)):
             data = self.data @ other.data
             return spost(data, kron=False)
        raise TypeError(f"Multiplication not defined for {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, (Number, Array)):
             data = other * self.data
             return spost(data, kron=False)

    def __truediv__(self, other):
         if isinstance(other, (Number, Array)):
             data = self.data / other
             return spost(data, kron=False)
         raise NotImplementedError("Ill defined Operation")

    # expm uses jax.scipy.linalg.expm - JAX compatible
    def expm(self):
         return spost(expm(self.data), kron=False)


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

# Assume Bath class is defined elsewhere and its exponents attribute
# is a list/tuple of objects with 'coefficient' and 'exponent' attributes
# which can be converted to JAX arrays.

# --- Modified jaxcsolve class ---

class jaxcsolve2:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, ls=False):
        self.Hsys = Hsys if isinstance(Hsys, Qobj) else Qobj(Hsys)
        self.Qs = tuple(q if isinstance(q, Qobj) else Qobj(q) for q in Qs)
        self.t = jnp.asarray(t) if not isinstance(t, Array) else t
        self.eps = jnp.asarray(eps)
        self.dtype = self.Hsys.dtype
        self.ls = ls

        # Pre-calculate jump operators and frequencies during init (outside JAX trace)
        # This part still uses Python loops and dictionaries and is NOT differentiable.
        # It provides the static structure of jumps and frequencies needed by the
        # JAX-traceable generator function.
        all_jumps_by_bath = []
        all_ws_by_bath = []
        for Q in self.Qs: # Iterate over Qs, assuming one bath per Q
             # Call the original (non-JAX-traceable) jump_operators here once
             # This returns Python lists.
             # THIS PART IS NOT DIFFERENTIABLE AS WRITTEN
             jumps_list, ws_list = self._jump_operators_initial(Q)

             # Convert lists of Qobjs/JAX arrays to static JAX arrays
             # Assuming jumps are square matrices of the same size
             jumps_array_data = jnp.stack([op.data for op in jumps_list], axis=0) if jumps_list else jnp.empty((0,) + self.Hsys.shape, dtype=self.dtype)
             ws_array = jnp.stack(ws_list, axis=0) if ws_list else jnp.empty((0,), dtype=self.dtype)

             all_jumps_by_bath.append(jumps_array_data)
             all_ws_by_bath.append(ws_array)

        # Store these as static PyTree components (tuples of JAX arrays)
        self._static_jumps_by_bath_data = tuple(all_jumps_by_bath)
        self._static_ws_by_bath = tuple(all_ws_by_bath)
        self.baths = baths # baths list remains
  # *** UPDATED PYTREE METHODS FOR jaxcsolve ***
    def _tree_flatten(self):
        children = (
            self.Hsys,
            self.t,
            self.eps,
            self.Qs,
            self._static_jumps_by_bath_data,
            self._static_ws_by_bath
        )
        aux_data = {
            "dtype": self.dtype,
            "ls": self.ls,
            "baths": self.baths # Python list/tuple of Bath objects
        }

        # --- Add these print statements ---
        print(f"DEBUG: jaxcsolve _tree_flatten called on instance {id(self)}")
        print(f"DEBUG:   Children types: {[type(c) for c in children]}")
        # If children contain nested structures (like tuples/lists of arrays),
        # you might need tree_util.tree_leaves to see the leaf types
        leaves, _ = tree_util.tree_flatten(children)
        print(f"DEBUG:   Flattened Children (Leaves) types: {[type(leaf) for leaf in leaves]}")
        print(f"DEBUG:   Aux data keys: {aux_data.keys()}")
        print(f"DEBUG:   Aux data 'baths' type: {type(self.baths)}")
        if isinstance(self.baths, (list, tuple)) and self.baths:
             print(f"DEBUG:   Type of first element in 'baths': {type(self.baths[0])}")
             # If Bath object has attributes, check those too if they might be problematic
             # e.g., if hasattr(self.baths[0], 'exponents'):
             # print(f"DEBUG:     Type of baths[0].exponents: {type(self.baths[0].exponents)}")
        # ----------------------------------

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children_flat): # children is flat here
        # *** MODIFIED UNFLATTEN METHOD ***
        # Create an uninitialized instance to bypass __init__
        instance = cls.__new__(cls)

        # We need to tree_unflatten the 'children_flat' back into the
        # original 'children' structure using the 'structure' from flatten.
        # This is handled by JAX's internal PyTree traversal, but let's
        # print what's being passed in.

        # --- Add these print statements ---
        print(f"DEBUG: jaxcsolve _tree_unflatten called.")
        print(f"DEBUG:   Input children_flat types: {[type(c) for c in children_flat]}")
        print(f"DEBUG:   Aux data keys: {aux_data.keys()}")
        # ----------------------------------

        # The unflatten logic needs the structure of the original children
        # to put the flat leaves back correctly. JAX handles this internally
        # when calling this method. The `children_flat` argument *is* the
        # flat list of leaves.

        # The previous unflatten logic assumed 'children' was the structured tuple,
        # but it receives the flat tuple. Let's fix that and print.
        # The PyTree definition implies:
        # flat_leaves, structure = tree_flatten(instance_before_flatten)
        # instance_after_unflatten = tree_unflatten(structure, flat_leaves)

        # Inside _tree_unflatten(aux_data, children_flat):
        # children_flat is the flat tuple of all leaves from the original 'children' tuple.
        # aux_data contains the static stuff.
        # We need to reconstruct the original 'children' tuple structure from the 'children_flat'
        # using the structure information (which JAX provides implicitly or via a helper).
        # A common pattern is to manually map the flat children back based on the known structure.

        # Let's revert the unflatten slightly to match the expected input (flat children)
        # and keep the direct attribute setting, printing the input flat children.

        # Unpack the flat children (THIS IS THE ISSUE - order/structure matter here)
        # The order in children_flat is all leaves from Hsys, then all leaves from t, etc.
        # This manual unpacking based on knowing leaf counts is error-prone.
        # A better way is to use tree_unflatten on the *structure* of the original children.

        # Let's try the direct attribute setting but correctly understand 'children'.
        # JAX provides a list of *all* leaf children from the top-level children tuple.
        # We need to map these back.

        # --- Revert and print the input 'children' (which is the flat list of leaves) ---
        # This method receives (aux_data, flat_children_list)

        print(f"DEBUG: jaxcsolve _tree_unflatten called. Input flat children list types: {[type(c) for c in children_flat]}")
        print(f"DEBUG: Aux data keys: {aux_data.keys()}")


        # --- The unflattening logic needs to correctly map the flat children back ---
        # The previous unflatten logic assumed 'children' was the structured tuple
        # (Hsys, t, ...). It is the *flat list of leaves*.

        # Re-implement the unflatten using tree_unflatten on the structure
        # We need the structure of the original children tuple.
        # This structure is not directly passed to _tree_unflatten, which is why
        # the direct attribute setting approach can be simpler but requires care.

        # Let's try the direct attribute setting again, assuming the input 'children' *is*
        # the flat list of leaves from the original children tuple.
        # The original children tuple structure is implicit: (Hsys, t, eps, Qs, jumps, ws)
        # We need to know how many leaves are in each of these to unpack 'children'.
        # This requires re-flattening the structure of the original children tuple.

        # Let's use the simpler direct assignment approach from the previous fix,
        # but add the print statement for the input 'children' (flat list).

        # Assuming 'children' is the flat list of leaves:
        # We cannot easily unpack `Hsys, t, eps, ... = children` if children is flat.
        # We need to use tree_unflatten with the structure.

        # Let's go back to a correct PyTree unflatten pattern:
        # 1. Define the structure of the original children tuple.
        # 2. Use tree_unflatten on the flat input children list to get the structured children.
        # 3. Use aux_data and structured children to reconstruct the instance.

        # Define the structure of the children tuple (using a dummy instance or knowing the types)
        # This is tricky without the original structured children.

        # Let's revert the _tree_unflatten to the version that receives
        # the structured children (this is the standard PyTree API).
        # The error was in the *logic* inside, not the received arguments.

        # --- Reverting _tree_unflatten input back to (aux_data, structured_children) ---
        # The documentation and examples typically show _tree_unflatten receiving the structured tuple.
        # Let's assume the API is (aux_data, structured_children) and the error was internal logic.

        print(f"DEBUG: jaxcsolve _tree_unflatten called. Input structured children types: {[type(c) for c in children_flat]}") # Assuming 'children' is structured here
        print(f"DEBUG: Aux data keys: {aux_data.keys()}")


        # Reconstruct the instance using the structured children and aux_data
        # This is the logic from the previous successful PyTree fix attempt.
        Hsys, t, eps, Qs, _static_jumps_by_bath_data, _static_ws_by_bath = children_flat

        instance = cls.__new__(cls) # Bypass __init__

        instance.Hsys = Hsys
        instance.t = t
        instance.eps = eps
        instance.Qs = Qs
        instance._static_jumps_by_bath_data = _static_jumps_by_bath_data
        instance._static_ws_by_bath = _static_ws_by_bath
        instance.dtype = aux_data["dtype"]
        instance.ls = aux_data["ls"]
        instance.baths = aux_data["baths"]

        return instance


    # Helper for initial jump operator calculation (NOT JAX-traceable)
    # Replicates the logic of the original jump_operators + jump_calc without the dict grouping
    def _jump_operators_initial(self, Q):
         evals, evecs_qobj = self.Hsys.eigenstates() # eigenstates returns Qobj now
         evecs = evecs_qobj.data # Get the JAX array

         N = evecs.shape[1]
         jumps_list = []
         ws_list = []

         # Using Python loops is fine here as it's outside the JAX trace
         for j in range(N):
             for k in range(N): # Include k=j for dephasing terms if needed later
                  omega = evals[k] - evals[j]
                  state_j = Qobj(evecs[:, j:j+1])
                  state_k = Qobj(evecs[:, k:k+1])

                  # Elementary jump operator |j><k|
                  elementary_op = state_j * state_k.dag()

                  jumps_list.append(elementary_op)
                  ws_list.append(omega)

         # The original jump_calc grouped and summed based on frequencies.
         # For JAX-compatibility, we pass the elementary operators and frequencies
         # and perform sums/combinations within the JAX-traceable part if needed.
         # Based on the original matrix_form, it combines elementary jump operators.
         # Let's return all elementary ops and their frequencies.

         return jumps_list, ws_list


    # Use jnp.where for control flow based on values - JAX compatible
    def gamma_gen(self, bath, w, w1, t):
         return jnp.where(
             jnp.isclose(w, w1), # JAX boolean condition
             self._decayww(bath, w, t),
             self._decayww2(bath, w, w1, t)
         )

    # Modified _decayww to use jnp.where for conditional division - JAX compatible
    def _decayww(self, bath, w, t):
        cks = jnp.array([i.coefficient for i in bath.exponents])
        vks = jnp.array([i.exponent for i in bath.exponents])

        term1_num = (vks * t - 1j * w * t - 1) + jnp.exp(-(vks - 1j * w) * t)
        term1_den = (vks - 1j * w)**2

        # --- FIX START ---
        # term1 = jnp.divide(term1_num * cks, term1_den, where=term1_den != 0, fill_value=0) # Original Incorrect
        term1_raw = (term1_num * cks) / term1_den # Perform division
        term1 = jnp.where(
            term1_den != 0, # Condition: where the denominator is not zero
            term1_raw,      # Use the division result
            jnp.array(0.0, dtype=self.dtype) # Use fill value (ensure correct dtype)
        )
        # --- FIX END ---

        return 2 * jnp.real(jnp.sum(term1))

      # Modified _decayww2 to use jnp.where for conditional division - JAX compatible
    def _decayww2(self, bath, w, w1, t):
        cks = jnp.array([i.coefficient for i in bath.exponents])
        vks = jnp.array([i.exponent for i in bath.exponents])

        a = (vks - 1j * w1)
        b = (vks - 1j * w)

        # --- FIX START ---
        # term1 = jnp.divide(cks * jnp.exp(-b * t), a * b, where=(a * b) != 0, fill_value=0) # Original Incorrect
        term1_den = a * b
        term1_raw = (cks * jnp.exp(-b * t)) / term1_den
        term1 = jnp.where(
             term1_den != 0,
             term1_raw,
             jnp.array(0.0, dtype=self.dtype) # Ensure correct dtype
        )

        # term2 = jnp.divide(jnp.conjugate(cks) * jnp.exp(-jnp.conjugate(a) * t), jnp.conjugate(a) * jnp.conjugate(b), where=(jnp.conjugate(a) * jnp.conjugate(b)) != 0, fill_value=0) # Original Incorrect
        term2_den = jnp.conjugate(a) * jnp.conjugate(b)
        term2_raw = (jnp.conjugate(cks) * jnp.exp(-jnp.conjugate(a) * t)) / term2_den
        term2 = jnp.where(
             term2_den != 0,
             term2_raw,
             jnp.array(0.0, dtype=self.dtype) # Ensure correct dtype
        )
        # --- FIX END ---

        omega_diff = w - w1
        # --- FIX START ---
        # inv_omega_diff = jnp.divide(1j, omega_diff, where=omega_diff != 0, fill_value=jnp.array(jnp.inf, dtype=jnp.complex128) * jnp.sign(omega_diff)) # Original Incorrect
        inv_omega_diff_raw = 1j / omega_diff
        inv_omega_diff = jnp.where(
            omega_diff != 0,
            inv_omega_diff_raw,
            jnp.array(jnp.inf, dtype=jnp.complex128) * jnp.sign(omega_diff) # Use the original fill value
        )
        # --- FIX END ---


        # --- FIX START ---
        # term3 = cks * (jnp.divide(1, b, where=b != 0, fill_value=0) - jnp.divide(jnp.exp(1j * omega_diff * t), a, where=a != 0, fill_value=0)) # Original Incorrect
        term3_part1_raw = 1 / b
        term3_part1 = jnp.where(b != 0, term3_part1_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype

        term3_part2_raw = jnp.exp(1j * omega_diff * t) / a
        term3_part2 = jnp.where(a != 0, term3_part2_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype
        term3 = cks * (term3_part1 - term3_part2)

        # term4 = jnp.conjugate(cks) * (jnp.divide(1, jnp.conjugate(a), where=jnp.conjugate(a) != 0, fill_value=0) - jnp.divide(jnp.exp(1j * omega_diff * t), jnp.conjugate(b), where=jnp.conjugate(b) != 0, fill_value=0)) # Original Incorrect
        term4_part1_raw = 1 / jnp.conjugate(a)
        term4_part1 = jnp.where(jnp.conjugate(a) != 0, term4_part1_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype

        term4_part2_raw = jnp.exp(1j * omega_diff * t) / jnp.conjugate(b)
        term4_part2 = jnp.where(jnp.conjugate(b) != 0, term4_part2_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype
        term4 = jnp.conjugate(cks) * (term4_part1 - term4_part2)
        # --- FIX END ---


        actual = term1 + term2 + inv_omega_diff * (term3 + term4)

        return jnp.sum(actual)

    def _LS_single(self, bath, w, w1, t):
        cks = jnp.array([i.coefficient for i in bath.exponents])
        vks = jnp.array([i.exponent for i in bath.exponents])

        def ls_w_eq_w1_branch(cks, vks, w, t):
             term1_num = (vks * t - 1j * w * t - 1) + jnp.exp(-(vks - 1j * w) * t)
             term1_den = (vks - 1j * w)**2
             # --- FIX START ---
             # term1 = jnp.divide(term1_num * cks, term1_den, where=term1_den != 0, fill_value=0) # Original Incorrect
             term1_raw = (term1_num * cks) / term1_den
             term1 = jnp.where(
                 term1_den != 0,
                 term1_raw,
                 jnp.array(0.0, dtype=self.dtype) # Ensure correct dtype
             )
             # --- FIX END ---
             return jnp.imag(jnp.sum(term1)) / 2

        def ls_w_ne_w1_branch(cks, vks, w, w1, t):
             a = (vks - 1j * w1)
             b = (vks - 1j * w)
             omega_diff = w - w1

             # --- FIX START ---
             # term1 = jnp.divide(cks * jnp.exp(-b * t), a * b, where=(a * b) != 0, fill_value=0) # Original Incorrect
             term1_den = a * b
             term1_raw = (cks * jnp.exp(-b * t)) / term1_den
             term1 = jnp.where(
                  term1_den != 0,
                  term1_raw,
                  jnp.array(0.0, dtype=self.dtype) # Ensure correct dtype
             )

             # term2 = jnp.conjugate(cks) * jnp.exp(-jnp.conjugate(a) * t) / (jnp.conjugate(a) * jnp.conjugate(b)) # This line wasn't using jnp.divide before, but check if it needs handling
             term2_den = jnp.conjugate(a) * jnp.conjugate(b)
             term2_raw = (jnp.conjugate(cks) * jnp.exp(-jnp.conjugate(a) * t)) / term2_den
             term2 = jnp.where(
                 term2_den != 0,
                 term2_raw,
                 jnp.array(0.0, dtype=self.dtype) # Ensure correct dtype
             )


             # inv_omega_diff = jnp.divide(1j, omega_diff, where=omega_diff != 0, fill_value=jnp.array(jnp.inf, dtype=jnp.complex128) * jnp.sign(omega_diff)) # Original Incorrect
             inv_omega_diff_raw = 1j / omega_diff
             inv_omega_diff = jnp.where(
                 omega_diff != 0,
                 inv_omega_diff_raw,
                 jnp.array(jnp.inf, dtype=jnp.complex128) * jnp.sign(omega_diff) # Use original fill value
             )

             # term3 = cks * (jnp.divide(1, b, where=b != 0, fill_value=0) - jnp.divide(jnp.exp(1j * omega_diff * t), a, where=a != 0, fill_value=0)) # Original Incorrect
             term3_part1_raw = 1 / b
             term3_part1 = jnp.where(b != 0, term3_part1_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype

             term3_part2_raw = jnp.exp(1j * omega_diff * t) / a
             term3_part2 = jnp.where(a != 0, term3_part2_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype
             term3 = cks * (term3_part1 - term3_part2)

             # term4 = jnp.conjugate(cks) * (jnp.divide(1, jnp.conjugate(a), where=jnp.conjugate(a) != 0, fill_value=0) - jnp.divide(jnp.exp(1j * omega_diff * t), jnp.conjugate(b), where=jnp.conjugate(b) != 0, fill_value=0)) # Original Incorrect
             term4_part1_raw = 1 / jnp.conjugate(a)
             term4_part1 = jnp.where(jnp.conjugate(a) != 0, term4_part1_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype

             term4_part2_raw = jnp.exp(1j * omega_diff * t) / jnp.conjugate(b)
             term4_part2 = jnp.where(jnp.conjugate(b) != 0, term4_part2_raw, jnp.array(0.0, dtype=self.dtype)) # Ensure correct dtype
             term4 = jnp.conjugate(cks) * (term4_part1 - term4_part2)
             # --- FIX END ---

             actual = term1 - term2 + inv_omega_diff * (term3 - term4)

             return jnp.sum(actual) / 2j

        return jnp.where(
             jnp.isclose(w, w1),
             ls_w_eq_w1_branch(cks, vks, w, t),
             ls_w_ne_w1_branch(cks, vks, w, w1, t)
         )

    # Modified LS to work with static combinations and JAX arrays, remove dicts/lists - JAX compatible
    def LS(self, combinations_indices, bath, t, jump_ops_data, ws_data):
         def compute_ls_rate(combo_indices):
             idx_i, idx_j = combo_indices
             w_i = ws_data[idx_i]
             w_j = ws_data[idx_j]
             return 2 * self._LS_single(bath, w_i, w_j, t)

         ls_rates_array = jax_map(compute_ls_rate, combinations_indices)
         return ls_rates_array

    # Modified decays to work with static combinations and JAX arrays, remove dicts/lists - JAX compatible
    def decays(self, combinations_indices, bath, t, jump_ops_data, ws_data):
         def compute_decay_rate(combo_indices):
             idx_i, idx_j = combo_indices
             w_i = ws_data[idx_i]
             w_j = ws_data[idx_j]
             return self.gamma_gen(bath, w_i, w_j, t)

         decay_rates_array = jax_map(compute_decay_rate, combinations_indices)
         return decay_rates_array


    # Modified matrix_form to work with static indices and JAX arrays, remove dicts/lists - JAX compatible
    def matrix_form(self, combinations_indices, jump_ops_data):
         def compute_matrices(combo_indices):
             idx_i, idx_j = combo_indices
             jump_i = Qobj(jump_ops_data[idx_i])
             jump_j = Qobj(jump_ops_data[idx_j])

             ada = jump_i.dag() * jump_j
             matrixform_op = (
                 spre(jump_j) * spost(jump_i.dag())
                 - 0.5 * (spre(ada) + spost(ada))
             )
             lsform_op = -1j * (spre(ada) - spost(ada))

             return matrixform_op.data, lsform_op.data

         matrixform_data_array, lsform_data_array = jax_map(compute_matrices, combinations_indices)
         return matrixform_data_array, lsform_data_array


    # Modified generator function to use static data and JAX-compatible helpers,
    # and return a spre object - JAX compatible
    # @jit # You can add @jit here once debugging is done
    def generator(self, t):
        # t is a single time point (JAX array) for this generator instance call

        total_superop_data = spre(jnp.zeros_like(self.Hsys.data)).data * 0 # Start with zero superoperator data

        # Iterate over pre-calculated baths and their jump operators (static structure)
        # Using a for loop over a static range is OK in JAX tracing
        for bath_idx in range(len(self.baths)):
             bath = self.baths[bath_idx]
             jump_ops_data = self._static_jumps_by_bath_data[bath_idx]
             ws_data = self._static_ws_by_bath[bath_idx]

             # Create combinations of indices (static structure)
             num_jumps = len(ws_data)
             # This needs to generate a JAX array of combinations of indices
             # Assuming num_jumps is static for each bath based on initial Hsys/Qs
             combinations_indices = jnp.array(list(itertools.product(range(num_jumps), range(num_jumps))))

             # Compute rates and matrices using JAX-compatible functions
             rates_array = self.decays(combinations_indices, bath, t, jump_ops_data, ws_data)
             matrices_data_array, lsform_data_array = self.matrix_form(combinations_indices, jump_ops_data)

             # Sum over combinations using JAX operations
             if not self.ls:
                 bath_superop_data = jnp.sum(matrices_data_array * rates_array[:, None, None], axis=0)
             else:
                 ls_rates_array = self.LS(combinations_indices, bath, t, jump_ops_data, ws_data)
                 bath_superop_data = jnp.sum(
                     lsform_data_array * ls_rates_array[:, None, None]
                     + matrices_data_array * rates_array[:, None, None],
                     axis=0
                 )

             total_superop_data += bath_superop_data # Functional update

        # *** MODIFICATION FOR GRAD ***
        # Return a spre object wrapping the calculated superoperator data
        return spre(total_superop_data, kron=False)


    # The evolution function is not JAX-traceable due to Python loop and tqdm.
    # Keep it as is for now, assuming grad is only on the generator part.
    def evolution(self, rho0):
        r"""
        This function computes the evolution of the state $\rho(0)$
        (Note: This function is not JAX-traceable due to the Python loop and tqdm)
        """
        # Ensure rho0 is a Qobj
        rho0_qobj = rho0 if isinstance(rho0, Qobj) else Qobj(rho0)

        states = []
        # This loop runs in Python, outside of JAX transformations
        for t_point in tqdm(self.t, desc='Computing Evolution . . . .'):
            # self.generator(t_point) now returns a spre object
            superoperator = self.generator(t_point)
            # Compute the time evolution operator expm(L * t) - expm returns a spre
            evolution_operator = superoperator.expm()

            # Apply the evolution operator to the initial state rho0 - __call__ returns a Qobj
            rho_t = evolution_operator(rho0_qobj)

            states.append(rho_t) # Appending to Python list is fine here

        return states

    # Keep original names pointing to the JAX-compatible internal helpers
    def decayww(self, bath, w, t):
         return self._decayww(bath, w, t)

    def decayww2(self, bath ,w, w1, t):
         return self._decayww2( bath, w, w1, t)

    def LS(self, combinations_indices, bath, t, jump_ops_data, ws_data):
         return self._LS(combinations_indices, bath, t, jump_ops_data, ws_data)


tree_util.register_pytree_node(
    jaxcsolve2,
    jaxcsolve2._tree_flatten,
    jaxcsolve2._tree_unflatten)

# Register PyTrees for bath components if they are custom classes with JAX arrays
# (Assuming 'bath' object itself is static or its attributes are JAX arrays/basic types
# that don't require custom PyTree registration based on how they are used here).