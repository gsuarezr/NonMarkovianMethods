import jax.numpy as jnp
from jax import tree_util

class GKLS:
    def __init__(self, Hsys, t, Qs):
        self.Hsys = Hsys.data
        self.Qs = [Q.data for Q in Qs ] 

    def _tree_flatten(self):
        children = (self.Hsys, self.Qs, self.baths)
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

    def jump_operators(self, Q, t=None):
        # 1. Get Eigenvalues and Eigenvectors
        H_matrix = self.Hsys.data
        evals, evecs = jnp.linalg.eigh(H_matrix)
        # 2. Transform Q to energy basis
        Q_data = Q.data 
        Q_eb = jnp.conj(evecs.T) @ Q_data @ evecs
        # 3. Create frequency matrix 
        bohr_frequencies = evals[None, :] - evals[ :, None]
        # 4. Round and find unique frequencies, Here is where a partial secular 
        # approximation would take place
        ws= bohr_frequencies.flatten()
        ws_rounded = jnp.round(ws, 12)
        # Get unique frequencies and an index mapping
        # size=n**2 ensures the output shape is static
        unique_ws, inverse_indices = jnp.unique(
            ws_rounded, return_inverse=True, size=self.n**2, fill_value=0.0
        )
        # 5. Group and Sum operators with the same frequency
        summed_amplitudes = jax.ops.segment_sum(
            Q_eb.flatten(), 
            inverse_indices, 
            num_segments=self.n**2
        )

        # 6. Reconstruct the Operators 
        def get_single_op(i):
            # Create a sparse-like matrix in energy basis for this specific frequency
            # Each index k where inverse_indices[k] == i belongs to this frequency
            mask = (inverse_indices == i).reshape(self.n, self.n)
            op_eb = jnp.where(mask, Q_eb, 0.0)
            # Transform back from the energy basis: A_w = U @ op_eb @ U.dag
            return evecs @ op_eb @ jnp.conj(evecs.T)

        # Vmap over the number of unique frequencies
        all_summed_operators = jax.vmap(get_single_op)(jnp.arange(self.n**2))
        
        # unique_ws: array of frequencies (the keys)
        # all_summed_operators: 3D array of matrices (the values)
        return unique_ws, all_summed_operators

    def _generator_jax(self, Q , bath, t):
            # 1. Setup Basis and extract clean JAX arrays
            H_data = self.Hsys.data 
            evals_raw, evecs_raw = jnp.linalg.eigh(H_data)
            
            # Keep standard ascending sort order
            idx = jnp.argsort(evals_raw)
            evals = evals_raw[idx]
            evecs = evecs_raw[:, idx]
            
            # Ensure Q is a dense 2D matrix (n, n) in original basis
            Q_orig = Q.data 
            
            # Rotate Q to energy basis for the internal calculation
            Q_eb = jnp.conj(evecs.T) @ Q_orig @ evecs
            
            # 2. Map Frequencies
            omega_matrix = evals[None,:] - evals[:, None]
            omega_flat = omega_matrix.flatten()

            # 3. Compute Rate and Lamb-Shift Tensors using nested vmap
            @jax.vmap
            def compute_all_rates(w_row):
                @jax.vmap
                def compute_single_pair(w_col):
                    gamma = self.decays(bath, w_row, w_col, t)
                    shift = self.lambshift(bath, w_row, w_col, t)
                    return gamma , shift
                return compute_single_pair(omega_flat)
            
            gamma_flat,shift_flat = compute_all_rates(omega_flat)
            gamma_tensor = gamma_flat.reshape(self.n, self.n, self.n, self.n)
            shift_tensor = shift_flat.reshape(self.n, self.n, self.n, self.n)
        
            # 4. Construct Effective Matrices M (REVERTED TO WORKING INDICES)
            M_eb = jnp.einsum('jkjm,jk,jm->km', gamma_tensor, jnp.conj(Q_eb).T, Q_eb)
            # 5. Construct Superoperator Components (Energy Basis)
            I = jnp.eye(n, dtype=jnp.complex128)

            # Sandwich part: Using 'jlkm' layout which unrolls correctly for your vectorization format
            L_sandwich_tensor = jnp.einsum('jklm,jk,lm->jlkm', gamma_tensor, jnp.conj(Q_eb).T, Q_eb)
            L_sandwich_eb = L_sandwich_tensor.reshape(n*n, n*n)

            # Anti-commutator part
            L_anticomm_eb = -0.5 * (jnp.kron(I, M_eb) + jnp.kron(M_eb.T, I))
            # Lambshift part
            L_unitary = jnp.zeros_like(L_anticomm_eb)
            if self.LS is True:
                S = jnp.einsum('jkjm,jk,jm->km', shift_tensor , jnp.conj(Q_eb).T, Q_eb)
                L_unitary += -1j *  (jnp.kron(I, S) - jnp.kron(S.T, I))
            # Total Liouvillian in the Energy Basis
            L_energy_basis =  L_unitary + L_sandwich_eb + L_anticomm_eb 

            # 6. FIXED ROTATION BACK TO ORIGINAL BASIS
            # Because the layout was inverted block-wise (top-left vs bottom-right), 
            # we flip the Kronecker components of the transformation operators.
            L_left = jnp.kron(jnp.conj(evecs), evecs)
            L_right = jnp.kron(evecs.T, jnp.conj(evecs.T))
            
            # Perform the basis rotation matrix multiplication
            L_original_basis = L_left @ L_energy_basis @ L_right

            return  L_original_basis

    def generator_jax(self, t):
        # Initialize an empty superoperator matrix of zeros in the correct shape
        total_generator = jnp.zeros((self.n**2, self.n**2), dtype=jnp.complex128)
        
        # Use a standard loop to accumulate tensors dynamically within the JAX trace.
        # JAX will cleanly trace each pass as an add operation on the matrix graph.
        for Q, bath in zip(self.Qs, self.baths):
            total_generator += self._generator_jax(Q, bath, t)
        return total_generator

tree_util.register_pytree_node(
GKLS,
GKLS._tree_flatten,
GKLS._tree_unflatten)