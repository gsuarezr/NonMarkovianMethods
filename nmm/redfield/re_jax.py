import numpy as np
import jax.numpy as jnp
from nmm.utils.utils import spre as spre
from nmm.utils.utils import spost as spost
from nmm.utils.utils import Qobj as Qobj
from multipledispatch import dispatch
from jax import tree_util
import jax
import diffrax
import functools
jax.config.update("jax_enable_x64", True)

class JAXGKLS:
    def __init__(self, Hsys, t, baths ,Qs,picture="S",LS=True):
        self.Hsys = Hsys
        self.t = t
        self.Qs = Qs        
        self.baths = baths        
        self.n = Hsys.data.shape[1]
        self.picture = picture
        self.LS = LS
    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.Qs)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def jump_operators(self, Q, t=None):
        # 1. Get Eigenvalues and Eigenvectors
        H_matrix = self.Hsys(t).data if callable(self.Hsys) else self.Hsys.data
        evals, evecs = jnp.linalg.eigh(H_matrix)
        # 2. Transform Q to energy basis
        Q_data = Q.data if hasattr(Q, 'data') else Q
        Q_eb = jnp.conj(evecs.T) @ Q_data @ evecs
        
        # 3. Create frequency matrix
        bohr_frequencies = evals[None, :] - evals[ :, None]
        
        # 4. Round and find unique frequencies
        ws= bohr_frequencies.flatten()
        ws_rounded = jnp.round(ws, 12)
        
        # Get unique frequencies and an index mapping
        # size=n**2 ensures the output shape is static for JIT
        unique_ws, inverse_indices = jnp.unique(
            ws_rounded, return_inverse=True, size=self.n**2, fill_value=0.0
        )

        # 5. Group and Sum operators with the same frequency
        # We create a 1D array of the Q_eb elements and use segment_sum
        # to group those that share a frequency.
        summed_amplitudes = jax.ops.segment_sum(
            Q_eb.flatten(), 
            inverse_indices, 
            num_segments=self.n**2
        )

        # 6. Reconstruct the Operators 
        # Since we need a "list" of operators, we return a 3D array: (Frequency_Index, N, N)
        # Each slice [i, :, :] is the sum of jump operators for that unique frequency.
        def get_single_op(i):
            # Create a sparse-like matrix in energy basis for this specific frequency
            # Each index k where inverse_indices[k] == i belongs to this frequency
            mask = (inverse_indices == i).reshape(self.n, self.n)
            op_eb = jnp.where(mask, Q_eb, 0.0)
            # Transform back to lab basis: A_w = U @ op_eb @ U.dag
            return evecs @ op_eb @ jnp.conj(evecs.T)

        # Vmap over the number of unique frequencies
        all_summed_operators = jax.vmap(get_single_op)(jnp.arange(self.n**2))
        
        # unique_ws: array of frequencies (the keys)
        # all_summed_operators: 3D array of matrices (the values)
        return unique_ws, all_summed_operators

    def decays(self, combinations, bath,approximated, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._gamma_gen(bath, i[1], i[0], t, approximated)
        return rates
    def matrix_form(self, jumps, combinations):
        matrixform = {}
        lsform= {}
        for i in combinations:
            ada=jumps[i[0]].dag()*jumps[i[1]]
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                (spre(ada) +spost(ada))))
            lsform[i]= 1j*(spre(ada)-spost(ada))
        return matrixform,lsform
        
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
        if np.isclose(w, 0).all():
            return 0
        return np.exp(-w / bath.T) / (1-np.exp(-w / bath.T))


    def decayww2_jax(self, bath, w_row, w_col, t):
        # cks and vks are arrays of shape (K,)
        cks = jnp.array([i.coefficient for i in bath.exponents])
        vks = jnp.array([i.exponent for i in bath.exponents])

        # No manual broadcasting needed! 
        # w_row and w_col are treated as scalars here because of vmap.
        t1_denom = vks - 1j * w_col
        t2_denom = jnp.conj(vks) + 1j * w_row
        
        term1 = cks / t1_denom
        term2 = jnp.conj(cks) / t2_denom
        
        exp1 = 1 - jnp.exp(-t1_denom * t)
        exp2 = 1 - jnp.exp(-t2_denom * t)
        
        # Sum over the K exponents
        result = jnp.sum(term1 * exp1 + term2 * exp2)

        if self.picture == "I":
            phase = jnp.exp(1j * (w_row - w_col) * t)
            return jnp.conj(result * phase)
        return jnp.conj(result)
    def _LS_jax(self, bath, w, w1, t):
            # 1. Extract the coefficients and exponents as clean JAX arrays.
            # Note: If bath.exponents changes size dynamically between runs, 
            # these should be padded or kept static to avoid re-compilation.
            cks = jnp.array([i.coefficient for i in bath.exponents], dtype=jnp.complex128)
            vks = jnp.array([i.exponent for i in bath.exponents], dtype=jnp.complex128)
            
            # 2. Vectorized calculation across all exponent items at once
            # Because cks and vks are arrays, all operations automatically broadcast.
            term1 = cks / (vks - 1j * w1)
            term2 = jnp.conj(cks) / (jnp.conj(vks) + 1j * w)
            
            term1 *= (1.0 - jnp.exp(-(vks - 1j * w1) * t))
            term2 *= (1.0 - jnp.exp(-(jnp.conj(vks) + 1j * w) * t))
            
            # 3. Sum across the vectorized axis cleanly in JAX
            total_sum = jnp.sum(term1 - term2)
            result = total_sum / (-2j)
            
            # 4. JAX-compliant conditional mapping
            # We use a standard Python if/else here because self.picture is a 
            # static configuration string known at compile time.
            if self.picture == "I":
                return jnp.conj(result * jnp.exp(1j * (w - w1) * t))
            else:
                return jnp.conj(result)
    def _generator_jax(self, Q , bath, t):
            n = self.n
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

            # 3. Compute Rate Tensors using nested vmap
            @jax.vmap
            def compute_all_rates(w_row):
                @jax.vmap
                def compute_single_pair(w_col):
                    gamma = self.decayww2_jax(bath, w_row, w_col, t)
                    shift = self._LS_jax(bath, w_row, w_col, t)
                    return gamma , shift
                return compute_single_pair(omega_flat)
            
            gamma_flat,shift_flat = compute_all_rates(omega_flat)
            gamma_tensor = gamma_flat.reshape(n, n, n, n)
            shift_tensor = shift_flat.reshape(n, n, n, n)
            # @jax.vmap
            # def map_j(w_j_row):          # Maps over j axis of omega_matrix
            #     @jax.vmap
            #     def map_k(w_jk):          # Maps over k axis of omega_matrix
            #         @jax.vmap
            #         def map_l(w_l_row):   # Maps over l axis of omega_matrix
            #             @jax.vmap
            #             def map_m(w_lm):  # Maps over m axis of omega_matrix
            #                 # Evaluate the rates using the exact physical frequency pairs
            #                 return self.decayww2_jax(bath, w_jk, w_lm, t),self._LS_jax(bath, w_jk, w_lm, t)
            #             return map_m(w_l_row)
            #         return map_l(omega_matrix)
            #     return map_k(w_j_row)
            # gamma_tensor,shift_tensor = map_j(omega_matrix)
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
        n = self.n
        # Initialize an empty superoperator matrix of zeros in the correct shape
        total_generator = jnp.zeros((n**2, n**2), dtype=jnp.complex128)
        
        # Use a standard loop to accumulate tensors dynamically within the JAX trace.
        # JAX will cleanly trace each pass as an add operation on the matrix graph.
        for Q, bath in zip(self.Qs, self.baths):
            total_generator += self._generator_jax(Q, bath, t)
        if self.picture=="S":
            I = jnp.eye(self.n, dtype=jnp.complex128)
            total_generator+=1j *  (jnp.kron(I, self.Hsys.data) - jnp.kron(self.Hsys.data.T, I))
        return total_generator

    @functools.partial(jax.jit, static_argnums=(0,))
    def evolution(self, rho0, t):
        """
        rho_init: Initial N x N density matrix (QuTiP Qobj or raw array)
        """
        # Flatten the initial density matrix into a 1D JAX vector
        y0 = rho0.data.flatten()
        def system_model( t, y, args):
            """
            t: scalar time
            y: flattened rho vector (shape: (N^2,))
            args: any static/dynamic parameters you want to pass
            """
            print("!!! JAX IS TRACING THIS STEP !!!") # Standard Python print, NOT jnp.print
            L = self.generator_jax(t)
            dydt = L @ y
            return dydt
        # Initialize the ODE term
        physics_term = diffrax.ODETerm(system_model)
        
        # Select a solver. 
        solver = diffrax.Tsit5()
        
        # Define execution boundaries and saving grid
        saveat = diffrax.SaveAt(ts=t)
        # 2. Wrap the core solver call in a local JIT function
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
        
        # 3. Execute the jitted solver
        sol = diffrax.diffeqsolve(
                physics_term,
                solver,
                t0=t[0],
                t1=t[-1],
                dt0=t[1]-t[0],
                y0=y0,
                saveat=diffrax.SaveAt(ts=t),
                max_steps=1_000_000_000,
                stepsize_controller=stepsize_controller
            )
        
        # Reshape the results back from (Num_steps, N^2) to (Num_steps, N, N)
        rho_trajectory = sol.ys.reshape(-1, self.n, self.n)
        
        return rho_trajectory

tree_util.register_pytree_node(
JAXGKLS,
JAXGKLS._tree_flatten,
JAXGKLS._tree_unflatten)


class JAXCUM:
    def __init__(self, Hsys, t, baths ,Qs,picture="S",LS=True):
        self.Hsys = Hsys
        self.t = t
        self.Qs = Qs        
        self.baths = baths        
        self.n = Hsys.data.shape[1]
        self.picture = picture
        self.LS = LS
    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.Qs)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def jump_operators(self, Q, t=None):
        # 1. Get Eigenvalues and Eigenvectors
        H_matrix = self.Hsys(t).data if callable(self.Hsys) else self.Hsys.data
        evals, evecs = jnp.linalg.eigh(H_matrix)
        # 2. Transform Q to energy basis
        Q_data = Q.data if hasattr(Q, 'data') else Q
        Q_eb = jnp.conj(evecs.T) @ Q_data @ evecs
        
        # 3. Create frequency matrix
        bohr_frequencies = evals[None, :] - evals[ :, None]
        
        # 4. Round and find unique frequencies
        ws=  bohr_frequencies.flatten()
        ws_rounded = jnp.round(ws, 12)
        
        # Get unique frequencies and an index mapping
        # size=n**2 ensures the output shape is static for JIT
        unique_ws, inverse_indices = jnp.unique(
            ws_rounded, return_inverse=True, size=self.n**2, fill_value=0.0
        )

        # 5. Group and Sum operators with the same frequency
        # We create a 1D array of the Q_eb elements and use segment_sum
        # to group those that share a frequency.
        summed_amplitudes = jax.ops.segment_sum(
            Q_eb.flatten(), 
            inverse_indices, 
            num_segments=self.n**2
        )

        # 6. Reconstruct the Operators 
        # Since we need a "list" of operators, we return a 3D array: (Frequency_Index, N, N)
        # Each slice [i, :, :] is the sum of jump operators for that unique frequency.
        def get_single_op(i):
            # Create a sparse-like matrix in energy basis for this specific frequency
            # Each index k where inverse_indices[k] == i belongs to this frequency
            mask = (inverse_indices == i).reshape(self.n, self.n)
            op_eb = jnp.where(mask, Q_eb, 0.0)
            # Transform back to lab basis: A_w = U @ op_eb @ U.dag
            return evecs @ op_eb @ jnp.conj(evecs.T)

        # Vmap over the number of unique frequencies
        all_summed_operators = jax.vmap(get_single_op)(jnp.arange(self.n**2))
        
        # unique_ws: array of frequencies (the keys)
        # all_summed_operators: 3D array of matrices (the values)
        return unique_ws, all_summed_operators

    def decays(self, combinations, bath,approximated, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._gamma_gen(bath, i[1], i[0], t, approximated)
        return rates
    def matrix_form(self, jumps, combinations):
        matrixform = {}
        lsform= {}
        for i in combinations:
            ada=jumps[i[0]].dag()*jumps[i[1]]
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                (spre(ada) +spost(ada))))
            lsform[i]= 1j*(spre(ada)-spost(ada))
        return matrixform,lsform
        
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
        if np.isclose(w, 0).all():
            return 0
        return np.exp(-w / bath.T) / (1-np.exp(-w / bath.T))

    def _decay_jax(self, bath, w, w1, t):
        cks_all = np.array([i.coefficient for i in bath.exponents], dtype=np.complex128)
        vks_all = np.array([i.exponent for i in bath.exponents], dtype=np.complex128)
        
        cks = jnp.array(cks_all)
        vks = jnp.array(vks_all)
        
        # --- BRANCH 1: w != w1 ---
        a = vks - 1j * w1
        b = vks - 1j * w
        
        term1 = cks * jnp.exp(-b * t) / (a * b)
        term2 = jnp.conj(cks) * jnp.exp(-jnp.conj(a) * t) / (jnp.conj(a) * jnp.conj(b))
        
        denom = jnp.where(jnp.abs(w - w1) < 1e-12, 1.0, w - w1)
        
        term3 = cks * ((1.0 / b) - (jnp.exp(1j * (w - w1) * t) / a))
        term4 = jnp.conj(cks) * ((1.0 / jnp.conj(a)) - (jnp.exp(1j * (w - w1) * t) / jnp.conj(b)))
        
        actual_diff = term1 + term2 + (1j * (term3 + term4) / denom)
        result_diff = jnp.sum(actual_diff)
        
        # --- BRANCH 2: w == w1 ---
        term1_same = (vks * t - 1j * w * t - 1.0) + jnp.exp(-(vks - 1j * w) * t)
        actual_same = term1_same * cks / (vks - 1j * w)**2
        result_same = 2.0 * jnp.real(jnp.sum(actual_same))
        
        # Branch choice based on frequency
        val = jnp.where(jnp.abs(w - w1) < 1e-12, result_same, result_diff)

        # --- CRITICAL FIX: Explicit t=0 Boundary Enforcement ---
        # Forces absolute zero if t is zero or close to machine epsilon, 
        # bypassing floating point noise or potential 0/0 edge cases inside JIT.
        return jnp.where(t <= 1e-14, 0.0 + 0.0j, val)


    def _LS_jax(self, bath, w, w1, t):
        cks_all = np.array([i.coefficient for i in bath.exponents], dtype=np.complex128)
        vks_all = np.array([i.exponent for i in bath.exponents], dtype=np.complex128)
        
        mask = np.imag(cks_all) >= 0
        cks = jnp.array(cks_all[mask])
        vks = jnp.array(vks_all[mask])
        
        # --- BRANCH 1: w != w1 ---
        a = vks - 1j * w1
        b = vks - 1j * w
        
        term1 = cks * jnp.exp(-b * t) / (a * b)
        term2 = jnp.conj(cks) * jnp.exp(-jnp.conj(a) * t) / (jnp.conj(a) * jnp.conj(b))
        
        denom = jnp.where(jnp.abs(w - w1) < 1e-12, 1.0, w - w1)
        
        term3 = cks * ((1.0 / b) - (jnp.exp(1j * (w - w1) * t) / a))
        term4 = jnp.conj(cks) * ((1.0 / jnp.conj(a)) - (jnp.exp(1j * (w - w1) * t) / jnp.conj(b)))
        
        actual_diff = term1 - term2 + (1j * (term3 - term4) / denom)
        result_diff = jnp.sum(actual_diff) / 2j
        
        # --- BRANCH 2: w == w1 ---
        term1_same = (vks * t - 1j * w * t - 1.0) + jnp.exp(-(vks - 1j * w) * t)
        actual_same = term1_same * cks / (vks - 1j * w)**2
        result_same = jnp.imag(jnp.sum(actual_same)) / 2.0
        
        # Branch choice based on frequency
        val = jnp.where(jnp.abs(w - w1) < 1e-12, result_same, result_diff)
        
        # --- CRITICAL FIX: Explicit t=0 Boundary Enforcement ---
        return jnp.where(t <= 1e-14, 0.0 + 0.0j, val)

    def _generator_jax(self, Q , bath, t):
            n = self.n
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
            omega_flat =  omega_matrix.flatten()

            # 3. Compute Rate Tensors using nested vmap
            @jax.vmap
            def compute_all_rates(w_row):
                @jax.vmap
                def compute_single_pair(w_col):
                    gamma = self._decay_jax(bath, w_col, w_row, t) # col, row for Kossakowski matrix convention
                    shift = self._LS_jax(bath, w_col, w_row, t)
                    return gamma , shift
                return compute_single_pair(omega_flat)
            
            gamma_flat,shift_flat = compute_all_rates(omega_flat)
            gamma_tensor = gamma_flat.reshape(n, n, n, n)
            shift_tensor = shift_flat.reshape(n, n, n, n)

            # 4. Construct Effective Matrices M 
            M_eb = jnp.einsum('jkjm,jk,jm->km', gamma_tensor, Q_eb,jnp.conj(Q_eb).T)
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
            L_left = jnp.kron(jnp.conj(evecs), evecs)
            L_right = jnp.kron(evecs.T, jnp.conj(evecs.T))
            
            # Perform the basis rotation matrix multiplication
            L_original_basis = L_left @ L_energy_basis @ L_right

            return  L_original_basis
    def generator_jax(self, t):
        n = self.n
        # Initialize an empty superoperator matrix of zeros in the correct shape
        total_generator = jnp.zeros((n**2, n**2), dtype=jnp.complex128)
        
        # Use a standard loop to accumulate tensors dynamically within the JAX trace.
        # JAX will cleanly trace each pass as an add operation on the matrix graph.
        for Q, bath in zip(self.Qs, self.baths):
            total_generator += self._generator_jax(Q, bath, t)
        return total_generator

    def _evolution(self, rho0, t):
        """
        rho_init: Initial N x N density matrix (QuTiP Qobj or raw array)
        """
        rho = jax.scipy.linalg.expm(self.generator_jax(t))@ rho0.data.flatten()
        return rho.reshape(self.n,self.n)
    @functools.partial(jax.jit, static_argnums=(0,))
    def evolution(self, rho0, t):
        """
        rho_init: Initial N x N density matrix (QuTiP Qobj or raw array)
        """
        if self.picture=="S":
            evo = lambda t: (-1j * self.Hsys * t).expm().data@self._evolution(rho0,t) @(1j * self.Hsys * t).expm().data
        else:
            evo = lambda t: self._evolution(rho0,t) 

        rho = jax.vmap(evo)(t)
        return rho

tree_util.register_pytree_node(
JAXCUM,
JAXCUM._tree_flatten,
JAXCUM._tree_unflatten)

class JAXGKLS2:
    def __init__(self, Hsys, t, baths, Qs, picture="S", LS=True):
        self.Hsys = Hsys
        self.t = t
        self.Qs = Qs        
        self.baths = baths        
        self.n = Hsys.data.shape[1]
        self.picture = picture
        self.LS = LS

    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.Qs)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def _precompute_static_baths(self):
        """Precomputes all time-independent terms for each Q and bath."""
        H_data = self.Hsys.data
        evals_raw, evecs_raw = jnp.linalg.eigh(H_data)
        idx = jnp.argsort(evals_raw)
        evals = evals_raw[idx]
        evecs = evecs_raw[:, idx]

        omega_matrix = evals[None, :] - evals[:, None]
        omega_flat = omega_matrix.flatten()

        static_bath_data = []
        for Q, bath in zip(self.Qs, self.baths):
            Q_orig = Q.data
            Q_eb = jnp.conj(evecs.T) @ Q_orig @ evecs

            cks = jnp.array([i.coefficient for i in bath.exponents], dtype=jnp.complex128)
            vks = jnp.array([i.exponent for i in bath.exponents], dtype=jnp.complex128)

            # Precompute static denominators
            t1_denom = vks[None, :] - 1j * omega_flat[:, None]      # (n^2, K)
            t2_denom = jnp.conj(vks)[None, :] + 1j * omega_flat[:, None] # (n^2, K)

            term1_static = cks[None, :] / t1_denom
            term2_static = jnp.conj(cks)[None, :] / t2_denom

            static_bath_data.append({
                'Q_eb': Q_eb,
                'vks': vks,
                'term1_static': term1_static,
                'term2_static': term2_static
            })

        return evecs, omega_matrix, omega_flat, evals, static_bath_data

    @functools.partial(jax.jit, static_argnums=(0,))
    def evolution(self, rho0, t):
        # 1. Precompute static/basis transformation elements
        evecs, omega_matrix, omega_flat, evals, static_bath_data = self._precompute_static_baths()

        # 2. Rotate initial state to energy basis (matching original L_right layout)
        rho0_eb = evecs.T @ rho0.data @ jnp.conj(evecs)

        # 3. Define the derivative in energy basis (shape: (n, n))
        def system_model_eb(t_val, rho_eb, args):
            drho_dt = jnp.zeros_like(rho_eb)

            # Precompute exp(1j * evals * t_val)
            e_val = jnp.exp(1j * evals * t_val)
            # Outer product to get E_matrix: E_matrix[j, k] = e_val[k] * conj(e_val[j])
            E_matrix = e_val[None, :] * jnp.conj(e_val[:, None])
            # Flatten to get exp_omega (which is E)
            exp_omega = E_matrix.flatten()

            # Hamiltonian term
            if self.picture == "S":
                drho_dt += 1j * omega_matrix * rho_eb

            # Accumulate contributions from each Q / bath
            for data in args:
                Q_eb = data['Q_eb']
                vks = data['vks']
                term1_static = data['term1_static']
                term2_static = data['term2_static']

                # Compute time-dependent exponentials in factorized form
                exp_vks = jnp.exp(-vks * t_val)
                exp1 = 1.0 - exp_omega[:, None] * exp_vks[None, :]
                exp2 = 1.0 - jnp.conj(exp_omega)[:, None] * jnp.conj(exp_vks)[None, :]

                # Sum over bath exponents
                S1 = jnp.sum(term1_static * exp1, axis=-1)  # (n^2,)
                S2 = jnp.sum(term2_static * exp2, axis=-1)  # (n^2,)

                # Construct gamma and shift tensors using fast outer sums
                gamma_flat = jnp.conj(S2[:, None] + S1[None, :]) # (n^2, n^2)
                
                if self.picture == "I":
                    # conj_phase = conj(E)[:, None] * E[None, :]
                    conj_phase = jnp.conj(exp_omega)[:, None] * exp_omega[None, :]
                    gamma_flat *= conj_phase

                gamma_tensor = gamma_flat.reshape(self.n, self.n, self.n, self.n)

                # Compute rate matrix M
                M_eb = jnp.einsum('jkjm,jk,jm->km', gamma_tensor, jnp.conj(Q_eb).T, Q_eb)

                # Anticommutator part
                drho_dt -= 0.5 * (M_eb.T @ rho_eb + rho_eb @ M_eb.T)

                # Sandwich part (directly in density matrix space)
                drho_dt += jnp.einsum('jklm,kj,lm,km->jl', gamma_tensor, jnp.conj(Q_eb), Q_eb, rho_eb)

                # Lamb shift part
                if self.LS:
                    result_ls = (S1[None, :] - S2[:, None]) / (-2j)
                    shift_flat = jnp.conj(result_ls)
                    if self.picture == "I":
                        shift_flat *= conj_phase
                    
                    shift_tensor = shift_flat.reshape(self.n, self.n, self.n, self.n)
                    S_matrix = jnp.einsum('jkjm,jk,jm->km', shift_tensor, jnp.conj(Q_eb).T, Q_eb)

                    drho_dt += 1j * (S_matrix.T @ rho_eb - rho_eb @ S_matrix.T)

            return drho_dt

        # 4. Solve the Diffeq in the energy basis
        physics_term = diffrax.ODETerm(system_model_eb)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

        sol = diffrax.diffeqsolve(
            physics_term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1] - t[0],
            y0=rho0_eb,  # Passes matrix y0 directly (Diffrax supports PyTree/matrix states!)
            saveat=diffrax.SaveAt(ts=t),
            max_steps=1_000_000_000,
            stepsize_controller=stepsize_controller,
            args=static_bath_data
        )

        # 5. Rotate the entire solution trajectory back to the original basis
        # sol.ys has shape (num_steps, n, n)
        rho_trajectory = jax.vmap(lambda r_eb: jnp.conj(evecs) @ r_eb @ evecs.T)(sol.ys)
        return rho_trajectory

tree_util.register_pytree_node(
JAXGKLS2,
JAXGKLS2._tree_flatten,
JAXGKLS2._tree_unflatten)


class JAXGKLS3:
    def __init__(self, Hsys, t, baths, Qs, picture="S", LS=True):
        self.Hsys = Hsys
        self.t = t
        self.Qs = Qs        
        self.baths = baths        
        self.n = Hsys.data.shape[1]
        self.picture = picture
        self.LS = LS

    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.Qs)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def _precompute_static_baths(self):
        """Precomputes all time-independent terms for each Q and bath."""
        H_data = self.Hsys.data
        evals_raw, evecs_raw = jnp.linalg.eigh(H_data)
        idx = jnp.argsort(evals_raw)
        evals = evals_raw[idx]
        evecs = evecs_raw[:, idx]

        omega_matrix = evals[None, :] - evals[:, None]
        omega_flat = omega_matrix.flatten()

        static_bath_data = []
        for Q, bath in zip(self.Qs, self.baths):
            Q_orig = Q.data
            Q_eb = jnp.conj(evecs.T) @ Q_orig @ evecs

            cks = jnp.array([i.coefficient for i in bath.exponents], dtype=jnp.complex128)
            vks = jnp.array([i.exponent for i in bath.exponents], dtype=jnp.complex128)

            # Precompute static denominators
            t1_denom = vks[None, :] - 1j * omega_flat[:, None]      # (n^2, K)
            t2_denom = jnp.conj(vks)[None, :] + 1j * omega_flat[:, None] # (n^2, K)

            term1_static = cks[None, :] / t1_denom
            term2_static = jnp.conj(cks)[None, :] / t2_denom

            static_bath_data.append({
                'Q_eb': Q_eb,
                'vks': vks,
                'term1_static': term1_static,
                'term2_static': term2_static
            })

        # Stack into a single PyTree of arrays
        stacked_bath_data = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *static_bath_data)
        return evecs, omega_matrix, evals, stacked_bath_data

    @functools.partial(jax.jit, static_argnums=(0,))
    def evolution(self, rho0, t):
        # 1. Precompute static/basis transformation elements
        evecs, omega_matrix, evals, stacked_bath_data = self._precompute_static_baths()

        # 2. Rotate initial state to energy basis (matching original L_right layout)
        rho0_eb = evecs.T @ rho0.data @ jnp.conj(evecs)

        # 3. Define the derivative in energy basis (shape: (n, n))
        def system_model_eb(t_val, rho_eb, args):
            drho_dt = jnp.zeros_like(rho_eb)

            # Precompute exp(1j * evals * t_val) and exp_omega
            e_val = jnp.exp(1j * evals * t_val)
            E_phase = e_val[None, :] * jnp.conj(e_val[:, None]) # E_phase[j, k] = e_val[k] * conj(e_val[j])
            exp_omega = E_phase.flatten()

            if self.picture == "I":
                E = E_phase
                conj_phase = jnp.conj(exp_omega)[:, None] * exp_omega[None, :]
            else:
                E = jnp.ones((self.n, self.n), dtype=jnp.complex128)
                # Hamiltonian term
                if self.picture == "S":
                    drho_dt += 1j * omega_matrix * rho_eb

            # Define single bath derivative function for vmap
            def single_bath_derivative(data):
                Q_eb = data['Q_eb']
                vks = data['vks']
                term1_static = data['term1_static']
                term2_static = data['term2_static']

                # Compute time-dependent exponentials in factorized form
                exp_vks = jnp.exp(-vks * t_val)
                exp1 = 1.0 - exp_omega[:, None] * exp_vks[None, :]
                exp2 = 1.0 - jnp.conj(exp_omega)[:, None] * jnp.conj(exp_vks)[None, :]

                # Sum over bath exponents
                S1 = jnp.sum(term1_static * exp1, axis=-1)  # (n^2,)
                S2 = jnp.sum(term2_static * exp2, axis=-1)  # (n^2,)

                # Reshape S1 and S2 to (n, n)
                G2 = jnp.conj(S2.reshape(self.n, self.n))
                G1 = jnp.conj(S1.reshape(self.n, self.n))

                # Apply time-dependent phases
                G2_prime = G2 * jnp.conj(E)
                G1_prime = G1 * E

                # Form intermediate matrix products
                X1 = E * Q_eb
                X2 = G1_prime * Q_eb
                Y1 = G2_prime * jnp.conj(Q_eb.T)
                Y2 = jnp.conj(E) * jnp.conj(Q_eb.T)

                # Compute rate matrix M_eb = Y1.T @ X1 + Y2.T @ X2
                M_eb = Y1.T @ X1 + Y2.T @ X2

                # Anticommutator and sandwich parts
                drho_dt_single = -0.5 * (M_eb.T @ rho_eb + rho_eb @ M_eb.T) + Y1 @ rho_eb @ X1.T + Y2 @ rho_eb @ X2.T

                # Lamb shift part
                if self.LS:
                    # G2_ls = conj(S2) / -2j, G1_ls = conj(S1) / 2j
                    G2_ls = G2 / (-2j)
                    G1_ls = G1 / (2j)
                    G2_prime_ls = G2_ls * jnp.conj(E)
                    G1_prime_ls = G1_ls * E
                    
                    Y1_ls = G2_prime_ls * jnp.conj(Q_eb.T)
                    X2_ls = G1_prime_ls * Q_eb
                    S_matrix = Y1_ls.T @ X1 + Y2.T @ X2_ls

                    drho_dt_single += 1j * (S_matrix.T @ rho_eb - rho_eb @ S_matrix.T)

                return drho_dt_single

            # Vectorize over all baths (along axis 0) and sum
            drho_dt_all = jax.vmap(single_bath_derivative)(args)
            drho_dt += jnp.sum(drho_dt_all, axis=0)

            return drho_dt

        # 4. Solve the Diffeq in the energy basis
        physics_term = diffrax.ODETerm(system_model_eb)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

        sol = diffrax.diffeqsolve(
            physics_term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1] - t[0],
            y0=rho0_eb,  # Passes matrix y0 directly (Diffrax supports PyTree/matrix states!)
            saveat=diffrax.SaveAt(ts=t),
            max_steps=1_000_000_000,
            stepsize_controller=stepsize_controller,
            args=stacked_bath_data
        )

        # 5. Rotate the entire solution trajectory back to the original basis
        # sol.ys has shape (num_steps, n, n)
        rho_trajectory = jax.vmap(lambda r_eb: jnp.conj(evecs) @ r_eb @ evecs.T)(sol.ys)
        return rho_trajectory

        # 4. Solve the Diffeq in the energy basis
        physics_term = diffrax.ODETerm(system_model_eb)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

        sol = diffrax.diffeqsolve(
            physics_term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1] - t[0],
            y0=rho0_eb,  # Passes matrix y0 directly (Diffrax supports PyTree/matrix states!)
            saveat=diffrax.SaveAt(ts=t),
            max_steps=1_000_000_000,
            stepsize_controller=stepsize_controller,
            args=static_bath_data
        )

        # 5. Rotate the entire solution trajectory back to the original basis
        # sol.ys has shape (num_steps, n, n)
        rho_trajectory = jax.vmap(lambda r_eb: jnp.conj(evecs) @ r_eb @ evecs.T)(sol.ys)
        return rho_trajectory

tree_util.register_pytree_node(
JAXGKLS3,
JAXGKLS3._tree_flatten,
JAXGKLS3._tree_unflatten)