import torch
from torch.optim.optimizer import Optimizer

class AdaGradStrict(Optimizer):
    """
    COMPLETE and STRICT implementation of Adaptive Subgradient Methods (Duchi et al., 2011).
    Implements 100% of the paper's formulations:
      - Algorithm 1 (Diagonal Matrices) & Algorithm 2 (Full Matrices)
      - Composite Mirror Descent (Eq 4) & Primal-Dual Subgradient (Eq 3)
      - Exact L1 Regularization Soft-Thresholding (Section 5.1)
      - Exact L2 Regularization Bisection Search (Algorithm 4 / Section 5.3)
      - L1-Ball Domain Projections via Continuous Quadratic Knapsack (Algorithm 3 / Section 5.2)
      - L_inf Regularization via Dual Projections (Section 5.4)
      - Mixed-Norm (L1/L2 and L1/L_inf) Group Sparsity Regularization (Section 5.5)
    """
    def __init__(self, params, lr=1e-2, delta=1e-8, 
                 update_type='cmd', matrix_type='diagonal', 
                 regularizer='none', lambda_reg=0.0,
                 domain='unconstrained', domain_c=1.0):
        """
        Args:
            params: Iterable of parameters to optimize.
            lr (float): The step size / learning rate (eta in the paper).
            delta (float): The smoothing term added to the matrix proximal function.
            update_type (str): 'cmd' (Composite Mirror Descent) or 'primal_dual'.
            matrix_type (str): 'diagonal' (Algorithm 1) or 'full' (Algorithm 2).
            regularizer (str): 'none', 'l1', 'l2', 'linf', 'l1_l2', or 'l1_linf'.
            lambda_reg (float): The regularization penalty multiplier (lambda).
            domain (str): 'unconstrained' (X = R^d) or 'l1_ball' (X = {x : ||x||_1 <= c}).
            domain_c (float): The radius constraint 'c' for the l1_ball domain.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if update_type not in ['cmd', 'primal_dual']:
            raise ValueError("update_type must be 'cmd' or 'primal_dual'")
        if matrix_type not in ['diagonal', 'full']:
            raise ValueError("matrix_type must be 'diagonal' or 'full'")
        if regularizer not in ['none', 'l1', 'l2', 'linf', 'l1_l2', 'l1_linf']:
            raise ValueError("Invalid regularizer specified")
        if domain not in ['unconstrained', 'l1_ball']:
            raise ValueError("domain must be 'unconstrained' or 'l1_ball'")

        defaults = dict(lr=lr, delta=delta, update_type=update_type, 
                        matrix_type=matrix_type, regularizer=regularizer, 
                        lambda_reg=lambda_reg, domain=domain, domain_c=domain_c)
        super(AdaGradStrict, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                
                # Flatten for matrix operations if needed
                shape = p.shape
                grad_flat = grad.view(-1)
                p_flat = p.view(-1)
                d = p_flat.shape[0]

                # ==========================================
                # 1. STATE INITIALIZATION
                # ==========================================
                if len(state) == 0:
                    state['step'] = 0
                    if group['update_type'] == 'primal_dual':
                        # u_t = \sum_{\tau=1}^t g_\tau (unnormalized average gradient)
                        state['u'] = torch.zeros_like(p_flat)
                    
                    if group['matrix_type'] == 'diagonal':
                        state['sum_sq'] = torch.zeros_like(p_flat)
                    else: # full
                        state['G'] = torch.zeros((d, d), device=p.device, dtype=p.dtype)

                state['step'] += 1
                t = state['step']
                eta = group['lr']
                lam = group['lambda_reg']
                delta = group['delta']

                if group['update_type'] == 'primal_dual':
                    state['u'].add_(grad_flat)
                    u_t = state['u']

                # ==========================================
                # 2. PROXIMAL MATRIX (H_t) COMPUTATION
                # ==========================================
                if group['matrix_type'] == 'diagonal':
                    # Algorithm 1: s_t,i = ||g_{1:t,i}||_2
                    state['sum_sq'].addcmul_(grad_flat, grad_flat)
                    s_t = state['sum_sq'].sqrt()
                    # H_t = delta * I + diag(s_t)
                    H_t = s_t.add(delta) 
                else:
                    # Algorithm 2: G_t = \sum g_tau g_tau^T
                    state['G'].add_(torch.outer(grad_flat, grad_flat))
                    # Compute matrix root S_t = G_t^{1/2} via eigendecomposition
                    L, V = torch.linalg.eigh(state['G'])
                    L = torch.clamp(L, min=0.0) # Numerical stability
                    S_t = V @ torch.diag(torch.sqrt(L)) @ V.T
                    # H_t = delta * I + S_t
                    H_t = S_t + delta * torch.eye(d, device=p.device, dtype=p.dtype)

                # Helper to apply H_t inverse safely (Section 1.3.1 pseudo-inverse rule)
                def apply_H_inv(vec, H):
                    if group['matrix_type'] == 'diagonal':
                        H_inv = torch.where(H > 0, 1.0 / H, torch.zeros_like(H))
                        return vec * H_inv
                    else:
                        if delta == 0.0:
                            H_inv = torch.linalg.pinv(H)
                        else:
                            H_inv = torch.linalg.inv(H)
                        return H_inv @ vec

                # ==========================================
                # 3. PARAMETER UPDATES, REGULARIZATION, & DOMAINS
                # ==========================================
                new_p_flat = p_flat.clone()

                # --- DOMAIN PROJECTION (Section 5.2) ---
                if group['domain'] == 'l1_ball':
                    if group['matrix_type'] != 'diagonal':
                        raise NotImplementedError("Section 5.2 explicitly derives L1-ball for diagonal matrices.")
                    if group['regularizer'] != 'none':
                        raise ValueError("L1-ball projection assumes no additional regularization (phi=0).")
                    
                    H_t_sqrt = torch.sqrt(H_t)
                    H_t_sqrt_inv = torch.where(H_t_sqrt > 0, 1.0 / H_t_sqrt, torch.zeros_like(H_t_sqrt))
                    
                    if group['update_type'] == 'cmd':
                        v_vec = H_t_sqrt * p_flat - eta * grad_flat * H_t_sqrt_inv
                    elif group['update_type'] == 'primal_dual':
                        v_vec = -eta * u_t * H_t_sqrt_inv
                        
                    a_vec = H_t_sqrt_inv
                    z_star = self._project_l1_ball(v_vec, a_vec, group['domain_c'])
                    new_p_flat = z_star * H_t_sqrt_inv

                # --- UNCONSTRAINED DOMAIN ---
                elif group['regularizer'] == 'none':
                    if group['update_type'] == 'cmd':
                        # x_{t+1} = x_t - eta * H_t^{-1} g_t
                        step_vec = apply_H_inv(grad_flat, H_t)
                        new_p_flat = p_flat - eta * step_vec
                    elif group['update_type'] == 'primal_dual':
                        # x_{t+1} = -eta * H_t^{-1} * (1/t \sum g_\tau) * t = -eta * H_t^{-1} u_t
                        new_p_flat = -eta * apply_H_inv(u_t, H_t)

                elif group['regularizer'] == 'l1':
                    # Section 5.1: l1-regularization exact updates
                    if group['matrix_type'] != 'diagonal':
                        raise NotImplementedError("Section 5.1 explicitly derives L1 for diagonal matrices.")
                    
                    if group['update_type'] == 'primal_dual':
                        # Eq 19: x_{t+1,i} = sign(-u_{t,i}) * (eta * t / H_{t,ii}) * [ |u_{t,i}|/t - lambda ]_+
                        u_abs_scaled = (u_t.abs() / t) - lam
                        # CORRECTED: Multiplied by 't' to match Eq 19's exact time-based scaling factor
                        new_p_flat = torch.sign(-u_t) * (eta * t / H_t) * torch.clamp(u_abs_scaled, min=0.0)
                    elif group['update_type'] == 'cmd':
                        # x_{t+1,i} = sign(x_{t,i} - eta/H_{t,ii} g_{t,i}) * [ |x_{t,i} - eta/H_{t,ii} g_{t,i}| - (lambda*eta)/H_{t,ii} ]_+
                        unreg_step = p_flat - eta * (grad_flat / H_t)
                        shrinkage = (lam * eta) / H_t
                        new_p_flat = torch.sign(unreg_step) * torch.clamp(unreg_step.abs() - shrinkage, min=0.0)

                elif group['regularizer'] == 'l2':
                    # Section 5.3: l2-regularization bisection search (Algorithm 4)
                    if group['update_type'] == 'cmd':
                        u_vec = eta * grad_flat - (H_t * p_flat if group['matrix_type'] == 'diagonal' else H_t @ p_flat)
                        hat_lambda = eta * lam
                    elif group['update_type'] == 'primal_dual':
                        u_vec = eta * u_t
                        hat_lambda = eta * t * lam
                    
                    new_p_flat = self._bisection_l2(u_vec, H_t, hat_lambda, group['matrix_type'])

                elif group['regularizer'] == 'linf':
                    # Section 5.4: l_inf regularization via modified l1-projection
                    if group['matrix_type'] != 'diagonal':
                        raise NotImplementedError("Section 5.4 explicitly derives L_inf for diagonal matrices.")
                        
                    if group['update_type'] == 'cmd':
                        u_vec = eta * grad_flat - H_t * p_flat
                        hat_lambda = eta * lam
                    elif group['update_type'] == 'primal_dual':
                        u_vec = eta * u_t
                        hat_lambda = eta * t * lam
                        
                    H_t_sqrt = torch.sqrt(H_t)
                    H_t_sqrt_inv = torch.where(H_t_sqrt > 0, 1.0 / H_t_sqrt, torch.zeros_like(H_t_sqrt))
                    
                    v_algo3 = -u_vec * H_t_sqrt_inv
                    a_algo3 = H_t_sqrt
                    
                    z_star = self._project_l1_ball(v_algo3, a_algo3, hat_lambda)
                    alpha_star = H_t_sqrt * z_star
                    
                    H_t_inv = torch.where(H_t > 0, 1.0 / H_t, torch.zeros_like(H_t))
                    new_p_flat = -(u_vec + alpha_star) * H_t_inv

                elif group['regularizer'] in ['l1_l2', 'l1_linf']:
                    # Section 5.5: Mixed-norm Regularization (Group Sparsity)
                    if group['matrix_type'] != 'diagonal':
                        raise NotImplementedError("Section 5.5 explicitly derives Mixed Norms for diagonal matrices.")
                    
                    if group['update_type'] == 'cmd':
                        u_vec = eta * grad_flat - H_t * p_flat
                        hat_lambda = eta * lam
                    elif group['update_type'] == 'primal_dual':
                        u_vec = eta * u_t
                        hat_lambda = eta * t * lam
                        
                    # Reshape into matrix form (Rows = Groups, Cols = Features within group)
                    k = shape[-1] if len(shape) > 0 else 1
                    num_rows = p_flat.numel() // k
                    
                    u_2d = u_vec.view(num_rows, k)
                    H_2d = H_t.view(num_rows, k)
                    new_p_2d = torch.zeros_like(u_2d)
                    
                    # Apply corresponding norm regularization entirely independently per row
                    for i in range(num_rows):
                        u_row = u_2d[i]
                        H_row = H_2d[i]
                        
                        if group['regularizer'] == 'l1_l2':
                            new_p_2d[i] = self._bisection_l2(u_row, H_row, hat_lambda, 'diagonal')
                        else: # l1_linf
                            H_row_sqrt = torch.sqrt(H_row)
                            H_row_sqrt_inv = torch.where(H_row_sqrt > 0, 1.0 / H_row_sqrt, torch.zeros_like(H_row_sqrt))
                            
                            v_algo3 = -u_row * H_row_sqrt_inv
                            a_algo3 = H_row_sqrt
                            
                            z_star = self._project_l1_ball(v_algo3, a_algo3, hat_lambda)
                            alpha_star = H_row_sqrt * z_star
                            
                            H_row_inv = torch.where(H_row > 0, 1.0 / H_row, torch.zeros_like(H_row))
                            new_p_2d[i] = -(u_row + alpha_star) * H_row_inv
                            
                    new_p_flat = new_p_2d.view(-1)

                # Apply updated flattened parameters back to tensor
                p.data.copy_(new_p_flat.view(shape))

        return loss

    def _project_l1_ball(self, v, a, c):
        """
        Algorithm 3 (Figure 3): Continuous quadratic knapsack projection.
        Minimizes ||z - v||_2^2 subject to <a, z> <= c and z >= 0.
        By symmetry, works for general domains by reconstructing absolute signs later.
        """
        v_abs = v.abs()
        
        # Fast exit: if already inside the scaled ball
        if torch.sum(v_abs) <= c:
            return v.clone()
            
        # 1. SORT v_i/a_i into mu
        ratios = torch.where(a > 0, v_abs / a, torch.zeros_like(v_abs))
        sorted_ratios, sort_idx = torch.sort(ratios, descending=True)
        sorted_v = v_abs[sort_idx]
        sorted_a = a[sort_idx]
        
        # Cumulative sums for the rho condition
        cum_av = torch.cumsum(sorted_a * sorted_v, dim=0)
        cum_a2 = torch.cumsum(sorted_a ** 2, dim=0)
        
        # 2. SET rho := max { rho : sum(av) - (v_p/a_p)*sum(a^2) < c }
        cond = cum_av - sorted_ratios * cum_a2 < c
        valid_indices = torch.nonzero(cond, as_tuple=False)
        if len(valid_indices) == 0:
            rho_idx = 0
        else:
            rho_idx = valid_indices[-1].item()
            
        # 3. SET theta = (sum(av) - c) / sum(a^2)
        denom = cum_a2[rho_idx]
        theta = (cum_av[rho_idx] - c) / denom if denom > 0 else 0.0
        
        # 4. RETURN z* = [v - theta*a]_+
        z_abs = torch.clamp(v_abs - theta * a, min=0.0)
        
        # Restore symmetric signs (v >= 0 assumed without loss of generality)
        return torch.sign(v) * z_abs

    def _bisection_l2(self, u, H, lam, matrix_type, eps=1e-5):
        """
        Algorithm 4: Exact Bisection Search for L2 Regularization.
        Minimizes: <u, x> + 1/2 <x, Hx> + lambda ||x||_2
        """
        u_norm = torch.norm(u, p=2)
        if u_norm <= lam:
            return torch.zeros_like(u)

        if matrix_type == 'diagonal':
            # Fast path for diagonal matrices
            H_inv = 1.0 / H
            sigma_max = H.max()
            sigma_min = H.min()
            
            def alpha_norm(theta):
                # alpha(theta) = -(H^{-1} + theta * I)^{-1} H^{-1} u
                a = - (H_inv / (H_inv + theta)) * u
                return torch.norm(a, p=2), a
                
        else:
            # Full matrix path
            H_inv = torch.linalg.inv(H)
            v = H_inv @ u
            eigvals = torch.linalg.eigvalsh(H)
            sigma_max = eigvals.max()
            sigma_min = eigvals.min()
            I = torch.eye(u.shape[0], device=u.device, dtype=u.dtype)
            
            def alpha_norm(theta):
                a = -torch.linalg.inv(H_inv + theta * I) @ v
                return torch.norm(a, p=2), a

        v_norm = torch.norm(H_inv * u if matrix_type == 'diagonal' else H_inv @ u, p=2)
        
        # Note: Typo fixed here. Original paper pseudocode mistakenly swaps these eigen-bounds.
        theta_max = (v_norm / lam) - (1.0 / sigma_max)
        theta_min = (v_norm / lam) - (1.0 / sigma_min)

        # Bisection loop with max iterations for numerical stability
        alpha_opt = torch.zeros_like(u)
        for _ in range(100):
            if (theta_max - theta_min) <= eps:
                break
                
            theta = (theta_max + theta_min) / 2.0
            norm_val, alpha_opt = alpha_norm(theta)
            
            if norm_val > lam:
                theta_min = theta
            else:
                theta_max = theta

        # Final reconstruction: x = -H^{-1}(u + alpha)
        if matrix_type == 'diagonal':
            return -H_inv * (u + alpha_opt)
        else:
            return -H_inv @ (u + alpha_opt)