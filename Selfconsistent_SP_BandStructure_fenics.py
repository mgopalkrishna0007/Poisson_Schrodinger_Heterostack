import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import dolfin as df
import warnings

# Suppress FEniCS and SLEPc warnings for a clean console output
warnings.filterwarnings("ignore")
df.set_log_level(df.LogLevel.WARNING)

# =============================================================================
# Helper Function: find_Ef_general
# =============================================================================
def find_Ef_general(Ec, Ev, psic, psiv, N_D_array, N_A_array, Vc_local, Vv_local, z, T, meff_e_array, meff_h_array, E_bind_D=0.0, E_bind_A=0.0, fully_ionized=True):
    """
    Computes the Fermi level (Ef) ensuring global charge neutrality.
    It balances 2D sheet carrier densities (electrons and holes) with 3D ionized impurities.
    """
    # Physical Constants
    e = 1.602176487E-19
    kB = 1.3806488E-23
    m0 = 9.10938188E-31
    hbar = 6.62606896E-34 / (2 * np.pi)

    # Thermal energy in eV (floor applied to avoid division by zero at 0K)
    kT_eV = (kB * T) / e if T > 0 else 1e-10

    # Calculate expectation value of the effective mass for each quantum state.
    # This accounts for wavefunction penetration into barriers with different masses.
    m_eff_n = np.array([np.trapz(meff_e_array * np.abs(psic[:, i])**2, x=z) for i in range(len(Ec))])
    m_eff_m = np.array([np.trapz(meff_h_array * np.abs(psiv[:, i])**2, x=z) for i in range(len(Ev))])
    
    # 2D Density of States (DOS) Prefactors for each subband
    pref_e = (m_eff_n * m0 * kB * T) / (np.pi * hbar**2)
    pref_h = (m_eff_m * m0 * kB * T) / (np.pi * hbar**2)

    def Q_residual(Ef):
        """Master charge neutrality equation. We seek Ef such that Q(Ef) = 0."""
        # 1. Closed-form sheet densities using Fermi-Dirac integrals for 2D systems
        N_n = pref_e * np.log(1 + np.exp((Ef - Ec) / kT_eV))
        P_m = pref_h * np.log(1 + np.exp((Ev - Ef) / kT_eV))
        
        # 2. Ionized Impurity Statistics (Incomplete vs Complete Ionization)
        if fully_ionized:
            ND_plus = N_D_array
            NA_minus = N_A_array
        else:
            ND_plus = N_D_array / (1 + 2 * np.exp((Ef - (Vc_local - E_bind_D)) / kT_eV))
            NA_minus = N_A_array / (1 + 0.5 * np.exp((Vv_local + E_bind_A - Ef) / kT_eV))
            
        # Total Net Charge = (Holes - Electrons) + (Ionized Donors - Ionized Acceptors)
        return np.sum(P_m) - np.sum(N_n) + np.trapz(ND_plus, x=z) - np.trapz(NA_minus, x=z)

    # Define wide search brackets spanning the entire potential landscape
    min_Ef = np.min(Vv_local) - 1.0 
    max_Ef = np.max(Vc_local) + 1.0

    # Solve for Ef using Brent's method (fast and robust root finding)
    try:
        Ef_sol = optimize.brentq(Q_residual, min_Ef, max_Ef)
    except ValueError:
        # Fallback to Bisection if the bracket needs expanding
        Ef_sol = optimize.bisect(Q_residual, min_Ef - 5.0, max_Ef + 5.0)

    # Re-evaluate final carrier profiles at the converged Ef
    N_n = pref_e * np.log(1 + np.exp((Ef_sol - Ec) / kT_eV))
    P_m = pref_h * np.log(1 + np.exp((Ev - Ef_sol) / kT_eV))
    
    if fully_ionized:
        ND_plus_z = N_D_array
        NA_minus_z = N_A_array
    else:
        ND_plus_z = N_D_array / (1 + 2 * np.exp((Ef_sol - (Vc_local - E_bind_D)) / kT_eV))
        NA_minus_z = N_A_array / (1 + 0.5 * np.exp((Vv_local + E_bind_A - Ef_sol) / kT_eV))

    net_integral_charge = Q_residual(Ef_sol)
    
    return Ef_sol, N_n, P_m, ND_plus_z, NA_minus_z, net_integral_charge

# =============================================================================
# FEniCS Solver: Schrodinger Equation 
# =============================================================================
def Schroed1D_dolfin(V_space, Vtot_array, Mass_array, n_eig):
    """
    Solves the 1D Time-Independent Schrodinger Equation using the Finite Element Method.
    Transforms the continuous differential equation into an algebraic eigenvalue problem.
    """
    mesh = V_space.mesh()
    u = df.TrialFunction(V_space)
    v = df.TestFunction(V_space)
    
    # Physical Constants
    h = 6.62606896E-34       
    hbar = h / (2 * np.pi)
    e = 1.602176487E-19      
    m0 = 9.10938188E-31  

    # Map numpy arrays to FEniCS Function objects over the mesh
    v2d = df.vertex_to_dof_map(V_space)
    Vtot_func = df.Function(V_space)
    Vtot_func.vector()[v2d] = Vtot_array
    
    Mass_func = df.Function(V_space)
    Mass_func.vector()[v2d] = Mass_array

    # Scale by 1e9 (1/nm) to keep matrices near O(1) preventing floating-point underflow
    scale = df.Constant(1e9)
    coeff_val = hbar**2 / (2 * m0 * e)
    
    # Variational Formulation of Hamiltonian: H = - (hbar^2 / 2m*) * d^2/dz^2 + V(z)
    a = scale * df.Constant(coeff_val) * (1.0 / Mass_func) * df.inner(df.grad(u), df.grad(v)) * df.dx + scale * Vtot_func * u * v * df.dx
    m_form = scale * u * v * df.dx

    # Assemble FEniCS matrices
    A = df.PETScMatrix()
    M_mat = df.PETScMatrix()
    df.assemble(a, tensor=A)
    df.assemble(m_form, tensor=M_mat)

    # Configure SLEPc EigenSolver (Shift-and-invert isolates the lowest energy states)
    solver = df.SLEPcEigenSolver(A, M_mat)
    solver.parameters['problem_type'] = 'gen_hermitian'
    solver.parameters['spectral_transform'] = 'shift-and-invert'
    solver.parameters['spectral_shift'] = np.min(Vtot_array) - 0.01 
    
    solver.solve(n_eig)

    nconv = solver.get_number_converged()
    if nconv < n_eig:
        n_eig = nconv

    E = []
    psi_array = np.zeros((len(Vtot_array), n_eig))
    
    # Extract eigenvalues (energy) and eigenvectors (wavefunctions)
    for i in range(n_eig):
        r, _, rx, _ = solver.get_eigenpair(i)
        E.append(r) 

        eig_func = df.Function(V_space)
        eig_func.vector()[:] = rx
        psi_val = eig_func.vector().get_local()[v2d]

        # L2 Normalization of the wavefunction: Integral(|psi|^2) dz = 1
        z_array = mesh.coordinates()[:, 0]
        norm = np.sqrt(np.trapz(np.abs(psi_val)**2, x=z_array))
        psi_array[:, i] = psi_val / norm

    # Sort states from lowest energy to highest
    if len(E) > 0:
        idx = np.argsort(E)
        E = np.array(E)[idx]
        psi_array = psi_array[:, idx]

    return E, psi_array

# =============================================================================
# FEniCS Solver: Poisson Equation
# =============================================================================
def Poisson_dolfin(V_space, rho_net_array, Epsi_array):
    """
    Solves the 1D Poisson Equation: d/dz ( epsilon * dV/dz ) = - rho / epsilon_0
    Outputs the resultant band bending potential (in eV).
    """
    u = df.TrialFunction(V_space)
    v = df.TestFunction(V_space)

    e = 1.602176487E-19
    Epsi0 = 8.854187817620E-12

    # Map charge and permittivity arrays to FEniCS Functions
    v2d = df.vertex_to_dof_map(V_space)
    rho_func = df.Function(V_space)
    rho_func.vector()[v2d] = rho_net_array
    
    Epsi_func = df.Function(V_space)
    Epsi_func.vector()[v2d] = Epsi_array

    coeff = df.Constant(e / Epsi0)

    # Variational Form: inner product integrates by parts
    a = Epsi_func * df.inner(df.grad(u), df.grad(v)) * df.dx
    L = -coeff * rho_func * v * df.dx

    # Dirichlet Boundary Condition: Potential is pinned to 0 at the boundaries
    def boundary(x, on_boundary):
        return on_boundary

    bc = df.DirichletBC(V_space, df.Constant(0.0), boundary)

    Vs_func = df.Function(V_space)
    df.solve(a == L, Vs_func, bc)

    return Vs_func.vector().get_local()[v2d]

# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":
    e     = 1.602176487E-19
    Epsi0 = 8.854187817620E-12

    # Simulation Parameters
    Nloops  = 500
    tol     = 1e-10     # Convergence Tolerance
    n       = 5         # Number of subbands to calculate
    ScF     = 0.05      # Visual scaling factor for plotting wavefunctions
    T       = 300       # Temperature (K)
    
    plot_field = 1

    # --- NON-UNIFORM MESH PARAMETERS ---
    dz_coarse = 5e-10    # 5 Angstroms in bulk layers
    dz_fine   = 0.5e-10  # 0.5 Angstroms at interfaces
    int_width = 2e-9     # Apply fine mesh within 2nm of any interface

    # ==========================================================
    # RIGOROUS MATERIAL PARAMETERS (300K)
    # [0] GaAs, [1] In(0.2)Ga(0.8)As
    # Dictionary: {ID: [Ec, Ev, meff_e, meff_h, Epsi_r]}
    # ==========================================================
    E_bind_D = 0.005   
    E_bind_A = 0.025   
    fully_ionized_flag = False 

    MAT_PROPS = {
        0: [1.424, 0.000, 0.067, 0.50, 12.9], 
        1: [1.246, 0.096, 0.055, 0.45, 13.9]  
    }

    # Layer Stack: [Material_ID, Thickness (nm), Donor Doping (1e18 cm-3)]
    M = np.array([
        [0,   10, 0],
        [0,   1,  5],
        [0,   5,  0],
        [1,   15, 0], 
        [0,   5,  0],
        [0,   1,  5],
        [0,   10, 0]
    ])

    # 1. Dynamically Build Non-Uniform Z Mesh
    interfaces = np.cumsum([0.0] + [row[1] * 1e-9 for row in M])
    z_max = interfaces[-1]
    
    z_points = [0.0]
    curr_z = 0.0
    while curr_z < z_max - 1e-15:
        dist = np.min(np.abs(interfaces - curr_z))
        step = dz_fine if dist <= int_width else dz_coarse
        
        next_int = interfaces[interfaces > curr_z + 1e-15]
        if len(next_int) > 0 and curr_z + step > next_int[0]:
            step = next_int[0] - curr_z
            
        if step < 1e-14: step = 1e-14 
        curr_z += step
        z_points.append(curr_z)

    z = np.array(z_points)
    Nz = len(z)

    # 2. Assign Material Properties to Nodes
    V0_c = np.zeros(Nz)
    V0_v = np.zeros(Nz)
    meff_e_arr = np.zeros(Nz)
    meff_h_arr = np.zeros(Nz)
    Epsi_arr = np.zeros(Nz)
    Dop = np.zeros(Nz)

    for k in range(Nz):
        z_eval = z[k] + 1e-13 
        if z_eval >= z_max: z_eval = z_max - 1e-13
        
        layer_idx = np.searchsorted(interfaces, z_eval) - 1
        layer_idx = max(0, min(layer_idx, len(M)-1))
        
        mat_id = int(M[layer_idx, 0])
        nd = M[layer_idx, 2] * 1e18 * 1e6
        props = MAT_PROPS[mat_id]
        
        V0_c[k] = props[0]
        V0_v[k] = props[1]
        meff_e_arr[k] = props[2]
        meff_h_arr[k] = props[3]
        Epsi_arr[k] = props[4]
        Dop[k] = nd

    N_D_array = Dop
    N_A_array = np.zeros_like(Dop) # so we have a n type device here , purely donor doped 
    total_original_charge = np.trapz(N_D_array, x=z) + np.trapz(N_A_array, x=z)

    # Local Bandgap Array
    Eg_arr = V0_c - V0_v

    # Initialize FEniCS Function Space
    mesh = df.IntervalMesh(Nz - 1, z[0], z[-1])
    mesh.coordinates()[:, 0] = z 
    V_space = df.FunctionSpace(mesh, 'CG', 1)

    Vs = np.zeros_like(z)
    Vsold = np.copy(Vs)
    nloop = 1

    ErrVec = [1]
    sumVtotVec = [1]
    convergence_met = False

    print(f"--- Starting FEniCS Solver | {Nz} Mesh Nodes Evaluated ---")
    
    # Self-Consistent Field (SCF) Iteration Loop
    while nloop <= Nloops:
        # Predictor-Corrector Mixing: Prevents numerical oscillations by damping potential updates
        x = 0.5 
        Vbending = Vs * x + Vsold * (1 - x)
        
        # Apply band bending to static flatband profiles
        Vtot_c = V0_c + Vbending
        Vtot_v = V0_v + Vbending 
        
        # Schrodinger Solvers for Electrons and Holes
        Ec, psic = Schroed1D_dolfin(V_space, Vtot_c, meff_e_arr, n)
        
        # Holes seek highest potential, so we invert the valence band for the solver
        Ev_inv, psiv = Schroed1D_dolfin(V_space, -Vtot_v, meff_h_arr, n)
        Ev = -np.array(Ev_inv) if len(Ev_inv) > 0 else np.array([])

        if len(Ec) == 0:
            print(f"Error: No electron eigenvalues converged at Loop {nloop}. Exiting.")
            break
            
        # Calculate Fermi Level based on charge neutrality
        Ef, N_n, P_m, ND_plus_z, NA_minus_z, net_integral_charge = find_Ef_general(
            Ec, Ev, psic, psiv, N_D_array, N_A_array, Vtot_c, Vtot_v, z, T, 
            meff_e_arr, meff_h_arr, E_bind_D, E_bind_A, fully_ionized_flag
        )
        
        # Map 2D sheet densities into full 3D spatial carrier profiles
        N_3D = np.sum(np.tile(N_n, (len(z), 1)) * np.abs(psic)**2, axis=1) if len(Ec) > 0 else np.zeros_like(z)
        P_3D = np.sum(np.tile(P_m, (len(z), 1)) * np.abs(psiv)**2, axis=1) if len(Ev) > 0 else np.zeros_like(z)
        rho_net = P_3D - N_3D + ND_plus_z - NA_minus_z
        
        # Solve Poisson equation to update internal electrostatic potential
        Vsold = np.copy(Vs)
        Vs = Poisson_dolfin(V_space, rho_net, Epsi_arr)
            
        Err = abs(1 - sumVtotVec[-1] / (np.sum(Vs) + 1e-20))
        sumVtotVec.append(np.sum(Vs))
        ErrVec.append(Err)

        # Diagnostics Output
        charge_ratio = abs(net_integral_charge / total_original_charge) if total_original_charge != 0 else 0
        Ec_str = ", ".join([f"{val:.4f}" for val in Ec[:min(3, len(Ec))]])
        Ev_str = ", ".join([f"{val:.4f}" for val in Ev[:min(3, len(Ev))]])
        
        print(f"\nLoop {nloop:02d} | Fermi Level (Ef) = {Ef:.4f} eV")
        print(f"         | E_c(1,2,3) = [{Ec_str}] eV")
        print(f"         | E_v(1,2,3) = [{Ev_str}] eV")
        print(f"         | Net Integrated Charge (Q_res) = {net_integral_charge:.3e} m^-2")
        print(f"         | Original Total Charge (Donors) = {total_original_charge:.3e} m^-2")
        print(f"         | Evaluated Charge Ratio        = {charge_ratio:.2e}")
        print(f"         | Convergence Criterion (Err)   = {Err:.3e} (Tol: {tol:.1e})")
        
        if Err < tol:
            print(f"\n--- Convergence Reached at Loop {nloop} (Tolerance Met) ---")
            convergence_met = True
            break
            
        nloop += 1
        
    if not convergence_met:
        print(f"\n--- Stopped: Maximum Iterations Reached ({Nloops}) ---")

    # =============================================================================
    # Plotting Sections
    # =============================================================================
    if len(Ec) > 0:
        plt.figure(figsize=(10, 6), facecolor='w')
        ax = plt.subplot(1, 1, 1)
        
        # Plot Flatband Potentials
        ax.plot(z * 1e9, V0_c, 'b--', linewidth=1.0, alpha=0.5, label='E_c (Flatband)')
        ax.plot(z * 1e9, V0_v, 'r--', linewidth=1.0, alpha=0.5, label='E_v (Flatband)')

        # Plot Self-Consistent Potentials
        ax.plot(z * 1e9, Vtot_c, 'b-', linewidth=2.0, label='E_c (Self-Consistent)')
        ax.plot(z * 1e9, Vtot_v, 'r-', linewidth=2.0, label='E_v (Self-Consistent)')
        
        # Plot Fermi Level
        ax.plot([z[0]*1e9, z[-1]*1e9], [Ef, Ef], color='lime', linestyle='--', linewidth=2.0, label='E_f')
        ax.text(z[-1]*1e9*0.98, Ef + 0.02, f'{Ef:.3f} eV', color='lime', ha='right', va='bottom', fontweight='bold')

        # --- NEW: Plot Derived Ev from Bandgap ---
        # Ev_derived = Converged Ec - Spatially Varying Bandgap
        # Using markers every 50 nodes to avoid a solid overlapping blob
        Ev_derived = Vtot_c - Eg_arr
        ax.plot(z * 1e9, Ev_derived, color='darkorange', linestyle='-', marker='o', markersize=4, markevery=max(1, len(z)//50), label='E_v (Derived = E_c - E_g)')

        # Darker Wavefunctions styled as Dash-Dot (-.)
        for i in range(min(3, len(Ec))): 
            psi_scaled = np.abs(psic[:, i])**2 / np.max(np.abs(psic[:, i])**2) * ScF + Ec[i]
            ax.plot(z * 1e9, psi_scaled, color='darkcyan', linestyle='-.', linewidth=1.5)

        if len(Ev) > 0:
            for i in range(min(3, len(Ev))):
                psi_v_scaled = -(np.abs(psiv[:, i])**2 / np.max(np.abs(psiv[:, i])**2)) * ScF + Ev[i]
                ax.plot(z * 1e9, psi_v_scaled, color='darkmagenta', linestyle='-.', linewidth=1.5)

        # --- NEW: 1D Mesh Visualization ---
        # Position the mesh line dynamically below the lowest point of the valence band
        mesh_baseline = np.min(Vtot_v) - 0.15 
        
        # Draw a faint guide line
        ax.axhline(mesh_baseline, color='gray', linewidth=0.8, alpha=0.5) 
        
        # Plot every mesh node as a vertical tick mark
        ax.plot(z * 1e9, np.full_like(z, mesh_baseline), '|', color='black', markersize=8, alpha=0.7, label='1D Mesh Nodes')

        ax.set_xlim([0, z[-1] * 1e9])
        ax.set_ylim([np.min(Vtot_v) - 0.2, np.max(Vtot_c) + 0.2]) 

        ax.set_xlabel('z (nm)', fontsize=14)
        ax.set_ylabel('Energy (eV)', fontsize=14)
        ax.set_title(f"Type-I Quantum Well | Ratio Final: {charge_ratio:.1e}", fontsize=12)
        ax.legend(loc='center right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if plot_field == 1:
            F = e * integrate.cumulative_trapezoid(rho_net, z, initial=0) / (np.mean(Epsi_arr) * Epsi0)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(z * 1e9, F * 1e-2 * 1e-3, 'r')
            ax2.plot(z * 1e9, N_D_array * 1e-18 * 1e-6, 'b', linestyle='--')
            ax1.grid(True)
            ax1.set_xlabel('z (nm)')
            ax1.set_ylabel('E- field (kV/cm)', color='red')
            ax2.set_ylabel('Donor Doping (1e18 cm-3)', color='blue')
            ax1.tick_params(axis='y', colors='red')
            ax2.tick_params(axis='y', colors='blue')

        plt.show()
