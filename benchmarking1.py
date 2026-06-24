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
    e = 1.602176487E-19
    kB = 1.3806488E-23
    m0 = 9.10938188E-31
    hbar = 6.62606896E-34 / (2 * np.pi)

    kT_eV = (kB * T) / e if T > 0 else 1e-10

    m_eff_n = np.array([np.trapz(meff_e_array * np.abs(psic[:, i])**2, x=z) for i in range(len(Ec))])
    m_eff_m = np.array([np.trapz(meff_h_array * np.abs(psiv[:, i])**2, x=z) for i in range(len(Ev))])
    
    pref_e = (m_eff_n * m0 * kB * T) / (np.pi * hbar**2)
    pref_h = (m_eff_m * m0 * kB * T) / (np.pi * hbar**2)

    def Q_residual(Ef):
        N_n = pref_e * np.log(1 + np.exp((Ef - Ec) / kT_eV))
        P_m = pref_h * np.log(1 + np.exp((Ev - Ef) / kT_eV))
        
        if fully_ionized:
            ND_plus = N_D_array
            NA_minus = N_A_array
        else:
            ND_plus = N_D_array / (1 + 2 * np.exp((Ef - (Vc_local - E_bind_D)) / kT_eV))
            NA_minus = N_A_array / (1 + 0.5 * np.exp((Vv_local + E_bind_A - Ef) / kT_eV))
            
        return np.sum(P_m) - np.sum(N_n) + np.trapz(ND_plus, x=z) - np.trapz(NA_minus, x=z)

    min_Ef = np.min(Vv_local) - 1.0 
    max_Ef = np.max(Vc_local) + 1.0

    try:
        Ef_sol = optimize.brentq(Q_residual, min_Ef, max_Ef)
    except ValueError:
        Ef_sol = optimize.bisect(Q_residual, min_Ef - 5.0, max_Ef + 5.0)

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
    mesh = V_space.mesh()
    u = df.TrialFunction(V_space)
    v = df.TestFunction(V_space)
    
    h = 6.62606896E-34       
    hbar = h / (2 * np.pi)
    e = 1.602176487E-19      
    m0 = 9.10938188E-31  

    v2d = df.vertex_to_dof_map(V_space)
    Vtot_func = df.Function(V_space)
    Vtot_func.vector()[v2d] = Vtot_array
    
    Mass_func = df.Function(V_space)
    Mass_func.vector()[v2d] = Mass_array

    scale = df.Constant(1e9)
    coeff_val = hbar**2 / (2 * m0 * e)
    
    a = scale * df.Constant(coeff_val) * (1.0 / Mass_func) * df.inner(df.grad(u), df.grad(v)) * df.dx + scale * Vtot_func * u * v * df.dx
    m_form = scale * u * v * df.dx

    A = df.PETScMatrix()
    M_mat = df.PETScMatrix()
    df.assemble(a, tensor=A)
    df.assemble(m_form, tensor=M_mat)

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
    
    for i in range(n_eig):
        r, _, rx, _ = solver.get_eigenpair(i)
        E.append(r) 

        eig_func = df.Function(V_space)
        eig_func.vector()[:] = rx
        psi_val = eig_func.vector().get_local()[v2d]

        z_array = mesh.coordinates()[:, 0]
        norm = np.sqrt(np.trapz(np.abs(psi_val)**2, x=z_array))
        psi_array[:, i] = psi_val / norm

    if len(E) > 0:
        idx = np.argsort(E)
        E = np.array(E)[idx]
        psi_array = psi_array[:, idx]

    return E, psi_array

# =============================================================================
# FEniCS Solver: Poisson Equation
# =============================================================================
def Poisson_dolfin(V_space, rho_net_array, Epsi_array):
    u = df.TrialFunction(V_space)
    v = df.TestFunction(V_space)

    e = 1.602176487E-19
    Epsi0 = 8.854187817620E-12

    v2d = df.vertex_to_dof_map(V_space)
    rho_func = df.Function(V_space)
    rho_func.vector()[v2d] = rho_net_array
    
    Epsi_func = df.Function(V_space)
    Epsi_func.vector()[v2d] = Epsi_array

    coeff = df.Constant(e / Epsi0)

    a = Epsi_func * df.inner(df.grad(u), df.grad(v)) * df.dx
    L = -coeff * rho_func * v * df.dx

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
    tol     = 1e-10     
    n       = 5         
    ScF     = 0.05      
    T       = 300       

    # --- NON-UNIFORM MESH PARAMETERS ---
    dz_coarse = 5e-10    
    dz_fine   = 0.5e-10  
    int_width = 2e-9     
    
    E_bind_D = 0.005   
    E_bind_A = 0.025   
    fully_ionized_flag = False 

    # ==========================================================
    # USER-DEFINED MATERIAL STACK
    # ==========================================================
    material = [
        [20.0, 'AlGaAs', 0.3, 0, 'n'],
        [10.0, 'GaAs', 0, 2e18, 'n'],
        [20.0, 'AlGaAs', 0.3, 0, 'n']
    ]

    # 1. Dynamically Build Non-Uniform Z Mesh based on user material stack
    thicknesses = [0.0] + [layer[0] * 1e-9 for layer in material]
    interfaces = np.cumsum(thicknesses)
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

    # 2. Assign Material Properties to Nodes dynamically
    V0_c = np.zeros(Nz)
    V0_v = np.zeros(Nz)
    meff_e_arr = np.zeros(Nz)
    meff_h_arr = np.zeros(Nz)
    Epsi_arr = np.zeros(Nz)
    N_D_array = np.zeros(Nz)
    N_A_array = np.zeros(Nz)

    for k in range(Nz):
        z_eval = z[k] + 1e-13 
        if z_eval >= z_max: z_eval = z_max - 1e-13
        
        layer_idx = np.searchsorted(interfaces, z_eval) - 1
        layer_idx = max(0, min(layer_idx, len(material)-1))
        
        thick, mat_name, x, dop_cm3, dop_type = material[layer_idx]
        
        # Base GaAs properties at 300K
        Eg_GaAs = 1.424
        me_GaAs = 0.067
        mh_GaAs = 0.50
        epsi_GaAs = 12.90
        
        if mat_name == 'GaAs' or x == 0:
            dEc = 0.0
            dEv = 0.0
            me = me_GaAs
            mh = mh_GaAs
            epsi = epsi_GaAs
        elif mat_name == 'AlGaAs':
            dEg = 1.155 * x + 0.37 * (x**2)
            dEc = 0.65 * dEg  
            dEv = 0.35 * dEg  
            me = me_GaAs + 0.083 * x
            mh = mh_GaAs + 0.29 * x
            epsi = epsi_GaAs - 2.82 * x
        else:
            raise ValueError(f"Material {mat_name} is not recognized.")
            
        V0_c[k] = Eg_GaAs + dEc
        V0_v[k] = 0.0 - dEv
        meff_e_arr[k] = me
        meff_h_arr[k] = mh
        Epsi_arr[k] = epsi
        
        dop_m3 = dop_cm3 * 1e6
        if dop_type.lower() == 'n':
            N_D_array[k] = dop_m3
        elif dop_type.lower() == 'p':
            N_A_array[k] = dop_m3

    total_original_charge = np.trapz(N_D_array, x=z) + np.trapz(N_A_array, x=z)

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
        x = 0.5 
        Vbending = Vs * x + Vsold * (1 - x)
        
        Vtot_c = V0_c + Vbending
        Vtot_v = V0_v + Vbending 
        
        Ec, psic = Schroed1D_dolfin(V_space, Vtot_c, meff_e_arr, n)
        Ev_inv, psiv = Schroed1D_dolfin(V_space, -Vtot_v, meff_h_arr, n)
        Ev = -np.array(Ev_inv) if len(Ev_inv) > 0 else np.array([])

        if len(Ec) == 0:
            print(f"Error: No electron eigenvalues converged at Loop {nloop}. Exiting.")
            break
            
        Ef, N_n, P_m, ND_plus_z, NA_minus_z, net_integral_charge = find_Ef_general(
            Ec, Ev, psic, psiv, N_D_array, N_A_array, Vtot_c, Vtot_v, z, T, 
            meff_e_arr, meff_h_arr, E_bind_D, E_bind_A, fully_ionized_flag
        )
        
        N_3D = np.sum(np.tile(N_n, (len(z), 1)) * np.abs(psic)**2, axis=1) if len(Ec) > 0 else np.zeros_like(z)
        P_3D = np.sum(np.tile(P_m, (len(z), 1)) * np.abs(psiv)**2, axis=1) if len(Ev) > 0 else np.zeros_like(z)
        rho_net = P_3D - N_3D + ND_plus_z - NA_minus_z
        
        Vsold = np.copy(Vs)
        Vs = Poisson_dolfin(V_space, rho_net, Epsi_arr)
            
        Err = abs(1 - sumVtotVec[-1] / (np.sum(Vs) + 1e-20))
        sumVtotVec.append(np.sum(Vs))
        ErrVec.append(Err)

        if Err < tol:
            print(f"\n--- Convergence Reached at Loop {nloop} (Tolerance Met) ---")
            convergence_met = True
            break
            
        nloop += 1
        
    if not convergence_met:
        print(f"\n--- Stopped: Maximum Iterations Reached ({Nloops}) ---")

    # =============================================================================
    # NEW Plotting Section
    # =============================================================================
    if len(Ec) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='w')
        
        # 1. Electric Field (V/m) vs Position (m)
        # Vs reflects the electron electrostatic potential energy shift in eV.
        # Thus E = d(Vs)/dz will directly yield the Electric field in V/m.
        E_field = np.gradient(Vs, z) 
        axs[0, 0].plot(z, E_field, 'r-', linewidth=2)
        axs[0, 0].set_xlabel('Position (m)', fontsize=12)
        axs[0, 0].set_ylabel('Electric Field (V/m)', fontsize=12)
        axs[0, 0].set_title('Electric Field', fontsize=14)
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Band Potential Energy (Joules) vs Position (m)
        U_c_joules = Vtot_c * e
        U_v_joules = Vtot_v * e
        Ef_joules = Ef * e
        axs[0, 1].plot(z, U_c_joules, 'b-', linewidth=2, label='E_c')
        axs[0, 1].plot(z, U_v_joules, 'g-', linewidth=2, label='E_v')
        axs[0, 1].axhline(Ef_joules, color='lime', linestyle='--', linewidth=2, label='E_f')
        axs[0, 1].set_xlabel('Position (m)', fontsize=12)
        axs[0, 1].set_ylabel('Potential Energy (Joules)', fontsize=12)
        axs[0, 1].set_title('Band Potential Energy', fontsize=14)
        axs[0, 1].legend(loc='best')
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Wavefunctions (States 1 and 2) vs Position (m)
        if len(Ec) >= 1:
            axs[1, 0].plot(z, psic[:, 0], color='darkcyan', linewidth=2, label='Psi 1 (Ground State)')
        if len(Ec) >= 2:
            axs[1, 0].plot(z, psic[:, 1], color='darkmagenta', linewidth=2, label='Psi 2 (1st Excited State)')
        axs[1, 0].set_xlabel('Position (m)', fontsize=12)
        axs[1, 0].set_ylabel(r'Wavefunction $\psi$ ($m^{-1/2}$)', fontsize=12)
        axs[1, 0].set_title('Electron Wavefunctions', fontsize=14)
        axs[1, 0].legend(loc='best')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative Sheet Charge Density (e/m^2) vs Position (m)
        # rho_net represents volumetric charge carriers (m^-3). Cumulative integration by m gives e/m^2.
        sigma = integrate.cumulative_trapezoid(rho_net, z, initial=0)
        axs[1, 1].plot(z, sigma, color='purple', linewidth=2)
        axs[1, 1].set_xlabel('Position (m)', fontsize=12)
        axs[1, 1].set_ylabel(r'Cumulative Area Charge $\sigma$ ($e/m^2$)', fontsize=12)
        axs[1, 1].set_title('Cumulative Sheet Charge Density', fontsize=14)
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


