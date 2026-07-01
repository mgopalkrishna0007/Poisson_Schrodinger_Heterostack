# the same code but with vaccum padding #

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dolfin as df
import warnings
from myschrodinger_1D import Schroed1D_dolfin
from mypoisson_1D import Poisson_dolfin
from find_fermilevel import find_Ef_general
# Suppress FEniCS and SLEPc warnings for a clean console output
warnings.filterwarnings("ignore")
df.set_log_level(df.LogLevel.WARNING)

# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":
    e     = 1.602176487E-19
    Epsi0 = 8.854187817620E-12

    # Simulation Parameters
    Nloops  = 300
    tol     = 1e-9      
    n       = 3         
    T       = 300       

    # 1. Layer Definitions (Polar Catastrophe Model)
    a = 0.39e-9          
    t_layer = a / 2.0    
    
    # --- DYNAMIC STACK GENERATION ---
    uc_STO = 150
    uc_LAO = 50
    
    # Vacuum Padding Definition
    vac_thickness = 5e-9
    vac_layers = int(round(vac_thickness / t_layer))
    
    vac_names = ["Vacuum"] * vac_layers
    vac_mat = [2] * vac_layers
    vac_charge = [0.0] * vac_layers
    
    sto_names = ["SrO","TiO2"] * uc_STO
    sto_mat = [0, 0] * uc_STO
    sto_charge = [0, 0] * uc_STO
    
    lao_names = ["LaO", "AlO2"] * uc_LAO
    lao_mat = [1, 1] * uc_LAO
    lao_charge = [1, -1] * uc_LAO
    # lao_charge = [0.5] + [c for pair in [(-1, 1)] * (uc_LAO - 1) for c in pair] + [-1, 0.5]
    
    layer_names  = vac_names + sto_names + lao_names + vac_names
    layer_mat    = vac_mat + sto_mat + lao_mat + vac_mat
    layer_charge = vac_charge + sto_charge + lao_charge + vac_charge

    # Material properties: {ID: [Ec, Ev, m_e, m_h, epsilon_init]}
    MAT_PROPS = {
        0: [2.25, 2.25 - 3.2, 0.4, 1.2, 300], 
        1: [0.15, 0.15 - 5.6, 0.4, 1.2, 24],
        2: [10.0, -10.0, 1.0, 1.0, 1.0]     # 2: Vacuum (Insulating barrier, eps=1)
    }
    points_per_layer = 20
    dz_fine = t_layer / points_per_layer  
    
    z_max = len(layer_names) * t_layer
    # Using np.linspace avoids floating point accumulation errors in np.arange
    Nz = int(round(z_max / dz_fine)) + 1
    z = np.linspace(0, z_max, Nz)
    Nz = len(z)

    V0_c = np.zeros(Nz)
    V0_v = np.zeros(Nz)
    meff_e_arr = np.zeros(Nz)
    meff_h_arr = np.zeros(Nz)
    Epsi_arr = np.zeros(Nz)
    
    N_D_array = np.zeros(Nz)
    N_A_array = np.zeros(Nz)

    sigma_0 = 1.0 / (a**2) 

    for k in range(Nz):
        z_eval = z[k] + 1e-15
        layer_idx = int(z_eval // t_layer)
        if layer_idx >= len(layer_names): layer_idx = len(layer_names) - 1
        
        mat_id = layer_mat[layer_idx]
        props = MAT_PROPS[mat_id]
        
        V0_c[k] = props[0]
        V0_v[k] = props[1]
        meff_e_arr[k] = props[2]
        meff_h_arr[k] = props[3]
        Epsi_arr[k] = props[4]

    for i, charge in enumerate(layer_charge):
        if charge != 0:
            z_center = (i + 0.5) * t_layer
            idx = np.argmin(np.abs(z - z_center))
            dz_local = z[idx+1] - z[idx] if idx < len(z)-1 else z[idx] - z[idx-1]
            
            if charge > 0:
                N_D_array[idx] += (charge * sigma_0) / dz_local
            elif charge < 0:
                N_A_array[idx] += (abs(charge) * sigma_0) / dz_local

    total_original_charge = np.trapz(N_D_array, x=z) - np.trapz(N_A_array, x=z)

    mesh = df.IntervalMesh(Nz - 1, z[0], z[-1])
    mesh.coordinates()[:, 0] = z 
    V_space = df.FunctionSpace(mesh, 'CG', 1)

    Vs = np.zeros_like(z)
    Vsold = np.copy(Vs)
    nloop = 1

    ErrVec = [1]
    sumVtotVec = [1]
    convergence_met = False

    print(f"--- Starting Solver | {Nz} Mesh Nodes ---")
    
    N_3D = np.zeros_like(z)
    P_3D = np.zeros_like(z)
    rho_net = np.zeros_like(z)

    # 3. Self-Consistent Field (SCF) Loop
    while nloop <= Nloops:
        x = 0.5
        Vbending = Vs * x + Vsold * (1 - x)
        
        Vtot_c = V0_c + Vbending
        Vtot_v = V0_v + Vbending 
        
        Ec, psic = Schroed1D_dolfin(V_space, Vtot_c, meff_e_arr, n)
        Ev_inv, psiv = Schroed1D_dolfin(V_space, -Vtot_v, meff_h_arr, n)
        Ev = -np.array(Ev_inv) if len(Ev_inv) > 0 else np.array([])

        if len(Ec) == 0:
            print("Error: Eigenvalues failed to converge.")
            break
            
        Ef, N_n, P_m, ND_plus_z, NA_minus_z, net_integral_charge = find_Ef_general(
            Ec, Ev, psic, psiv, N_D_array, N_A_array, Vtot_c, Vtot_v, z, T, 
            meff_e_arr, meff_h_arr, 0.0, 0.0, True
        )
        
        N_3D = np.sum(np.tile(N_n, (len(z), 1)) * np.abs(psic)**2, axis=1) if len(Ec) > 0 else np.zeros_like(z)
        P_3D = np.sum(np.tile(P_m, (len(z), 1)) * np.abs(psiv)**2, axis=1) if len(Ev) > 0 else np.zeros_like(z)
        rho_net = P_3D - N_3D + ND_plus_z - NA_minus_z
        
        Vsold = np.copy(Vs)
        Vs = Poisson_dolfin(V_space, rho_net, Epsi_arr)
            
        Err = abs(1 - sumVtotVec[-1] / (np.sum(Vs) + 1e-20))
        sumVtotVec.append(np.sum(Vs))
        ErrVec.append(Err)

        charge_ratio = abs(net_integral_charge / total_original_charge) if total_original_charge != 0 else 0
        Ec_str = ", ".join([f"{val:.4f}" for val in Ec[:min(3, len(Ec))]])
        Ev_str = ", ".join([f"{val:.4f}" for val in Ev[:min(3, len(Ev))]])
        
        print(f"\nLoop {nloop:02d} | Fermi Level (Ef) = {Ef:.4f} eV")
        print(f"         | E_c(1,2,3) = [{Ec_str}] eV")
        print(f"         | E_v(1,2,3) = [{Ev_str}] eV")
        print(f"         | Net Integrated Charge (Q_res) = {net_integral_charge:.3e} m^-2")
        print(f"         | Original Total Charge (Donors) = {total_original_charge:.3e} m^-2")
        print(f"         |chaerge rato (abs(net_integral_charge / total_original_charge)) = {charge_ratio}")
        print(f"         | Convergence Criterion (Err)   = {Err:.3e} (Tol: {tol:.1e})")
        
        if Err < tol:
            print(f"\n--- Convergence Reached at Loop {nloop} ---")
            convergence_met = True
            break
            
        nloop += 1

    # =============================================================================
    # 3 Separate Plotting Windows 
    # =============================================================================
    
    layer_colors = {
        "Vacuum": "#e0e0e0", 
        "TiO2": "#add8e6", 
        "SrO": "#90ee90",  
        "LaO": "#f08080",  
        "AlO2": "#fffacd"  
    }
    
    def add_backgrounds(ax):
        for i in range(len(layer_names)):
            z_start = i * t_layer * 1e9
            z_end = (i + 1) * t_layer * 1e9
            ax.axvspan(z_start, z_end, color=layer_colors[layer_names[i]], alpha=0.4)

# --- PANEL 1: Energy Bands & Wavefunctions ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(z * 1e9, Vtot_c, 'b-', linewidth=2.0, label='Conduction Band (E_c)')
    ax1.plot(z * 1e9, Vtot_v, 'r-', linewidth=2.0, label='Valence Band (E_v)')
    ax1.axhline(Ef, color='lime', linestyle='--', linewidth=2.0, label='Fermi Level (E_f)')

    wf_scale = 0.5  
    # MODIFIED: Always plot the first 3 (or available) lowest states, ignoring the Ef filter
    for i in range(min(3, len(Ec))):
        wf_prob = np.abs(psic[:, i])**2
        wf_norm = wf_prob / np.max(wf_prob) if np.max(wf_prob) > 0 else wf_prob
        # Shift the wavefunction up to its corresponding energy level
        ax1.plot(z * 1e9, Ec[i] + wf_scale * wf_norm, 
                 linestyle='-.', linewidth=1.5, label=f'|psi_c{i}|^2 (E={Ec[i]:.2f} eV)')

    add_backgrounds(ax1)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_xlabel('Position z (nm)', fontsize=12)
    ax1.set_title('Band Diagram & Wavefunctions', fontsize=14, fontweight='bold')
    
    # MODIFIED: Dynamically adjust y-limits so the wavefunctions don't get cut off
    max_wf_energy = Ec[min(3, len(Ec))-1] + wf_scale if len(Ec) > 0 else np.max(Vtot_c)
    ax1.set_ylim(np.min(Vtot_v) - 0.5, max(np.max(Vtot_c), max_wf_energy) + 0.5)
    
    ax1.legend(loc='best')
    fig1.tight_layout()

    # --- PANEL 2: Charge Density (Net + 2DEG) ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    rho_plot = rho_net * dz_fine / sigma_0 
    n_plot = -N_3D * dz_fine / sigma_0  
    
    ax2.plot(z * 1e9, rho_plot, color='purple', linewidth=1.5, label='Net Charge Density')
    ax2.fill_between(z * 1e9, 0, rho_plot, color='purple', alpha=0.3)
    
    ax2.plot(z * 1e9, n_plot, color='blue', linestyle='--', linewidth=2.0, label='Mobile Electrons (2DEG)')
    ax2.fill_between(z * 1e9, 0, n_plot, color='blue', alpha=0.5)

    add_backgrounds(ax2)
    ax2.set_ylabel('Charge Density (e/a^2)', fontsize=12)
    ax2.set_xlabel('Position z (nm)', fontsize=12)
    ax2.set_title('Charge Distribution across Slab', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    fig2.tight_layout()

    # --- PANEL 3: Electric Field ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    E_field_final = -np.gradient(Vs, z)
    ax3.plot(z * 1e9, E_field_final * 1e-2 * 1e-3, color='crimson', linewidth=2.0, label='Electric Field')
    
    add_backgrounds(ax3)
    ax3.set_ylabel('Electric Field (kV/cm)', fontsize=12)
    ax3.set_xlabel('Position z (nm)', fontsize=12)
    ax3.set_title('Internal Electric Field', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, z[-1] * 1e9)
    ax3.legend(loc='best')
    fig3.tight_layout()

    # Show all active figures simultaneously and block execution until closed
    plt.show(block=True)