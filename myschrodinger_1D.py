import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import dolfin as df


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

