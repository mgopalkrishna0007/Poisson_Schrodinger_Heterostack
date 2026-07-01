import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import dolfin as df


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
        return on_boundary and df.near(x[0], 0.0)

    bc = df.DirichletBC(V_space, df.Constant(0.0), boundary)

    Vs_func = df.Function(V_space)
    df.solve(a == L, Vs_func, bc)

    return Vs_func.vector().get_local()[v2d]