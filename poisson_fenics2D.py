from fenics import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Create mesh and function space
# ----------------------------------------------------------------------
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

# ----------------------------------------------------------------------
# 2. Boundary condition
# ----------------------------------------------------------------------
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# ----------------------------------------------------------------------
# 3. Variational problem: -Δu = f  with f = -6
# ----------------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# ----------------------------------------------------------------------
# 4. Solve the linear system
# ----------------------------------------------------------------------
u = Function(V)          # solution function
solve(a == L, u, bc)

# ----------------------------------------------------------------------
# 5. Compute and print errors
# ----------------------------------------------------------------------
# L2 error
error_L2 = errornorm(u_D, u, "L2")

# Maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u   = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

print("\n" + "=" * 40)
print("Error analysis")
print("=" * 40)
print(f"L2 error (errornorm)      : {error_L2:.4e}")
print(f"Max error at vertices     : {error_max:.4e}")
print("=" * 40 + "\n")

# ----------------------------------------------------------------------
# 6. Save solution in XDMF format (for ParaView)
# ----------------------------------------------------------------------
xdmf_file = XDMFFile("poisson_solution.xdmf")
xdmf_file.write(u)
print("Solution saved to 'poisson_solution.xdmf' – open with ParaView.\n")

# ----------------------------------------------------------------------
# 7. Plot solution and mesh (optional, requires interactive window)
# ----------------------------------------------------------------------
plot(u, title="Numerical solution")
plot(mesh, title="Mesh")
