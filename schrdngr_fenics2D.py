from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------------------
# 1. Mesh and function space
print("Creating mesh and function space...")
mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, 'Lagrange', 3)

# ----------------------------------------------------------------------
# 2. Boundary condition (zero on all boundaries)
def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), u0_boundary)

# ----------------------------------------------------------------------
# 3. Define trial, test functions and potential
u = TrialFunction(V)
v = TestFunction(V)
# FIX: Expression must have a degree in legacy FEniCS
Pot = Expression('0.0', degree=0)   # zero potential – free particle in a box

# ----------------------------------------------------------------------
# 4. Assemble stiffness matrix (A) and mass matrix (M)
print("Assembling stiffness matrix A (Laplacian + potential)...")
a = (inner(grad(u), grad(v)) + Pot * u * v) * dx
L = Constant(0.0) * v * dx

A = PETScMatrix()
_ = PETScVector()          # dummy RHS vector (not used in eigenvalue problem)
assemble_system(a, L, bc, A_tensor=A, b_tensor=_)

print("Assembling mass matrix M...")
m = u * v * dx
M = PETScMatrix()
# FIX: Do NOT apply boundary conditions to the mass matrix
assemble(m, tensor=M)

print("Matrices assembled. Size: {} x {}".format(A.size(0), A.size(1)))

# ----------------------------------------------------------------------
# 5. Configure and solve eigenvalue problem (A ψ = λ M ψ)
eigensolver = SLEPcEigenSolver(A, M)
eigensolver.parameters['spectrum'] = 'smallest magnitude'   # smallest |λ|
eigensolver.parameters['tolerance'] = 1.0e-15
eigensolver.parameters['verbose'] = True                    # prints SLEPc iterations

print("\nSolving for the 5 smallest eigenvalues (in magnitude)...")
eigensolver.solve(5)   # request 5 eigenpairs
print("Eigensolver finished.\n")

# ----------------------------------------------------------------------
# 6. Create output directory and files for ParaView
output_dir = "schrodinger_results"
os.makedirs(output_dir, exist_ok=True)
eigenvalues_file = os.path.join(output_dir, "eigenvalues.txt")
print(f"Saving eigenvalues to: {eigenvalues_file}")

# Open file to write eigenvalues
with open(eigenvalues_file, 'w') as f:
    f.write("# Eigenvalues (real part) for the Schrödinger equation on a unit square\n")
    f.write("# Dirichlet boundary conditions, zero potential\n")
    f.write("# Format: index, eigenvalue\n")

# ----------------------------------------------------------------------
# 7. Post‑processing: extract, save, and optionally plot each eigenfunction
u_func = Function(V)   # function to hold the eigenvector

for i in range(5):
    # Extract eigenvalue and eigenvector
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    # r = real part of eigenvalue (should be real, imaginary part c ≈ 0)
    
    # Copy eigenvector into the Function
    u_func.vector()[:] = rx
    
    # Compute L2 norm of eigenfunction (should be 1.0 if mass‑normalised)
    norm = np.sqrt(assemble(u_func * u_func * dx))
    
    # Print detailed information
    print("--------------------------------------------------")
    print("Eigenpair {}:".format(i))
    print("  Eigenvalue       = {:.8e}".format(r))
    print("  Imaginary part   = {:.2e}".format(c))
    print("  L2 norm of ψ     = {:.8f}".format(norm))
    print("  min(ψ) / max(ψ)  = {:.3e} / {:.3e}".format(u_func.vector().min(), u_func.vector().max()))
    
    # Append eigenvalue to text file
    with open(eigenvalues_file, 'a') as f:
        f.write(f"{i} {r:.12e}\n")
    
    # ------------------------------------------------------------------
    # Save eigenfunction in XDMF format (readable by ParaView)
    xdmf_filename = os.path.join(output_dir, f"eigenfunction_{i}.xdmf")
    # FIX: Provide a proper name for the function and write without extra argument
    u_func.rename(f"eigenvalue_{r:.6e}", "eigenfunction")
    with XDMFFile(MPI.comm_world, xdmf_filename) as xdmf:
        xdmf.write(u_func)
    print(f"  Eigenfunction saved to: {xdmf_filename}")
    
    # ------------------------------------------------------------------
    # (Optional) Plot with matplotlib – keep for quick visual check
    plt.figure(figsize=(6, 5))
    p = plot(u_func, mode='color', cmap='viridis')
    plt.colorbar(p, label='ψ(x,y)')
    plt.title(f'Eigenfunction for eigenvalue λ = {r:.5e}')
    plt.xlabel('x')
    plt.ylabel('y')
    png_filename = os.path.join(output_dir, f"eigenfunction_{i}_lambda_{r:.3e}.png")
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {png_filename}")
    plt.show()
    plt.close()

print("\n" + "="*50)
print("All data saved. To visualise in ParaView:")
print("  1. Open ParaView")
print("  2. File -> Open -> navigate to the folder 'schrodinger_results'")
print("  3. Select the .xdmf files (e.g., eigenfunction_0.xdmf)")
print("  4. Click Apply")
print("="*50)
