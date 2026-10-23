# Julia Advanced Numerical Methods

Educational implementations of numerical algorithms for scientific and engineering computing.

## Modules
- Sparse finite-difference 1D Poisson operator
- Conjugate Gradient (CG) for symmetric positive-definite systems
- Jacobi iteration
- Power iteration for dominant eigenvalues
- Direct solves, residual checks, condition number diagnostics

## Run
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/demo.jl
julia --project=. test/runtests.jl
```

## Numerical cautions
CG assumes SPD input and is not a general-purpose solver. Power iteration may fail to converge to the desired eigenpair if the starting vector has zero projection onto the dominant eigenvector or if dominant eigenvalue magnitudes are tied. `cond(Matrix(A))` converts sparse matrices to dense and is appropriate only for small examples.

## Next steps
Add preconditioners, benchmark against Julia built-in sparse solvers, and investigate convergence for ill-conditioned systems.

MIT licensed. Educational starter code, not benchmarked or validated on production workloads.
