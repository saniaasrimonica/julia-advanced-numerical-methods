module AdvancedNumericalMethods
using LinearAlgebra, SparseArrays
export poisson1d, conjugate_gradient, power_iteration, solve_and_diagnose, jacobi_iteration

function poisson1d(n::Integer)
    n>=2 || throw(ArgumentError("n must be at least 2"))
    h=1/(n+1)
    A=spdiagm(-1=>fill(-1.0,n-1),0=>fill(2.0,n),1=>fill(-1.0,n-1))/h^2
    return A, h
end
function solve_and_diagnose(A::AbstractMatrix,b::AbstractVector)
    size(A,1)==size(A,2)==length(b) || throw(DimensionMismatch("dimensions must agree"))
    x=A\b
    r=norm(A*x-b)
    return (solution=x,residual_norm=r,relative_residual=r/max(norm(b),eps(Float64)),condition_number=cond(Matrix(A)))
end
function conjugate_gradient(A::AbstractMatrix,b::AbstractVector;tol=1e-10,maxiter=10*length(b))
    size(A,1)==size(A,2)==length(b) || throw(DimensionMismatch("dimensions must agree"))
    tol>0 && maxiter>0 || throw(ArgumentError("tol and maxiter must be positive"))
    x=zeros(Float64,length(b));r=Float64.(b-A*x);p=copy(r)
    rsold=dot(r,r);history=[sqrt(rsold)]
    threshold=tol*max(norm(b),1.0)
    if history[end]<=threshold
        return (solution=x,converged=true,iterations=0,residual_history=history)
    end
    for k in 1:maxiter
        Ap=A*p
        denom=dot(p,Ap)
        denom>0 || throw(ArgumentError("CG requires symmetric positive-definite A"))
        alpha=rsold/denom
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew=dot(r,r);push!(history,sqrt(rsnew))
        if sqrt(rsnew)<=threshold
            return (solution=x,converged=true,iterations=k,residual_history=history)
        end
        p .= r .+ (rsnew/rsold).*p
        rsold=rsnew
    end
    return (solution=x,converged=false,iterations=maxiter,residual_history=history)
end
function power_iteration(A::AbstractMatrix;maxiter=2000,tol=1e-10)
    size(A,1)==size(A,2) || throw(DimensionMismatch("A must be square"))
    x=ones(Float64,size(A,1));x./=norm(x)
    lambda=dot(x,A*x)
    for k in 1:maxiter
        y=A*x;ny=norm(y)
        ny>0 || throw(ArgumentError("zero matrix or invariant zero subspace"))
        x=y/ny
        next=dot(x,A*x)
        if norm(A*x-next*x)<tol
            return (eigenvalue=next,eigenvector=x,iterations=k,converged=true)
        end
        lambda=next
    end
    return (eigenvalue=lambda,eigenvector=x,iterations=maxiter,converged=false)
end
function jacobi_iteration(A::AbstractMatrix,b::AbstractVector;maxiter=10000,tol=1e-10)
    size(A,1)==size(A,2)==length(b) || throw(DimensionMismatch("dimensions must agree"))
    d=diag(A);all(!iszero, d) || throw(ArgumentError("zero diagonal"))
    x=zeros(Float64,length(b))
    for k in 1:maxiter
        next=x+(b-A*x)./d
        if norm(A*next-b)<=tol*max(norm(b),1.0)
            return (solution=next,iterations=k,converged=true)
        end
        x=next
    end
    return (solution=x,iterations=maxiter,converged=false)
end
end
