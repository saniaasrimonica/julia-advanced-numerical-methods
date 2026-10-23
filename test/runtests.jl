using Test, AdvancedNumericalMethods, LinearAlgebra
@testset "Advanced numerical methods" begin
    A,h=poisson1d(20)
    @test size(A)==(20,20)
    @test h≈1/21
    b=ones(20)
    cg=conjugate_gradient(A,b)
    @test cg.converged
    @test norm(A*cg.solution-b)<1e-7
    d=solve_and_diagnose(A,b)
    @test d.relative_residual<1e-10
    M=[4.0 1.0;1.0 3.0]
    eig=power_iteration(M)
    @test eig.converged
    @test eig.eigenvalue≈maximum(eigvals(M)) atol=1e-7
    jac=jacobi_iteration(M,[1.0,2.0])
    @test jac.converged
    @test norm(M*jac.solution-[1.0,2.0])<1e-7
    @test_throws ArgumentError poisson1d(1)
end
