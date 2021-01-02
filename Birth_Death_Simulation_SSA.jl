# %%
## note: SSAstepper is an accurate simulator but it is significantly slower than the TauLeap method. 
## should only be used for one trajectory or a small population

using Plots;gr()
using DifferentialEquations # at least compatible with DifferentialEquations.jl v6.16
using DifferentialEquations.EnsembleAnalysis
# using JLD
# using DataFrames
# using KernelDensity
# using Profile

# %%
## birth rates
f0(η) = (1 - η[1] - η[2] - η[3])^2
f1(η) = η[1]^2
f2(η) = η[2]^2
f3(η) = η[3]^2
f4(η) = 2 * η[1] * (1 - η[1] - η[2] - η[3])
f5(η) = 2 * η[1] * η[2]
f6(η) = 2 * η[2] * η[3]
f7(η) = 2 * η[2] * (1 - η[1] - η[2] - η[3])
f8(η) = 2 * η[1] * η[3]
f9(η) = 2 * η[3] * (1 - η[1] - η[2] - η[3])

function R0(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (n0 * f0(η) + (n0 * f4(η) + f0(η) * n4)/2 + r * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f7(η) + f0(η) * n7)/2 + (1 - r) * (n0 * f9(η) + f0(η) * n9)/2 + 1/4 * (n4 * f4(η)) + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + r^2/4 * n5 * f5(η) + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n7 * f7(η) + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (n0^2 + n0 * n4 + r * n0 * n5 + n0 * n7 + (1 - r) * n0 * n9 + 1/4 * n4^2 + r/2 * n4 * n5 + 1/2 * n4 * n7 + (1 - r)/2 * n4 * n9 + r^2/4 * n5^2 + r/2 * n5 * n7 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n7^2 + (1 - r)/2 * n7 * n9 + (1 - r)^2/4 * n9^2)
    return r
end

function R1(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (n1 * f1(η) + (n1 * f4(η) + f1(η) * n4)/2 + (1 - r) * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f8(η) + f1(η) * n8)/2 + r * (n1 * f9(η) + f1(η) * n9)/2 + 1/4 * n4 * f4(η) + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (1 - r)^2/4 * n5 * f5(η) + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n8 * f8(η) + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + r^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 +n9) * (n1^2 + n1 * n4 + (1 - r) * n1 * n5 + n1 * n8 + r * n1 * n9 + 1/4 * n4^2 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n8 + r/2 * n4 * n9 + (1 - r)^2/4 * n5^2 + (1 - r)/2 * n5 * n8 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n8^2 + r/2 * n8 * n9 + r^2/4 * n9^2)
    return r
end

function R2(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (n2 * f2(η) + (1 - r) * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f6(η) + f2(η) * n6)/2 + (n2 * f7(η) + f2(η) * n7)/2 + r * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)^2/4 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n6 * f6(η) + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/4 * n7 * f7(η) + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + r^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (n2^2 + (1 - r) * n2 * n5 + n2 * n6 + n2 * n7 + r * n2 * n9 + (1 - r)^2/4 * n5^2 + (1 - r)/2 * n5 * n6 + (1 - r)/2 * n5 * n7 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n6^2 + 1/2 * n6 * n7 + r/2 * n6 * n9 + 1/4 * n7^2 + r/2 * n7 * n9 + r^2/4 * n9^2)
    return r
end

function R3(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (n3 * f3(η) + r * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f6(η) + f3(η) * n6)/2+ (n3 * f8(η) + f3(η) * n8)/2 + (1 - r) * (n3 * f9(η) + f3(η) * n9)/2 + r^2/4 * n5 * f5(η)+ r/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2+ (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n6 * f6(η) + 1/2 * (n6 * f8(η) + f6(η) * n8)/2+ (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/4 * n8 * f8(η) + (1 - r)/2 * (n8 * f9(η)+ f8(η) * n9)/2 + (1 - r)^2/4 * n9 * f9(η))+ (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 +n9) * (n3^2 + r * n3 * n5+ n3 * n6 + n3 * n8 + (1 - r) * n3 * n9 + r^2/4 * n5^2 + r/2 * n5 * n6 + r/2 * n5 * n8+ (r * (1 - r))/2 * n5 * n9 + 1/4 * n6^2 + 1/2 * n6 * n8 + (1 - r)/2 * n6 * n9+ 1/4 * n8^2 + (1 - r)/2 * n8 * n9 + (1 - r)^2/4 * n9^2)

    return r
end

function R4(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f1(η) + f0(η) * n1)/2 + (n1 * f4(η) + f1(η) * n4)/2 + r * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f7(η) + f1(η) * n7)/2 + (1 - r) * (n1 * f9(η) + f1(η) * n9)/2 + (n0 * f4(η) + f0(η) * n4)/2 + (1 - r) * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f8(η) + f0(η) * n8)/2 + r * (n0 * f9(η) + f0(η) * n9)/2 + 1/2 * n4 * f4(η) + 1/2 * (n4 * f5(η) + f4(η) * n5) + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + 1/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n1 + n1 * n4 + r * n1 * n5 + n1 * n7 + (1 - r) * n1 * n9 + n0 * n4 + (1 - r) * n0 * n5 + n0 * n8 + r * n0 * n9 + 1/2 * n4^2 + 1/2 * n4 * n5 + 1/2 * n4 * n7 + 1/2 * n4 * n8 + 1/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + (1 - r)/2 * n5 * n7 + r/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n7 * n8 + r/2 * n7 * n9 + (1 - r)/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R5(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n1 * f2(η) + f1(η) * n2)/2 + (1 - r) * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f6(η) + f1(η) * n6)/2 + (n1 * f7(η) + f1(η) * n7)/2 + r * (n1 * f9(η) + f1(η) * n9)/2 + (n2 * f4(η) + f2(η) * n4)/2 + (1 - r) * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f8(η) + f2(η) * n8)/2 + r * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (1 - r)^2/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + r * (1 - r) * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + r^2/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n1 * n2 + (1 - r) * n1 * n5 + n1 * n6 + n1 * n7 + r * n1 * n9 + n2 * n4 + (1 - r) * n2 * n5 + n2 * n8 + r * n2 * n9 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n7 + r/2 * n4 * n9 + (1 - r)^2/2 * n5^2 + (1 - r)/2 * n5 * n6 + (1 - r)/2 * n5 * n7 + (1 - r)/2 * n5 * n8 + r * (1 - r) * n5 * n9 + 1/2 * n6 * n8 + r/2 * n6 * n9 + 1/2 * n7 * n8 + r/2 * n7 * n9 + r/2 * n8 * n9 + r^2/2 * n9^2)
    return r
end

function R6(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n2 * f3(η) + f2(η) * n3)/2 + r * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f6(η) + f2(η) * n6)/2 + (n2 * f8(η) + f2(η) * n8)/2 + (1 - r) * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r) * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f6(η) + f3(η) * n6)/2 + (n3 * f7(η) + f3(η) * n7)/2 + r * (n3 * f9(η) + f3(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + 1/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * n6 * f6(η) + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + 1/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n2 * n3 + r * n2 * n5 + n2 * n6 + n2 * n8 + (1 - r) * n2 * n9 + (1 - r) * n3 * n5 + n3 * n6 + n3 * n7 + r * n3 * n9 + (r * (1 - r))/2 * n5^2 + 1/2 * n5 * n6 + r/2 * n5 * n7 + (1 - r)/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6^2 + 1/2 * n6 * n7 + 1/2 * n6 * n8 + 1/2 * n6 * n9 + 1/2 * n7 * n8 + (1 - r)/2 * n7 * n9 + r/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R7(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f2(η) + f0(η) * n2)/2 + (1 - r) * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f6(η) + f0(η) * n6)/2 + (n0 * f7(η) + f0(η) * n7)/2 + r * (n0 * f9(η) + f0(η) * n9)/2 + (n2 * f4(η) + f2(η) * n4)/2 + r * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f7(η) + f2(η) * n7)/2 + (1 - r) * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + r/2 * (n5 * f6(η) + f5(η) * n6)/2 + 1/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * n7 * f7(η) + 1/2 * (n7 * f9(η) + f7(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n2 + (1 - r) * n0 * n5 + n0 * n6 + n0 * n7 + r * n0 * n9 + n2 * n4 + r * n2 * n5 + n2 * n7 + (1 - r) * n2 * n9 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n7 + r/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + r/2 * n5 * n6 + 1/2 * n5 * n7 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6 * n7 + (1 - r)/2 * n6 * n9 + 1/2 * n7^2 + 1/2 * n7 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R8(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n1 * f3(η) + f1(η) * n3)/2 + r * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f6(η) + f1(η) * n6)/2 + (n1 * f8(η) + f1(η) * n8)/2 + (1 - r) * (n1 * f9(η) + f1(η) * n9)/2 + (n3 * f4(η) + f3(η) * n4)/2 + (1 - r) * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f8(η) + f3(η) * n8)/2 + r * (n3 * f9(η) + f3(η) * n9)/2 + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + 1/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * n8 * f8(η) + 1/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n1 * n3 + r * n1 * n5 + n1 * n6 + n1 * n8 + (1 - r) * n1 * n9 + n3 * n4 + (1 - r) * n3 * n5 + n3 * n8 + r * n3 * n9 + r/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n8 + (1 - r)/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + (1 - r)/2 * n5 * n6 + 1/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6 * n8 + r/2 * n6 * n9 + 1/2 * n8^2 + 1/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R9(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f3(η) + f0(η) * n3)/2 + r * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f6(η) + f0(η) * n6)/2 + (n0 * f8(η) + f0(η) * n8)/2 + (1 - r) * (n0 * f9(η) + f0(η) * n9)/2 + (n3 * f4(η) + f3(η) * n4)/2 + r * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f7(η) + f3(η) * n7)/2 + (1 - r) * (n3 * f9(η) + f3(η) * n9)/2 + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + r^2/2 * n5 * f5(η) + r/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2 + r * (1 - r) * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)/2 * (n8 * f9(η) + f8(η) * n9)/2 + (1 - r)^2/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n3 + r * n0 * n5 + n0 * n6 + n0 * n8 + (1 - r) * n0 * n9 + n3 * n4 + r * n3 * n5 + n3 * n7 + (1 - r) * n3 * n9 + r/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n8 + (1 - r)/2 * n4 * n9 + r^2/2 * n5^2 + r/2 * n5 * n6 + r/2 * n5 * n7 + r/2 * n5 * n8 + r * (1 - r) * n5 * n9 + 1/2 * n6 * n7 + (1 - r)/2 * n6 * n9 + 1/2 * n7 * n8 + (1 - r)/2 * n7 * n9 + (1 - r)/2 * n8 * n9 + (1 - r)^2/2 * n9^2)
    return r
end

# death rates

function Q0(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[1] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n0
    return q
end

function Q1(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[2] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n1
    return q
end

function Q2(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[3] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n2
    return q
end

function Q3(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[4] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n3
    return q
end

function Q4(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[5] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n4
    return q
end

function Q5(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[6] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n5
    return q
end

function Q6(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[7] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n6
    return q
end

function Q7(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[8] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n7
    return q
end

function Q8(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[9] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n8
    return q
end

function Q9(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    α,μ,s,β,m,r,N = p
    q = (μ + s[10] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n9
    return q
end

# %%

# define the rates for the jumps

rate0_b(u,p,t) = R0(u,p,η)
affect0_b!(integrator) = (integrator.u[1] = integrator.u[1] + 1)
jump0_b = ConstantRateJump(rate0_b,affect0_b!)

rate0_d(u,p,t) = Q0(u,p,η)
affect0_d!(integrator) = (integrator.u[1] = integrator.u[1] - 1)
jump0_d = ConstantRateJump(rate0_d,affect0_d!)

rate1_b(u,p,t) = R1(u,p,η)
affect1_b!(integrator) = (integrator.u[2] = integrator.u[2] + 1)
jump1_b = ConstantRateJump(rate1_b,affect1_b!)

rate1_d(u,p,t) = Q1(u,p,η)
affect1_d!(integrator) = (integrator.u[2] = integrator.u[2] - 1)
jump1_d = ConstantRateJump(rate1_d,affect1_d!)

rate2_b(u,p,t) = R2(u,p,η)
affect2_b!(integrator) = (integrator.u[3] = integrator.u[3] + 1)
jump2_b = ConstantRateJump(rate2_b,affect2_b!)

rate2_d(u,p,t) = Q2(u,p,η)
affect2_d!(integrator) = (integrator.u[3] = integrator.u[3] - 1)
jump2_d = ConstantRateJump(rate2_d,affect2_d!)

rate3_b(u,p,t) = R3(u,p,η)
affect3_b!(integrator) = (integrator.u[4] = integrator.u[4] + 1)
jump3_b = ConstantRateJump(rate3_b,affect3_b!)

rate3_d(u,p,t) = Q3(u,p,η)
affect3_d!(integrator) = (integrator.u[4] = integrator.u[4] - 1)
jump3_d = ConstantRateJump(rate3_d,affect3_d!)

rate4_b(u,p,t) = R4(u,p,η)
affect4_b!(integrator) = (integrator.u[5] = integrator.u[5] + 1)
jump4_b = ConstantRateJump(rate4_b,affect4_b!)

rate4_d(u,p,t) = Q4(u,p,η)
affect4_d!(integrator) = (integrator.u[5] = integrator.u[5] - 1)
jump4_d = ConstantRateJump(rate4_d,affect4_d!)

rate5_b(u,p,t) = R5(u,p,η)
affect5_b!(integrator) = (integrator.u[6] = integrator.u[6] + 1)
jump5_b = ConstantRateJump(rate5_b,affect5_b!)

rate5_d(u,p,t) = Q5(u,p,η)
affect5_d!(integrator) = (integrator.u[6] = integrator.u[6] - 1)
jump5_d = ConstantRateJump(rate5_d,affect5_d!)

rate6_b(u,p,t) = R6(u,p,η)
affect6_b!(integrator) = (integrator.u[7] = integrator.u[7] + 1)
jump6_b = ConstantRateJump(rate6_b,affect6_b!)

rate6_d(u,p,t) = Q6(u,p,η)
affect6_d!(integrator) = (integrator.u[7] = integrator.u[7] - 1)
jump6_d = ConstantRateJump(rate6_d,affect6_d!)

rate7_b(u,p,t) = R7(u,p,η)
affect7_b!(integrator) = (integrator.u[8] = integrator.u[8] + 1)
jump7_b = ConstantRateJump(rate7_b,affect7_b!)

rate7_d(u,p,t) = Q7(u,p,η)
affect7_d!(integrator) = (integrator.u[8] = integrator.u[8] - 1)
jump7_d = ConstantRateJump(rate7_d,affect7_d!)

rate8_b(u,p,t) = R8(u,p,η)
affect8_b!(integrator) = (integrator.u[9] = integrator.u[9] + 1)
jump8_b = ConstantRateJump(rate8_b,affect8_b!)

rate8_d(u,p,t) = Q8(u,p,η)
affect8_d!(integrator) = (integrator.u[9] = integrator.u[9] - 1)
jump8_d = ConstantRateJump(rate8_d,affect8_d!)

rate9_b(u,p,t) = R9(u,p,η)
affect9_b!(integrator) = (integrator.u[10] = integrator.u[10] + 1)
jump9_b = ConstantRateJump(rate9_b,affect9_b!)

rate9_d(u,p,t) = Q9(u,p,η)
affect9_d!(integrator) = (integrator.u[10] = integrator.u[10] - 1)
jump9_d = ConstantRateJump(rate9_d,affect9_d!)

jumps = JumpSet(jump0_b,jump0_d,
         jump1_b,jump1_d,
         jump2_b,jump2_d,
         jump3_b,jump3_d,
         jump4_b,jump4_d,
         jump5_b,jump5_d,
         jump6_b,jump6_d,
         jump7_b,jump7_d,
         jump8_b,jump8_d,
         jump9_b,jump9_d)

# %%
# functions for transforming genotype numbers n into haplotype frequencies θ
θ1(n) = (2*n[2]+n[6]+n[9]+n[5])/(2*sum(n))
θ2(n) = (2*n[3]+n[8]+n[6]+n[7])/(2*sum(n))
θ3(n) = (2*n[4]+n[10]+n[9]+n[7])/(2*sum(n))

# %%
# parameters
diffusion_rescale = 1.0 # optional
rate_rescale = 1.0 # optional
α = 1.0 / rate_rescale # reproduction rate
μ = 0.1 / rate_rescale # baseline mortality rate
s = 10.0 / rate_rescale # strength of incompatibility

# classical DMI's dominance
h1 = 1.0
h2 = 1.0
h3 = 1.0

## choose from below the type of DMI to simulate:
S = [s, 0, 0, 0, s * h1, s * h3, 0, s * h2, 0, s * h3] # classical DMI
# S = [s,0,0,s,0,0,0,0,0,s] # cis-regulation, dominant (one bad haplotype ruins the entire phenotype)
# S = [s,0,0,0,s/2,0,0,s/2,0,s/2] # cis-regulation, recessive (one good haplotype is enough to function)
# S = [s,0,0,s,s/2,0,s/2,s/2,s/2,s] # symmetric, cis-regulation
# S = [s,0,0,s,s/2,0,s/2,s/2,s/2,0] # symmetric, trans-regulation
# S = [s,0,0,s,s/2,s/1000,s/2,s/2,s/2,s/1000] # symmetric trans-reg with F1 selection

N_target = 1000 ## target population size
β = ((α - μ)/N_target) / (diffusion_rescale * rate_rescale) # calculate the density dependent mortality rate to match the target population size

m = 0.01 / (diffusion_rescale)
r = 0.25
N = (α - μ)/β
ω = α * r * (1-m/2)
λ = α * m / 2

p = [α,μ,S,β,m,r,N]
η = [1.0,0.0,0.0] # Ab,aB,ab in the foreign population


u₀ = [0.0,0.0,N,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
tspan = (0.0,10000.0)


prob = DiscreteProblem(u₀,tspan,p)
jump_prob = JumpProblem(prob,Direct(),jumps,save_positions=(true,true)) # you can modify save_positions=(false,false) to discard the intermediate positions and save only the initial and the final points

condition(u,t,integrator) = (θ2(u) <= 0.0)
affect_stop!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition,affect_stop!,save_positions=(false,true))

sol = solve(jump_prob,SSAStepper(),callback=cb)

fig = plot()
plot!(fig,sol.t,θ2.(sol.u))
plot!(fig,sol.t,θ1.(sol.u))
