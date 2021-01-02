# Written in Julia 1.5
using Plots;pyplot()
using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
# using JLD
# using DataFrames
# using KernelDensity
using Random
using LaTeXStrings

# %%
dir_output = "" # write the directory of the output
cd(dir_output)

# %%
## simulating the full birth-death model

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
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (n0 * f0(η) + (n0 * f4(η) + f0(η) * n4)/2 + r * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f7(η) + f0(η) * n7)/2 + (1 - r) * (n0 * f9(η) + f0(η) * n9)/2 + 1/4 * (n4 * f4(η)) + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + r^2/4 * n5 * f5(η) + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n7 * f7(η) + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (n0^2 + n0 * n4 + r * n0 * n5 + n0 * n7 + (1 - r) * n0 * n9 + 1/4 * n4^2 + r/2 * n4 * n5 + 1/2 * n4 * n7 + (1 - r)/2 * n4 * n9 + r^2/4 * n5^2 + r/2 * n5 * n7 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n7^2 + (1 - r)/2 * n7 * n9 + (1 - r)^2/4 * n9^2)
    return r
end

function R1(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (n1 * f1(η) + (n1 * f4(η) + f1(η) * n4)/2 + (1 - r) * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f8(η) + f1(η) * n8)/2 + r * (n1 * f9(η) + f1(η) * n9)/2 + 1/4 * n4 * f4(η) + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (1 - r)^2/4 * n5 * f5(η) + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n8 * f8(η) + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + r^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 +n9) * (n1^2 + n1 * n4 + (1 - r) * n1 * n5 + n1 * n8 + r * n1 * n9 + 1/4 * n4^2 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n8 + r/2 * n4 * n9 + (1 - r)^2/4 * n5^2 + (1 - r)/2 * n5 * n8 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n8^2 + r/2 * n8 * n9 + r^2/4 * n9^2)
    return r
end

function R2(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (n2 * f2(η) + (1 - r) * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f6(η) + f2(η) * n6)/2 + (n2 * f7(η) + f2(η) * n7)/2 + r * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)^2/4 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n6 * f6(η) + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/4 * n7 * f7(η) + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + r^2/4 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (n2^2 + (1 - r) * n2 * n5 + n2 * n6 + n2 * n7 + r * n2 * n9 + (1 - r)^2/4 * n5^2 + (1 - r)/2 * n5 * n6 + (1 - r)/2 * n5 * n7 + (r * (1 - r))/2 * n5 * n9 + 1/4 * n6^2 + 1/2 * n6 * n7 + r/2 * n6 * n9 + 1/4 * n7^2 + r/2 * n7 * n9 + r^2/4 * n9^2)
    return r
end

function R3(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (n3 * f3(η) + r * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f6(η) + f3(η) * n6)/2+ (n3 * f8(η) + f3(η) * n8)/2 + (1 - r) * (n3 * f9(η) + f3(η) * n9)/2 + r^2/4 * n5 * f5(η)+ r/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2+ (r * (1 - r))/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/4 * n6 * f6(η) + 1/2 * (n6 * f8(η) + f6(η) * n8)/2+ (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/4 * n8 * f8(η) + (1 - r)/2 * (n8 * f9(η)+ f8(η) * n9)/2 + (1 - r)^2/4 * n9 * f9(η))+ (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 +n9) * (n3^2 + r * n3 * n5+ n3 * n6 + n3 * n8 + (1 - r) * n3 * n9 + r^2/4 * n5^2 + r/2 * n5 * n6 + r/2 * n5 * n8+ (r * (1 - r))/2 * n5 * n9 + 1/4 * n6^2 + 1/2 * n6 * n8 + (1 - r)/2 * n6 * n9+ 1/4 * n8^2 + (1 - r)/2 * n8 * n9 + (1 - r)^2/4 * n9^2)

    return r
end

function R4(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f1(η) + f0(η) * n1)/2 + (n1 * f4(η) + f1(η) * n4)/2 + r * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f7(η) + f1(η) * n7)/2 + (1 - r) * (n1 * f9(η) + f1(η) * n9)/2 + (n0 * f4(η) + f0(η) * n4)/2 + (1 - r) * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f8(η) + f0(η) * n8)/2 + r * (n0 * f9(η) + f0(η) * n9)/2 + 1/2 * n4 * f4(η) + 1/2 * (n4 * f5(η) + f4(η) * n5) + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + 1/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n1 + n1 * n4 + r * n1 * n5 + n1 * n7 + (1 - r) * n1 * n9 + n0 * n4 + (1 - r) * n0 * n5 + n0 * n8 + r * n0 * n9 + 1/2 * n4^2 + 1/2 * n4 * n5 + 1/2 * n4 * n7 + 1/2 * n4 * n8 + 1/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + (1 - r)/2 * n5 * n7 + r/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n7 * n8 + r/2 * n7 * n9 + (1 - r)/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R5(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n1 * f2(η) + f1(η) * n2)/2 + (1 - r) * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f6(η) + f1(η) * n6)/2 + (n1 * f7(η) + f1(η) * n7)/2 + r * (n1 * f9(η) + f1(η) * n9)/2 + (n2 * f4(η) + f2(η) * n4)/2 + (1 - r) * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f8(η) + f2(η) * n8)/2 + r * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (1 - r)^2/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + (1 - r)/2 * (n5 * f7(η) + f5(η) * n7)/2 + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + r * (1 - r) * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + r/2 * (n7 * f9(η) + f7(η) * n9)/2 + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + r^2/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n1 * n2 + (1 - r) * n1 * n5 + n1 * n6 + n1 * n7 + r * n1 * n9 + n2 * n4 + (1 - r) * n2 * n5 + n2 * n8 + r * n2 * n9 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n7 + r/2 * n4 * n9 + (1 - r)^2/2 * n5^2 + (1 - r)/2 * n5 * n6 + (1 - r)/2 * n5 * n7 + (1 - r)/2 * n5 * n8 + r * (1 - r) * n5 * n9 + 1/2 * n6 * n8 + r/2 * n6 * n9 + 1/2 * n7 * n8 + r/2 * n7 * n9 + r/2 * n8 * n9 + r^2/2 * n9^2)
    return r
end

function R6(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n2 * f3(η) + f2(η) * n3)/2 + r * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f6(η) + f2(η) * n6)/2 + (n2 * f8(η) + f2(η) * n8)/2 + (1 - r) * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r) * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f6(η) + f3(η) * n6)/2 + (n3 * f7(η) + f3(η) * n7)/2 + r * (n3 * f9(η) + f3(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + 1/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + (1 - r)/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * n6 * f6(η) + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + 1/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + r/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n2 * n3 + r * n2 * n5 + n2 * n6 + n2 * n8 + (1 - r) * n2 * n9 + (1 - r) * n3 * n5 + n3 * n6 + n3 * n7 + r * n3 * n9 + (r * (1 - r))/2 * n5^2 + 1/2 * n5 * n6 + r/2 * n5 * n7 + (1 - r)/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6^2 + 1/2 * n6 * n7 + 1/2 * n6 * n8 + 1/2 * n6 * n9 + 1/2 * n7 * n8 + (1 - r)/2 * n7 * n9 + r/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R7(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f2(η) + f0(η) * n2)/2 + (1 - r) * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f6(η) + f0(η) * n6)/2 + (n0 * f7(η) + f0(η) * n7)/2 + r * (n0 * f9(η) + f0(η) * n9)/2 + (n2 * f4(η) + f2(η) * n4)/2 + r * (n2 * f5(η) + f2(η) * n5)/2 + (n2 * f7(η) + f2(η) * n7)/2 + (1 - r) * (n2 * f9(η) + f2(η) * n9)/2 + (1 - r)/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f7(η) + f4(η) * n7)/2 + r/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + r/2 * (n5 * f6(η) + f5(η) * n6)/2 + 1/2 * (n5 * f7(η) + f5(η) * n7)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * n7 * f7(η) + 1/2 * (n7 * f9(η) + f7(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n2 + (1 - r) * n0 * n5 + n0 * n6 + n0 * n7 + r * n0 * n9 + n2 * n4 + r * n2 * n5 + n2 * n7 + (1 - r) * n2 * n9 + (1 - r)/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n7 + r/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + r/2 * n5 * n6 + 1/2 * n5 * n7 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6 * n7 + (1 - r)/2 * n6 * n9 + 1/2 * n7^2 + 1/2 * n7 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R8(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n1 * f3(η) + f1(η) * n3)/2 + r * (n1 * f5(η) + f1(η) * n5)/2 + (n1 * f6(η) + f1(η) * n6)/2 + (n1 * f8(η) + f1(η) * n8)/2 + (1 - r) * (n1 * f9(η) + f1(η) * n9)/2 + (n3 * f4(η) + f3(η) * n4)/2 + (1 - r) * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f8(η) + f3(η) * n8)/2 + r * (n3 * f9(η) + f3(η) * n9)/2 + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + (r * (1 - r))/2 * n5 * f5(η) + (1 - r)/2 * (n5 * f6(η) + f5(η) * n6)/2 + 1/2 * (n5 * f8(η) + f5(η) * n8)/2 + (r^2 + (1 - r)^2)/2 * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f8(η) + f6(η) * n8)/2 + r/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * n8 * f8(η) + 1/2 * (n8 * f9(η) + f8(η) * n9)/2 + (r * (1 - r))/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n1 * n3 + r * n1 * n5 + n1 * n6 + n1 * n8 + (1 - r) * n1 * n9 + n3 * n4 + (1 - r) * n3 * n5 + n3 * n8 + r * n3 * n9 + r/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n8 + (1 - r)/2 * n4 * n9 + (r * (1 - r))/2 * n5^2 + (1 - r)/2 * n5 * n6 + 1/2 * n5 * n8 + (r^2 + (1 - r)^2)/2 * n5 * n9 + 1/2 * n6 * n8 + r/2 * n6 * n9 + 1/2 * n8^2 + 1/2 * n8 * n9 + (r * (1 - r))/2 * n9^2)
    return r
end

function R9(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    r = α * m * (2 * (n0 * f3(η) + f0(η) * n3)/2 + r * (n0 * f5(η) + f0(η) * n5)/2 + (n0 * f6(η) + f0(η) * n6)/2 + (n0 * f8(η) + f0(η) * n8)/2 + (1 - r) * (n0 * f9(η) + f0(η) * n9)/2 + (n3 * f4(η) + f3(η) * n4)/2 + r * (n3 * f5(η) + f3(η) * n5)/2 + (n3 * f7(η) + f3(η) * n7)/2 + (1 - r) * (n3 * f9(η) + f3(η) * n9)/2 + r/2 * (n4 * f5(η) + f4(η) * n5)/2 + 1/2 * (n4 * f6(η) + f4(η) * n6)/2 + 1/2 * (n4 * f8(η) + f4(η) * n8)/2 + (1 - r)/2 * (n4 * f9(η) + f4(η) * n9)/2 + r^2/2 * n5 * f5(η) + r/2 * (n5 * f6(η) + f5(η) * n6)/2 + r/2 * (n5 * f7(η) + f5(η) * n7)/2 + r/2 * (n5 * f8(η) + f5(η) * n8)/2 + r * (1 - r) * (n5 * f9(η) + f5(η) * n9)/2 + 1/2 * (n6 * f7(η) + f6(η) * n7)/2 + (1 - r)/2 * (n6 * f9(η) + f6(η) * n9)/2 + 1/2 * (n7 * f8(η) + f7(η) * n8)/2 + (1 - r)/2 * (n7 * f9(η) + f7(η) * n9)/2 + (1 - r)/2 * (n8 * f9(η) + f8(η) * n9)/2 + (1 - r)^2/2 * n9 * f9(η)) + (α * (1 - m))/(n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) * (2 * n0 * n3 + r * n0 * n5 + n0 * n6 + n0 * n8 + (1 - r) * n0 * n9 + n3 * n4 + r * n3 * n5 + n3 * n7 + (1 - r) * n3 * n9 + r/2 * n4 * n5 + 1/2 * n4 * n6 + 1/2 * n4 * n8 + (1 - r)/2 * n4 * n9 + r^2/2 * n5^2 + r/2 * n5 * n6 + r/2 * n5 * n7 + r/2 * n5 * n8 + r * (1 - r) * n5 * n9 + 1/2 * n6 * n7 + (1 - r)/2 * n6 * n9 + 1/2 * n7 * n8 + (1 - r)/2 * n7 * n9 + (1 - r)/2 * n8 * n9 + (1 - r)^2/2 * n9^2)
    return r
end

# death rates

function Q0(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[1] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n0
    return q
end

function Q1(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[2] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n1
    return q
end

function Q2(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[3] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n2
    return q
end

function Q3(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[4] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n3
    return q
end

function Q4(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[5] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n4
    return q
end

function Q5(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[6] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n5
    return q
end

function Q6(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[7] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n6
    return q
end

function Q7(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[8] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n7
    return q
end

function Q8(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[9] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n8
    return q
end

function Q9(n,p,η)
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = n .* (n .>= 0)
    α,μ,s,β,m,r,N = p
    q = (μ + s[10] + β * (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9)) * n9
    return q
end

# %%
# The tau-leaping method
# the rate function
function rate(out,u,p,t)
    out[1] = R0(u,p,η)
    out[2] = R1(u,p,η)
    out[3] = R2(u,p,η)
    out[4] = R3(u,p,η)
    out[5] = R4(u,p,η)
    out[6] = R5(u,p,η)
    out[7] = R6(u,p,η)
    out[8] = R7(u,p,η)
    out[9] = R8(u,p,η)
    out[10] = R9(u,p,η)
    out[11] = Q0(u,p,η)
    out[12] = Q1(u,p,η)
    out[13] = Q2(u,p,η)
    out[14] = Q3(u,p,η)
    out[15] = Q4(u,p,η)
    out[16] = Q5(u,p,η)
    out[17] = Q6(u,p,η)
    out[18] = Q7(u,p,η)
    out[19] = Q8(u,p,η)
    out[20] = Q9(u,p,η)
end

function c(du,u,p,t,counts,mark)
    du[1] = counts[1] - counts[11]
    du[2] = counts[2] - counts[12]
    du[3] = counts[3] - counts[13]
    du[4] = counts[4] - counts[14]
    du[5] = counts[5] - counts[15]
    du[6] = counts[6] - counts[16]
    du[7] = counts[7] - counts[17]
    du[8] = counts[8] - counts[18]
    du[9] = counts[9]  - counts[19]
    du[10] = counts[10]  - counts[20]
end


# declare the form of dc and the regular jump
dc = zeros(10,20)
regular_jumps = RegularJump(rate,c,20;mark_dist = nothing)


# %%
# functions for transforming genotype numbers n into haplotype frequencies θ
θ1(n) = (2*n[2]+n[6]+n[9]+n[5])/(2*sum(n))
θ2(n) = (2*n[3]+n[8]+n[6]+n[7])/(2*sum(n))
θ3(n) = (2*n[4]+n[10]+n[9]+n[7])/(2*sum(n))
ld(n) = θ3(n) - (θ3(n)+θ1(n))*(θ3(n)+θ2(n))
pop_size(n) = sum(n)

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

N_target = 10000 ## target population size
β = ((α - μ)/N_target) / (diffusion_rescale * rate_rescale) # calculate the density dependent mortality rate to match the target population size

m = 0.01 / (diffusion_rescale)
r = 0.25
N = (α - μ)/β
ω = α * r * (1-m/2)
λ = α * m / 2

p = [α,μ,S,β,m,r,N]
η = [1.0,0.0,0.0] # Ab,aB,ab in the foreign population

u₀ = [0.0,0.0,N,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # initial condition
tspan = (0.0,3000.0)
dt = 0.1
ensembleRepeats = 50 # total number of independent simulations

prob = DiscreteProblem(u₀,tspan,p)
jump_prob = JumpProblem(prob,Direct(),regular_jumps)

condition(u,t,integrator) = (θ2(u) <= 0.0)
affect_stop!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition,affect_stop!,save_positions=(false,true))

fig = plot(size=(200,150))

for i in 1:ensembleRepeats
    sol = solve(jump_prob,TauLeaping();dt=dt,adaptive=false,callback=cb) # directly solve once
    plot!(sol.t,θ2.(sol.u),linecolor=:steelblue,linealpha = 0.25,label="",grid = :off,framestyle=:box)
end

xlims!(fig,tspan)
ylims!(fig,(0,1))
yticks!(fig,[0,1])
xticks!(fig,:none)
xlabel!(fig,"time")

display(fig)

# %%
cd(dir_output)
prefix="" # name of output files
savefig(fig,"$prefix.pdf")
savefig(fig,"$prefix.tex")
savefig(fig,"$prefix.svg")
