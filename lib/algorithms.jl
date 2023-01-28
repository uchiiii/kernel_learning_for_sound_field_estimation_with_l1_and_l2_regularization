module algorithms

using LinearAlgebra

include("utils.jl")
import .utils

function getQ(
    λ::Float64,
    Σ,
    s::Array{<:Complex,1},
    α::Array{<:Complex,1},
    K::Array{<:Complex,2},
)::Float64
    return real.((K * α - s)'Σ * (K * α - s) + λ * α'K * α)
end

function reg_term_2(F::Int, λ2::Float64, q::Array{Float64,2})::Float64
    ans = 0.0
    for f = 1:F-1
        ans += utils.norm2(q[f, :] - q[f+1, :])
    end
    return λ2 * ans
end


function getK(D::Int, q::Array{Float64,1}, Ks::Array{<:Array{<:Complex,2},1})
    Kcur = zeros(ComplexF64, size(Ks[1]))
    for d = 1:D
        Kcur += q[d] * Ks[d]
    end
    return Kcur
end

function get_armijo_step_size(
    λ::Float64,
    Σ,
    D::Int,
    Ks::Array{<:Array{<:Complex,2},1},
    qs::Array{Float64,1},
    s::Array{<:Complex,1},
    α::Array{<:Complex,1},
    step0::Float64,
    Jcur::AbstractFloat,
    Dr::Array{Float64,1},
    ∂J::Array{Float64,1},
    c1 = 0.5,
    T = 0.5,
)::Float64
    step = step0
    m = Dr'∂J

    while true
        K_next = getK(D, qs + step * Dr, Ks)
        α = inv(K_next + λ * Σ) * s
        Jnew = getQ(λ, Σ, s, α, K_next)

        if Jnew <= Jcur + c1 * step * m
            return step
        else
            step = step * T
        end
        # println(step);
    end

    return step / 2
end

function get_armijo_step_size_2(
    λ::Float64,
    λ2::Float64,
    Σ,
    D::Int,
    F::Int,
    Ks::Array{<:Array{<:Complex,2},1},
    qs::Array{Float64,1},
    q::Array{Float64,2},
    s::Array{<:Complex,1},
    α::Array{<:Complex,1},
    step0::Float64,
    Jcur::AbstractFloat,
    Dr::Array{Float64,1},
    ∂J::Array{Float64,1},
    c1 = 0.5,
    T = 0.5,
)::Float64
    step = step0
    m = Dr'∂J

    while true
        K_next = getK(D, qs + step * Dr, Ks)
        α = inv(K_next + λ * Σ) * s
        Jnew = getQ(λ, Σ, s, α, K_next) + reg_term_2(F, λ2, q)

        if Jnew <= Jcur + c1 * step * m
            return step
        else
            step = step * T
        end
        # println(step);
    end

    return step / 2
end

function proximal(y::Vector{Float64})::Vector{Float64}
    u = sort(y, rev = true)
    D = size(y)[1]
    ρ = 1
    sm = 0.0
    for j = 1:D
        sm += u[j]
        if u[j] + 1.0 / j * (1.0 - sm) <= 0
            sm -= u[j]
            break
        else
            ρ = j
        end
    end
    λ = 1.0 / ρ * (1 - sm)
    x = max.(y .+ λ, 0.0)
    return x
end

function proximal2(y::Vector{Float64})::Vector{Float64}
    u = sort(y, rev = true)
    D = size(y)[1]
    sm = 0.0
    k = D
    for j = 1:D
        sm += u[j]
        if (sm - 1.0) / j >= u[j]
            k = j - 1
            sm -= u[j]
            break
        end
    end
    τ = (sm - 1.0) / k
    return max.(y .- τ, 0.0)
end

function proximal_l1(x)
    n = size(x)[1]
    A = ones(n)'
    return x - A'inv(A * A') * (A * x - 1.0)
    # return x - γ * (x/γ - A'inv(A*A')*(A*(x/γ) .- 1.0));
end

function sproximal_positive(x)
    # return x - γ * max(1.0/γ * x, 0.0);
    return x - max.(x, 0.0)
end

function sproximal_mixed_norm(x::Vector{Float64}, γ::Float64, λ::Float64)::Vector{Float64}
    return x .* min(1.0, λ / utils.norm2(x))
end
function sproximal_mixed_norm(x::Array{Float64,2}, γ::Float64, λ::Float64)::Array{Float64,2}
    return x .* min(1.0, λ / utils.norm2(x))
end

function proximal_mixed_norm(
    x::Vector{Float64},
    γ::Float64,
    λ::Float64 = 1.0,
)::Vector{Float64}
    return x .* (1.0 - γ * λ / max(utils.norm2(x), γ * λ))
end
function proximal_mixed_norm(
    x::Array{Float64,2},
    γ::Float64,
    λ::Float64 = 1.0,
)::Array{Float64,2}
    return x .* (1.0 - γ * λ / max(utils.norm2(x), γ * λ))
end

function proximal_mixed_norm_2(x, γ, λ = 1.0)
    return x .* max(1.0 - λ * γ / utils.norm2(x), 0.0)
end

function mkl_l1(
    K::Array{Matrix{ComplexF64},1},
    s::Array{ComplexF64,1},
    λ::Real, # q0::Array{Float64, 1}
)::Vector{Float64}
    Σ = I
    ε = 1e-5
    D = size(K)[1]
    q = fill(1.0 / D, D)

    γMAX = 100000.2

    #while norm(α1-α2) > ε
    for iter = 1:100

        K_cur = getK(D, q, K)
        α = inv(K_cur + λ * Σ) * s
        J = getQ(λ, Σ, s, α, K_cur)

        ∂J = zeros(ComplexF64, D)
        invT = inv(K_cur + λ * Σ)
        for d = 1:D
            ∂J[d] = -s'invT * K[d] * invT * s
        end
        ∂J = λ * real.(∂J)
        # println("∂J: ", ∂J);

        # if sum(abs.(∂J)) < 0.01 * D
        #     break;
        # end
        dJmin = minimum(∂J[q.>0.0])
        dJmax = maximum(∂J[q.>0.0])
        if abs(dJmin - dJmax) < ε && sum(∂J[q.==0.0] .< dJmax) == 0
            break
        end

        μ = argmax(q)
        # println("μ : ", μ);

        Dr = zeros(Float64, D)
        for d = 1:D
            if d == μ

            else
                if abs(q[d]) < ε && ∂J[d] - ∂J[μ] > 0
                    Dr[d] = 0.0
                    q[d] = 0.0
                elseif q[d] > 0
                    # else
                    Dr[d] = -∂J[d] + ∂J[μ]
                end
            end
        end

        for d = 1:D
            if d != μ && q[d] > 0
                # if d != μ
                Dr[μ] -= Dr[d]
            end
        end

        Ĵ = 0.0
        q̂ = copy(q)
        D̂r = copy(Dr)
        α̂ = copy(α)

        γcur = 0.0
        while Ĵ + ε < J
            q = copy(q̂)
            Dr = copy(D̂r)

            K_cur = getK(D, q, K)
            α = inv(K_cur + λ * Σ) * s
            J = getQ(λ, Σ, s, α, K_cur)

            μ = argmax(q)

            ν = 1

            γmax = γMAX

            for d = 1:D
                # if d == μ
                #     continue;
                # end
                if Dr[d] < 0.0 && -q[d] / Dr[d] < γmax
                    γmax = -q[d] / Dr[d]
                    ν = d
                end
            end
            # println("ν : ", ν);
            # if γmax == γMAX
            #     break;
            # end

            q̂ = q + γmax * Dr
            μ = 0
            if ν != 1
                μ = 1
            else
                μ = 2
            end
            for d = 1:D
                if d == ν
                    continue
                end
                if q̂[d] > q̂[μ]
                    μ = d
                end
            end
            # println("μ :", μ);

            D̂r[μ] = Dr[μ] + Dr[ν]
            D̂r[ν] = 0.0
            q̂[ν] = 0.0

            K̂next = getK(D, q̂, K)
            α̂ = inv(K̂next + λ * Σ) * s
            Ĵ = getQ(λ, Σ, s, α̂, K̂next)

            γcur = γmax
        end

        # armijos
        if γcur > 0
            step = get_armijo_step_size(λ, Σ, D, K, q, s, α, γcur, J, Dr, ∂J)
        else
            step = 0
        end

        # println("step size: ", step);

        q += step * Dr
    end
    return q
end

function mkl_l2(
    K::Array{Matrix{ComplexF64},1},
    s::Array{ComplexF64,1},
    λ::Real;
    ratio = 0.5,
    Λ = 1.0,
)
    Σ = I
    ε = 1e-6
    D = length(K)
    M = length(s)
    q0 = fill(0.0, D)
    q = fill(1.0 / D, D)

    K0 = zeros(ComplexF64, M, M)
    for d = 1:D
        K0 += q[d] * K[d]
    end

    α1 = inv(K0 + λ * Σ) * s
    α2 = fill(0.0 + 0.0im, M)

    itr = 1
    while utils.norm2(α1 - α2) > ε
        itr += 1
        α2 = α1

        v = zeros(Float64, D)
        for d = 1:D
            v[d] = real(α2'K[d] * α2)
        end

        q = q0 + Λ * (v ./ utils.norm2(v))

        K_cur = zeros(ComplexF64, M, M)
        for d = 1:D
            K_cur += q[d] * K[d]
        end

        α1 = ratio * α2 + (1 - ratio) * inv(K_cur + λ * Σ) * s
    end
    return q
end

end
