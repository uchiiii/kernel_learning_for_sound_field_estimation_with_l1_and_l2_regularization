using Distributions;
using LinearAlgebra;
using SpecialFunctions;
using ProgressBars;
using BenchmarkTools;
using Plots;

include("lib/MyModule.jl");
using .MyModule:
    normalize_gaunt,
    sphbesselj_mod,
    sphwavefunj,
    translationj,
    harmonic_polynomial,
    imagesource,
    inverse_sabine;
include("config.jl");
using .config: r;

# settings
# const NUM_ITERATION = 1;

const δ = 1e-10

const c_speed = 340.0;
# const ω = 2 * π * 500; # 500 Hz
# const k = ω / c_speed;

const L = 1; # The number of sources

const NUM_ITERATION = 50;

function define_mic()
    println("setting microphones...")

    # microphones' setting
    M = 50 # The number of microphones
    global M

    νₘ = fill(0, M)
    @assert size(νₘ)[1] == M
    global νₘ

    R = 0.4 # [m] radius
    global R
    # r = Array{Float64, 2}(undef, M, 3);
    # for i = 1:M
    #     r[i,:] = [R * cos(2π * (i-1) /M), R * sin(2π * (i-1) / M), 0.0];
    # end
    @assert size(r) == (M, 3) # 3 dimentional
    global c = fill(Dict{Tuple{Int,Int},Complex{Float64}}(), M)
    for i = 1:M
        for ν = 0:νₘ[i]
            for μ = -ν:ν
                c[i][(ν, μ)] = sqrt(4π)
            end
        end
    end
    # for i = 1:M
    #     for ν = 0:νₘ[i]
    #         for μ = -ν:ν
    #             if ν == 0
    #                 c[i][(ν, μ)] = √π;
    #             elseif ν == 1
    #                 c[i][(ν, μ)] = 2π/3 *  conj(harmonic_polynomial(ν, μ, r[i,:]));
    #             else
    #                 c[i][(ν, μ)] = 0.0;
    #             end
    #         end
    #     end
    # end
end

function define_signal(freq)

end


"""
r is r - rm 
"""
function vm(
    k::Real,
    m,
    r::Vector{<:Real},
    a::Vector{<:Real},
    β::Vector{<:Real},
    η::Vector{<:Real},
)
    @assert size(a) == size(β)
    @assert size(a) == size(η)
    ans::ComplexF64 = 0.0
    for ν = 0:νₘ[m], μ = -ν:ν
        φ::ComplexF64 = 0.0
        for (l, ai) in enumerate(a)
            ηv = [cos(η[l]), sin(η[l]), 0.0]
            φ += 1.0 * ai / (C(β[l])) * sphwavefunj(ν, μ, k * r + 1.0im * β[l] * ηv)
        end
        ans += c[m][(ν, μ)] * φ
    end
    return ans
end

function NMSE(evalpoints, u_true::Array{<:Complex,3}, u_est::Array{<:Complex,3})
    bunshi::Float64 = 0.0
    bunbo::Float64 = 0.0
    for (ii, x) in enumerate(evalpoints)
        for (jj, y) in enumerate(evalpoints)
            for (kk, z) in enumerate(evalpoints)
                if x * x + y * y + z * z <= R * R # evaluation points
                    bunshi += abs2(u_true[ii, jj, kk] - u_est[ii, jj, kk])
                    bunbo += abs2(u_true[ii, jj, kk])
                end
            end
        end
    end

    return 10.0 * log10(bunshi / bunbo)
end


function C(β::AbstractFloat)
    if β == 0.0
        return 1.0
    else
        return (exp(β) - exp(-β)) / (2 * β)
    end
end

function cmvnormalpdf(x::Array{<:Complex,1}, μ::Array{<:Complex,1}, p::Array{<:Complex,2})
    @assert size(x) == size(μ)
    @assert size(x)[1] == size(p)[1]
    return exp(-log(det(p)) - (x - μ)'inv(p) * (x - μ))
end

function logCMvNormalPDF(
    x::Array{<:Complex,1},
    μ::Array{<:Complex,1},
    P::Array{<:Complex,2},
)
    @assert size(x) == size(μ)
    @assert size(x)[1] == size(P)[1]
    return real(-log(det(P)) - (x - μ)'inv(P) * (x - μ))
end

function getQ(
    λ::Float64,
    Σ::Matrix{<:Complex},
    s::Array{<:Complex,1},
    α::Array{<:Complex,1},
    K::Array{<:Complex,2},
)::Float64
    return real.((K * α - s)'Σ * (K * α - s) + λ * α'K * α)
end

function getK(D::Int, qs::Array{Float64,1}, Ks::Array{<:Array{<:Complex,2},1})
    Kcur = zeros(ComplexF64, size(Ks[1]))
    for d = 1:D
        Kcur += qs[d] * Ks[d]
    end
    return Kcur
end

function get_armijo_step_size(
    λ::Float64,
    Σ::Matrix{<:Complex},
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
)

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

# function precision(vec::Array{Float64, 1}, sm::Float64, diff:Float64)

# end

function mkl_l1_test(freq, β::Real)
    println("Multiple kernel learning algorithm for L1 starts....")

    ε = 1e-6

    # noise setting

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    # plain wave setting
    # xinc = zeros(Float64, L, 3);
    # xinc[1,:] = [1.0, 0.0, 0.0];
    # xinc[2,:] = [cos(π/2), sin(π/2), 0.0];

    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    # psx[2, :] = [0.0, 2.5, 0.0];    

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed


    r_0 = [0.0, 0.0, 0.0]


    # received signals
    # s = fill(0.0+0.0im, M);
    # for si = 1:L
    #     ss = fill(0.0+0.0im, M)
    #     for i = 1:M 
    #         for ν = 0:νₘ[i], μ = -ν:ν
    #             ss[i] += c[i][(ν, μ)]  * harmonic_polynomial(ν, μ, xinc[si,:]);
    #         end
    #         # println(ss[i]);
    #         s[i] += ss[i] * 1.0/L * exp(-1.0im * k * (xinc[si,:] ⋅ r[i,:]));
    #     end
    # end

    # received signals (point source, only supports pressure microphones)
    # s = fill(0.0+0.0im, M);
    # for i = 1:M
    #     for si = 1:L # for num_source
    #         s[i] += 1.0 / (4π) * exp(-1.0im * k * norm(r[i, :] - psx[si, :])) / norm(r[i, :] - psx[si, :]);
    #     end
    # end

    # received signals (point source with reverberation)
    room_size = [6.0, 4.0, 3.0]
    s = fill(0.0 + 0.0im, M)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for i = 1:M
            for v_si = 1:v_L
                s[i] +=
                    1.0 * v_a[v_si] / (4π) * exp(-1.0im * k * norm(r[i, :] - v_psx[v_si])) /
                    norm(r[i, :] - v_psx[v_si])
            end
        end
    end


    # add noise
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise


    D = 50

    η = fill(0.0, D)
    for d = 1:D
        η[d] = 2π / D * d - π
    end

    # make Kd 
    K = Vector{Matrix{ComplexF64}}(undef, D)
    for d = 1:D
        K[d] = zeros(ComplexF64, M, M)
    end
    for d = 1:D
        for m1 = 1:M, m2 = 1:M
            # get K_{m1, m2}
            for ν1 = 0:νₘ[m1], μ1 = -ν1:ν1
                for ν2 = 0:νₘ[m2], μ2 = -ν2:ν2
                    Tc::Complex{Float64} = 0.0
                    ηv = [cos(η[d]), sin(η[d]), 0.0]
                    Tc +=
                        1.0 / (C(β)) * translationj(
                            ν1,
                            μ1,
                            ν2,
                            μ2,
                            k * (r[m1, :] - r[m2, :]) + 1im * β * ηv,
                        )
                    K[d][m1, m2] += conj(c[m1][(ν1, μ1)]) * c[m2][(ν2, μ2)] * Tc
                end
            end
        end
    end

    Λ = 1.0
    ratio = 0.5

    q = fill(1.0 / D, D)

    γMAX = 100000.2

    #while norm(α1-α2) > ε
    for iter = 1:NUM_ITERATION

        K_cur = getK(D, q, K)
        α = inv(K_cur + λ * Σ) * s
        J = getQ(λ, Σ, s, α, K_cur)

        ∂J = zeros(ComplexF64, D)
        for d = 1:D
            for m1 = 1:M, m2 = 1:M
                ∂J[d] =
                    α'K[d]'Σ * K_cur * α + α'K_cur'Σ * K[d] * α - s'Σ * K[d] * α -
                    α'K[d]'Σ * s + λ * α'K[d] * α
            end
        end
        ∂J = real.(∂J)
        println("∂J: ", ∂J)

        # if sum(abs.(∂J)) < 0.01 * D
        #     break;
        # end

        μ = argmax(q)
        println("μ : ", μ)

        Dr = zeros(Float64, D)
        for d = 1:D
            if d == μ

            else
                if abs(q[d]) < ε && ∂J[d] - ∂J[μ] > 0
                    Dr[d] = 0.0
                    q[d] = 0.0
                elseif q[d] > 0
                    Dr[d] = -∂J[d] + ∂J[μ]
                end
            end
        end

        for d = 1:D
            if d != μ && q[d] > 0
                Dr[μ] -= Dr[d]
            end
        end

        Ĵ = 0.0
        q̂ = copy(q)
        D̂r = copy(Dr)
        α̂ = copy(α)

        println("Dr: ", Dr)
        println("sum Dr: ", sum(Dr))
        println("Jcur: ", J)
        p = plot(
            η,
            Dr,
            marker = :circle,
            linestyle = :dash,
            ylabel = "weight",
            xlabel = "θ (rad)",
        )
        savefig(p, "result/mkl1_derivative_before.svg")

        γcur = 0.0
        while Ĵ < J
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
            println("ν : ", ν)

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
            println("μ :", μ)

            D̂r[μ] = Dr[μ] + Dr[ν]
            D̂r[ν] = 0.0
            q̂[ν] = 0.0

            println("sum D̂r : ", sum(D̂r))
            println("sum q̂ : ", sum(q̂))
            p = plot(
                η,
                D̂r,
                marker = :circle,
                linestyle = :dash,
                ylabel = "weight",
                xlabel = "θ (rad)",
            )
            savefig(p, "result/mkl1_D̂r.svg")
            p = plot(
                η,
                q̂,
                marker = :circle,
                linestyle = :dash,
                ylabel = "weight",
                xlabel = "θ (rad)",
            )
            savefig(p, "result/mkl1_intermidiate_p̂.svg")


            K̂next = getK(D, q̂, K)
            α̂ = inv(K̂next + λ * Σ) * s
            Ĵ = getQ(λ, Σ, s, α̂, K̂next)
            println("Ĵ : ", Ĵ)
            println("J : ", J)

            γcur = γmax
            # println("current q: ", q);
            println("--------------------------------------------------")
        end

        # armijos
        if γcur > 0
            step = get_armijo_step_size(λ, Σ, D, K, q, s, α, γcur, J, Dr, ∂J)
        else
            step = 0
        end

        println("step size: ", step)

        q += step * Dr
        println("Dr: ", Dr)
        p = plot(
            η,
            Dr,
            marker = :circle,
            linestyle = :dash,
            ylabel = "weight",
            xlabel = "θ (rad)",
        )
        savefig(p, "result/mkl1_derivative.svg")
        p = plot(
            η,
            q,
            marker = :circle,
            linestyle = :dash,
            ylabel = "weight",
            xlabel = "θ (rad)",
        )
        savefig(p, "result/mkl1_intermidiate.svg")
        println("current iteration: ", iter)
        println("sum q: ", sum(q))
    end

    p1 = plot(
        η,
        q,
        marker = :circle,
        linestyle = :dash,
        ylabel = "weight",
        xlabel = "θ (rad)",
    )
    savefig(p1, "result/multiple_kernel_learning_L1.svg")
end

function mkl_l1_given_β(freq, β::Real)

    ε = 1e-5

    # noise setting

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    # plain wave setting
    # xinc = zeros(Float64, L, 3);
    # xinc[1,:] = [1.0, 0.0, 0.0];
    # xinc[2,:] = [cos(π/2), sin(π/2), 0.0];


    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    # psx[2, :] = [0.0, 2.5, 0.0];

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed


    r_0 = [0.0, 0.0, 0.0]


    # received signals
    # s = fill(0.0+0.0im, M);
    # for si = 1:L
    #     ss = fill(0.0+0.0im, M)
    #     for i = 1:M 
    #         for ν = 0:νₘ[i], μ = -ν:ν
    #             ss[i] += c[i][(ν, μ)]  * harmonic_polynomial(ν, μ, xinc[si,:]);
    #         end
    #         # println(ss[i]);
    #         s[i] += ss[i] * 1.0/L * exp(-1.0im * k * (xinc[si,:] ⋅ r[i,:]));
    #     end
    # end

    # received signals (point source, only supports pressure microphones)
    # s = fill(0.0+0.0im, M);
    # for i = 1:M
    #     for si = 1:L # for num_source
    #         s[i] += 1.0 / (4π) * exp(-1.0im * k * norm(r[i, :] - psx[si, :])) / norm(r[i, :] - psx[si, :]);
    #     end
    # end

    # received signals (point source with reverberation)
    room_size = [6.0, 4.0, 3.0]
    s = fill(0.0 + 0.0im, M)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for i = 1:M
            for v_si = 1:v_L
                s[i] +=
                    1.0 * v_a[v_si] / (4π) * exp(-1.0im * k * norm(r[i, :] - v_psx[v_si])) /
                    norm(r[i, :] - v_psx[v_si])
            end
        end
    end
    # add noise
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise


    D = 50

    η = fill(0.0, D)
    for d = 1:D
        η[d] = 2π / D * d - π
    end

    # make Kd 
    K = Vector{Matrix{ComplexF64}}(undef, D)
    for d = 1:D
        K[d] = zeros(ComplexF64, M, M)
    end
    for d = 1:D
        for m1 = 1:M, m2 = 1:M
            # get K_{m1, m2}
            for ν1 = 0:νₘ[m1], μ1 = -ν1:ν1
                for ν2 = 0:νₘ[m2], μ2 = -ν2:ν2
                    Tc::Complex{Float64} = 0.0
                    ηv = [cos(η[d]), sin(η[d]), 0.0]
                    Tc +=
                        1.0 / (C(β)) * translationj(
                            ν1,
                            μ1,
                            ν2,
                            μ2,
                            k * (r[m1, :] - r[m2, :]) + 1im * β * ηv,
                        )
                    K[d][m1, m2] += conj(c[m1][(ν1, μ1)]) * c[m2][(ν2, μ2)] * Tc
                end
            end
        end
    end

    Λ = 1.0
    ratio = 0.5

    q = fill(1.0 / D, D)

    γMAX = 100000.2

    #while norm(α1-α2) > ε
    for iter = 1:NUM_ITERATION

        K_cur = getK(D, q, K)
        α = inv(K_cur + λ * Σ) * s
        J = getQ(λ, Σ, s, α, K_cur)

        ∂J = zeros(ComplexF64, D)
        for d = 1:D
            for m1 = 1:M, m2 = 1:M
                ∂J[d] =
                    α'K[d]'Σ * K_cur * α + α'K_cur'Σ * K[d] * α - s'Σ * K[d] * α -
                    α'K[d]'Σ * s + λ * α'K[d] * α
            end
        end
        ∂J = real.(∂J)
        # println("∂J: ", ∂J);

        # if sum(abs.(∂J)) < 0.01 * D
        #     break;
        # end

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
                    Dr[d] = -∂J[d] + ∂J[μ]
                end
            end
        end

        for d = 1:D
            if d != μ && q[d] > 0
                Dr[μ] -= Dr[d]
            end
        end

        Ĵ = 0.0
        q̂ = copy(q)
        D̂r = copy(Dr)
        α̂ = copy(α)

        # println("Dr: ", Dr);
        # println("sum Dr: ", sum(Dr));
        # println("Jcur: ", J);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative_before.svg");

        γcur = 0.0
        while Ĵ + ε < J
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

            # println("γmax : ", γmax);
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
            # println("μ :", μ)

            D̂r[μ] = Dr[μ] + Dr[ν]
            D̂r[ν] = 0.0
            q̂[ν] = 0.0

            # println("sum D̂r : ", sum(D̂r));
            # println("sum q̂ : ", sum(q̂));
            # p = plot(η, D̂r, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_D̂r.svg");
            # p = plot(η, q̂, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_intermidiate_p̂.svg");


            K̂next = getK(D, q̂, K)
            α̂ = inv(K̂next + λ * Σ) * s
            Ĵ = getQ(λ, Σ, s, α̂, K̂next)
            # println("Ĵ : ",Ĵ);
            # println("J : ", J);

            γcur = γmax
            # println("current q: ", q);
            # println("--------------------------------------------------");
        end
        # println("Dr: ", Dr);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative.svg");
        # p = plot(η, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_intermidiate.svg");

        # armijos
        if γcur > 0
            step = get_armijo_step_size(λ, Σ, D, K, q, s, α, γcur, J, Dr, ∂J)
        else
            step = 0
        end

        # println("step size: ", step);

        q += step * Dr
        # println("current iteration: ", iter);
        # println("sum q: ",sum(q));
    end

    # setting for evaluation points
    evalpoint = range(-R, R, step = 0.1)
    gNUM = length(evalpoint)

    # get true val for plain wave
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             u_true[ii,jj,kk] = exp(-1.0im * k * (xinc ⋅ cur_r));
    #         end
    #     end
    # end

    # get true val for point sources
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             for l = 1:L
    #                 u_true[ii, jj, kk] += 1.0 / (4π) * exp(-1.0im * k * norm(cur_r - psx[l, :])) / norm(cur_r - psx[l, :]);
    #             end
    #         end
    #     end
    # end

    # get true val for point source with reverberation
    u_true = zeros(ComplexF64, gNUM, gNUM, gNUM)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for (ii, x) in enumerate(evalpoint)
            for (jj, y) in enumerate(evalpoint)
                for (kk, z) in enumerate(evalpoint)
                    cur_r = [x, y, z]
                    for v_si = 1:v_L
                        u_true[ii, jj, kk] +=
                            1.0 * v_a[v_si] / (4π) *
                            exp(-1.0im * k * norm(cur_r - v_psx[v_si])) /
                            norm(cur_r - v_psx[v_si])
                    end
                end
            end
        end
    end
    # fig_u_true = heatmap(real.(transpose(u_true)), xlabel="x", ylabel="y", title="u_true"); # xlims=(-R, R), ylims=(-R,R), title="u_true");
    # savefig(fig_u_true, "result/u_true.svg"); 

    # estimate
    u_est = zeros(ComplexF64, gNUM, gNUM, gNUM)

    # get K
    # println("start K");
    K_cur = fill(0.0 + 0.0im, M, M)
    for d = 1:D
        K_cur += q[d] * K[d]
    end

    # get α
    # println("start α");
    α = inv(K_cur + λ * Σ) * s
    # println(α);

    # esitmate u
    # println("start u_est");
    for (ii, x) in enumerate(evalpoint)
        for (jj, y) in enumerate(evalpoint)
            for (kk, z) in enumerate(evalpoint)
                # @show ii, jj, kk
                for m = 1:M
                    r_rm = [x, y, z] - r[m, :]
                    u_est[ii, jj, kk] += α[m] * vm(k, m, r_rm, q, fill(β, D), η)
                end
            end
        end
    end

    # println("start nmse");
    return NMSE(evalpoint, u_true, u_est)
end

function mkl_l1_given_η(freq, η::Real)

    ε = 1e-5

    # noise setting

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    # plain wave setting
    # xinc = zeros(Float64, L, 3);
    # xinc[1,:] = [1.0, 0.0, 0.0];
    # xinc[2,:] = [cos(π/2), sin(π/2), 0.0];

    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    # psx[2, :] = [0.0, 2.5, 0.0];

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed


    r_0 = [0.0, 0.0, 0.0]


    # received signals
    # s = fill(0.0+0.0im, M);
    # for si = 1:L
    #     ss = fill(0.0+0.0im, M)
    #     for i = 1:M 
    #         for ν = 0:νₘ[i], μ = -ν:ν
    #             ss[i] += c[i][(ν, μ)]  * harmonic_polynomial(ν, μ, xinc[si,:]);
    #         end
    #         # println(ss[i]);
    #         s[i] += ss[i] * 1.0/L * exp(-1.0im * k * (xinc[si,:] ⋅ r[i,:]));
    #     end
    # end

    # received signals (point source, only supports pressure microphones)
    # s = fill(0.0+0.0im, M);
    # for i = 1:M
    #     for si = 1:L # for num_source
    #         s[i] += 1.0 / (4π) * exp(-1.0im * k * norm(r[i, :] - psx[si, :])) / norm(r[i, :] - psx[si, :]);
    #     end
    # end

    # received signals (point source with reverberation)
    room_size = [6.0, 4.0, 3.0]
    s = fill(0.0 + 0.0im, M)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for i = 1:M
            for v_si = 1:v_L
                s[i] +=
                    1.0 * v_a[v_si] / (4π) * exp(-1.0im * k * norm(r[i, :] - v_psx[v_si])) /
                    norm(r[i, :] - v_psx[v_si])
            end
        end
    end

    # add noise
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise

    D = 50

    MX = 10.0
    β = fill(0.0, D)
    for d = 1:D
        β[d] = MX / D * d
    end

    # make Kd 
    K = Vector{Matrix{ComplexF64}}(undef, D)
    for d = 1:D
        K[d] = zeros(ComplexF64, M, M)
    end
    for d = 1:D
        for m1 = 1:M, m2 = 1:M
            # get K_{m1, m2}
            for ν1 = 0:νₘ[m1], μ1 = -ν1:ν1
                for ν2 = 0:νₘ[m2], μ2 = -ν2:ν2
                    Tc::Complex{Float64} = 0.0
                    ηv = [cos(η), sin(η), 0.0]
                    Tc +=
                        1.0 / (C(β[d])) * translationj(
                            ν1,
                            μ1,
                            ν2,
                            μ2,
                            k * (r[m1, :] - r[m2, :]) + 1im * β[d] * ηv,
                        )
                    K[d][m1, m2] += conj(c[m1][(ν1, μ1)]) * c[m2][(ν2, μ2)] * Tc
                end
            end
        end
    end

    Λ = 1.0
    ratio = 0.5

    q = fill(1.0 / D, D)

    γMAX = 100000.2

    #while norm(α1-α2) > ε
    for iter = 1:NUM_ITERATION

        K_cur = getK(D, q, K)
        α = inv(K_cur + λ * Σ) * s
        J = getQ(λ, Σ, s, α, K_cur)

        ∂J = zeros(ComplexF64, D)
        for d = 1:D
            for m1 = 1:M, m2 = 1:M
                ∂J[d] =
                    α'K[d]'Σ * K_cur * α + α'K_cur'Σ * K[d] * α - s'Σ * K[d] * α -
                    α'K[d]'Σ * s + λ * α'K[d] * α
            end
        end
        ∂J = real.(∂J)
        # println("∂J: ", ∂J);

        # if sum(abs.(∂J)) < 0.01 * D
        #     break;
        # end

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
                    Dr[d] = -∂J[d] + ∂J[μ]
                end
            end
        end

        for d = 1:D
            if d != μ && q[d] > 0
                Dr[μ] -= Dr[d]
            end
        end

        Ĵ = 0.0
        q̂ = copy(q)
        D̂r = copy(Dr)
        α̂ = copy(α)

        # println("Dr: ", Dr);
        # println("sum Dr: ", sum(Dr));
        # println("Jcur: ", J);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative_before.svg");

        γcur = 0.0
        while Ĵ + ε < J
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

            # println("γmax : ", γmax);
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

            # println("sum D̂r : ", sum(D̂r));
            # println("sum q̂ : ", sum(q̂));
            # p = plot(η, D̂r, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_D̂r.svg");
            # p = plot(η, q̂, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_intermidiate_p̂.svg");


            K̂next = getK(D, q̂, K)
            α̂ = inv(K̂next + λ * Σ) * s
            Ĵ = getQ(λ, Σ, s, α̂, K̂next)
            # println("Ĵ : ",Ĵ);
            # println("J : ", J);

            γcur = γmax
            # println("current q: ", q);
            # println("--------------------------------------------------");
        end
        # println("Dr: ", Dr);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative.svg");
        # p = plot(η, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_intermidiate.svg");

        # armijos
        if γcur > 0
            step = get_armijo_step_size(λ, Σ, D, K, q, s, α, γcur, J, Dr, ∂J)
        else
            step = 0
        end

        # println("step size: ", step);

        q += step * Dr
        # p = plot(β, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_intermidiate_$(floor(Int,freq)).svg");
        # println("current iteration: ", iter);
        # println("sum q: ",sum(q));
    end

    # setting for evaluation points
    evalpoint = range(-R, R, step = 0.1)
    gNUM = length(evalpoint)

    # get true val for plain wave
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             u_true[ii,jj,kk] = exp(-1.0im * k * (xinc ⋅ cur_r));
    #         end
    #     end
    # end

    # get true val for point sources
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             for l = 1:L
    #                 u_true[ii, jj, kk] += 1.0 / (4π) * exp(-1.0im * k * norm(cur_r - psx[l, :])) / norm(cur_r - psx[l, :]);
    #             end
    #         end
    #     end
    # end

    # get true val for point source with reverberation
    u_true = zeros(ComplexF64, gNUM, gNUM, gNUM)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for (ii, x) in enumerate(evalpoint)
            for (jj, y) in enumerate(evalpoint)
                for (kk, z) in enumerate(evalpoint)
                    cur_r = [x, y, z]
                    for v_si = 1:v_L
                        u_true[ii, jj, kk] +=
                            1.0 * v_a[v_si] / (4π) *
                            exp(-1.0im * k * norm(cur_r - v_psx[v_si])) /
                            norm(cur_r - v_psx[v_si])
                    end
                end
            end
        end
    end
    # fig_u_true = heatmap(real.(transpose(u_true)), xlabel="x", ylabel="y", title="u_true"); # xlims=(-R, R), ylims=(-R,R), title="u_true");
    # savefig(fig_u_true, "result/u_true.svg"); 

    # estimate
    u_est = zeros(ComplexF64, gNUM, gNUM, gNUM)

    # get K
    # println("start K");
    K_cur = fill(0.0 + 0.0im, M, M)
    for d = 1:D
        K_cur += q[d] * K[d]
    end

    # get α
    # println("start α");
    α = inv(K_cur + λ * Σ) * s
    # println(α);

    # esitmate u
    # println("start u_est");
    for (ii, x) in enumerate(evalpoint)
        for (jj, y) in enumerate(evalpoint)
            for (kk, z) in enumerate(evalpoint)
                # @show ii, jj, kk
                for m = 1:M
                    r_rm = [x, y, z] - r[m, :]
                    u_est[ii, jj, kk] += α[m] * vm(k, m, r_rm, q, β, fill(η, D))
                end
            end
        end
    end

    # println("start nmse");
    return NMSE(evalpoint, u_true, u_est)
end

function mkl_l1(freq)

    ε = 1e-5

    # noise setting

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    # plain wave setting
    # xinc = zeros(Float64, L, 3);
    # xinc[1,:] = [1.0, 0.0, 0.0];
    # xinc[2,:] = [cos(π/2), sin(π/2), 0.0];

    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    # psx[2, :] = [0.0, 2.5, 0.0];

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed

    # received signals
    # s = fill(0.0+0.0im, M);
    # for si = 1:L
    #     ss = fill(0.0+0.0im, M)
    #     for i = 1:M 
    #         for ν = 0:νₘ[i], μ = -ν:ν
    #             ss[i] += c[i][(ν, μ)]  * harmonic_polynomial(ν, μ, xinc[si,:]);
    #         end
    #         # println(ss[i]);
    #         s[i] += ss[i] * 1.0/L * exp(-1.0im * k * (xinc[si,:] ⋅ r[i,:]));
    #     end
    # end

    # received signals (point source, only supports pressure microphones)
    # s = fill(0.0+0.0im, M);
    # for i = 1:M
    #     for si = 1:L # for num_source
    #         s[i] += 1.0 / (4π) * exp(-1.0im * k * norm(r[i, :] - psx[si, :])) / norm(r[i, :] - psx[si, :]);
    #     end
    # end

    # received signals (point source with reverberation)
    room_size = [6.0, 4.0, 3.0]
    s = fill(0.0 + 0.0im, M)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for i = 1:M
            for v_si = 1:v_L
                s[i] +=
                    1.0 * v_a[v_si] / (4π) * exp(-1.0im * k * norm(r[i, :] - v_psx[v_si])) /
                    norm(r[i, :] - v_psx[v_si])
            end
        end
    end

    # add noise
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise

    Dβ = 10
    Dη = 10
    D = Dβ * Dη

    MX = 10.0
    β = fill(0.0, D)
    η = fill(0.0, D)
    for d = 1:D
        d1 = Int(floor(d / Dβ))
        d2 = d % Dη
        β[d] = MX / Dβ * d1
        η[d] = 2π / Dη * d2 - π
    end

    # make Kd 
    K = Vector{Matrix{ComplexF64}}(undef, D)
    for d = 1:D
        K[d] = zeros(ComplexF64, M, M)
    end
    for d = 1:D
        for m1 = 1:M, m2 = 1:M
            # get K_{m1, m2}
            for ν1 = 0:νₘ[m1], μ1 = -ν1:ν1
                for ν2 = 0:νₘ[m2], μ2 = -ν2:ν2
                    Tc::Complex{Float64} = 0.0
                    ηv = [cos(η[d]), sin(η[d]), 0.0]
                    Tc +=
                        1.0 / (C(β[d])) * translationj(
                            ν1,
                            μ1,
                            ν2,
                            μ2,
                            k * (r[m1, :] - r[m2, :]) + 1im * β[d] * ηv,
                        )
                    K[d][m1, m2] += conj(c[m1][(ν1, μ1)]) * c[m2][(ν2, μ2)] * Tc
                end
            end
        end
    end

    Λ = 1.0
    ratio = 0.5

    q = fill(1.0 / D, D)

    γMAX = 100000.2

    #while norm(α1-α2) > ε
    for iter = 1:NUM_ITERATION

        K_cur = getK(D, q, K)
        α = inv(K_cur + λ * Σ) * s
        J = getQ(λ, Σ, s, α, K_cur)

        ∂J = zeros(ComplexF64, D)
        for d = 1:D
            for m1 = 1:M, m2 = 1:M
                ∂J[d] =
                    α'K[d]'Σ * K_cur * α + α'K_cur'Σ * K[d] * α - s'Σ * K[d] * α -
                    α'K[d]'Σ * s + λ * α'K[d] * α
            end
        end
        ∂J = real.(∂J)
        # println("∂J: ", ∂J);

        if sum(abs.(∂J)) < 0.01 * D
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
                    Dr[d] = -∂J[d] + ∂J[μ]
                end
            end
        end

        for d = 1:D
            if d != μ && q[d] > 0
                Dr[μ] -= Dr[d]
            end
        end

        Ĵ = 0.0
        q̂ = copy(q)
        D̂r = copy(Dr)
        α̂ = copy(α)

        # println("Dr: ", Dr);
        # println("sum Dr: ", sum(Dr));
        # println("Jcur: ", J);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative_before.svg");

        γcur = 0.0
        while Ĵ + ε < J
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

            # println("γmax : ", γmax);
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
            println("μ :", μ)

            D̂r[μ] = Dr[μ] + Dr[ν]
            D̂r[ν] = 0.0
            q̂[ν] = 0.0

            # println("sum D̂r : ", sum(D̂r));
            # println("sum q̂ : ", sum(q̂));
            # p = plot(η, D̂r, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_D̂r.svg");
            # p = plot(η, q̂, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
            # savefig(p, "result/mkl1_intermidiate_p̂.svg");


            K̂next = getK(D, q̂, K)
            α̂ = inv(K̂next + λ * Σ) * s
            Ĵ = getQ(λ, Σ, s, α̂, K̂next)
            # println("Ĵ : ",Ĵ);
            # println("J : ", J);

            γcur = γmax
            # println("current q: ", q);
            # println("--------------------------------------------------");
        end
        # println("Dr: ", Dr);
        # p = plot(η, Dr, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_derivative.svg");
        # p = plot(η, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_intermidiate.svg");

        # armijos
        if γcur > 0
            step = get_armijo_step_size(λ, Σ, D, K, q, s, α, γcur, J, Dr, ∂J)
        else
            step = 0
        end

        # println("step size: ", step);

        q += step * Dr
        # p = plot(β, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="θ (rad)");
        # savefig(p, "result/mkl1_intermidiate_$(floor(Int,freq)).svg");
        # println("current iteration: ", iter);
        # println("sum q: ",sum(q));
    end

    # setting for evaluation points
    evalpoint = range(-R, R, step = 0.1)
    gNUM = length(evalpoint)

    # get true val for plain wave
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             u_true[ii,jj,kk] = exp(-1.0im * k * (xinc ⋅ cur_r));
    #         end
    #     end
    # end

    # get true val for point sources
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             for l = 1:L
    #                 u_true[ii, jj, kk] += 1.0 / (4π) * exp(-1.0im * k * norm(cur_r - psx[l, :])) / norm(cur_r - psx[l, :]);
    #             end
    #         end
    #     end
    # end

    # get true val for point source with reverberation
    u_true = zeros(ComplexF64, gNUM, gNUM, gNUM)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for (ii, x) in enumerate(evalpoint)
            for (jj, y) in enumerate(evalpoint)
                for (kk, z) in enumerate(evalpoint)
                    cur_r = [x, y, z]
                    for v_si = 1:v_L
                        u_true[ii, jj, kk] +=
                            1.0 * v_a[v_si] / (4π) *
                            exp(-1.0im * k * norm(cur_r - v_psx[v_si])) /
                            norm(cur_r - v_psx[v_si])
                    end
                end
            end
        end
    end
    # fig_u_true = heatmap(real.(transpose(u_true)), xlabel="x", ylabel="y", title="u_true"); # xlims=(-R, R), ylims=(-R,R), title="u_true");
    # savefig(fig_u_true, "result/u_true.svg"); 

    # estimate
    u_est = zeros(ComplexF64, gNUM, gNUM, gNUM)

    # get K
    # println("start K");
    K_cur = fill(0.0 + 0.0im, M, M)
    for d = 1:D
        K_cur += q[d] * K[d]
    end

    # get α
    # println("start α");
    α = inv(K_cur + λ * Σ) * s
    # println(α);

    # esitmate u
    # println("start u_est");
    for (ii, x) in enumerate(evalpoint)
        for (jj, y) in enumerate(evalpoint)
            for (kk, z) in enumerate(evalpoint)
                # @show ii, jj, kk
                for m = 1:M
                    r_rm = [x, y, z] - r[m, :]
                    u_est[ii, jj, kk] += α[m] * vm(k, m, r_rm, q, β, η)
                end
            end
        end
    end

    # println("start nmse");
    return NMSE(evalpoint, u_true, u_est)
end

function test()
    println("test starts....")


    freqs = range(100, 1500, step = 100)

    β = [0.0]
    @assert size(β)[1] == L
    η = [0.0]
    @assert size(η)[1] == L
    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = test_given_β_η(f, β, η)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        marker = :circle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "β=0.0, θ=0",
        linewidth = 3,
    )


    # β = [4.0];
    # @assert size(β)[1] == L
    # η = [0.0];
    # @assert size(η)[1] == L

    # # freqs = [500.0];
    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = test_given_β_η(f, β, η);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE(dB)", xlabel="Frequency(Hz)", label="β=4.0, θ=0");

    # β = [16.0];
    # @assert size(β)[1] == L
    # η = [0.0];
    # @assert size(η)[1] == L
    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = test_given_β_η(f, β, η);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE (dB)", xlabel="Frequency (Hz)", label="β=16.0, θ=0", linewidth=3, legend=:bottomright);

    # β = [16.0];
    # @assert size(β)[1] == L
    # η = [π/6];
    # @assert size(η)[1] == L
    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = test_given_β_η(f, β, η);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE (dB)", xlabel="Frequency (Hz)", label="β=16.0, θ=π/6", linewidth=3);

    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = mkl_l1_given_β(f, 5.0)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        marker = :circle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "Proposed(β=5.0, θ:auto)",
        legendfontsize = 9,
        linewidth = 3,
        legend = :bottomright,
    )


    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = mkl_l1_given_η(f, 0.0)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        marker = :circle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "Proposed(β:auto, θ=0.0)",
        legendfontsize = 9,
        linewidth = 3,
        legend = :bottomright,
    )

    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = mkl_l1(f)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        marker = :circle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "Proposed(β:auto, θ:auto)",
        legendfontsize = 9,
        linewidth = 3,
        legend = :bottomright,
    )
    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = mkl_l2(f, 10.0);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE(dB)", xlabel="Frequency(Hz)", label="β=10.0, auto(L₂)");

    savefig(pp, "result/nmse_l1_multiple_kernel_learning.pdf")
end


function test_given_β_η(f::Real, β::Vector{<:Real}, η::Vector{<:Real})

    # noise setting
    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    # plain wave setting
    # xinc = [1.0, 0.0, 0.0];

    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    # psx[2, :] = [0.0, 2.5, 0.0];

    ω = 2 * π * f # 500 Hz
    k = ω / c_speed

    r_0 = [0.0, 0.0, 0.0]

    # received signals
    # s = fill(0.0+0.0im, M);
    # for i = 1:M 
    #     for ν = 0:νₘ[i], μ = -ν:ν
    #         s[i] += c[i][(ν, μ)]  * harmonic_polynomial(ν, μ, xinc);
    #     end
    #     s[i] = s[i] * exp(-1.0im * k * (xinc ⋅ r[i,:]));
    # end

    # received signals (point source, only supports pressure microphones)
    # s = fill(0.0+0.0im, M);
    # for i = 1:M
    #     for si = 1:L # for num_source
    #         s[i] += 1.0 / (4π) * exp(-1.0im * k * norm(r[i, :] - psx[si, :])) / norm(r[i, :] - psx[si, :]);
    #     end
    # end

    # received signals (point source with reverberation)
    room_size = [6.0, 4.0, 3.0]
    s = fill(0.0 + 0.0im, M)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for i = 1:M
            for v_si = 1:v_L
                s[i] +=
                    1.0 * v_a[v_si] / (4π) * exp(-1.0im * k * norm(r[i, :] - v_psx[v_si])) /
                    norm(r[i, :] - v_psx[v_si])
            end
        end
    end

    # add noise
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise

    # weight
    a = fill(1 / L, L)

    # setting for evaluation points
    evalpoint = range(-R, R, step = 0.1)
    gNUM = length(evalpoint)

    # get true val for plain wave
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             u_true[ii,jj,kk] = exp(-1.0im * k * (xinc ⋅ cur_r));
    #         end
    #     end
    # end

    # get true val for point sources
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for (ii,x) = enumerate(evalpoint)
    #     for (jj,y) = enumerate(evalpoint)
    #         for (kk,z) = enumerate(evalpoint) 
    #             cur_r = [x, y, z];
    #             for l = 1:L
    #                 u_true[ii, jj, kk] += 1.0 / (4π) * exp(-1.0im * k * norm(cur_r - psx[l, :])) / norm(cur_r - psx[l, :]);
    #             end
    #         end
    #     end
    # end

    # get true val for point source with reverberation
    u_true = zeros(ComplexF64, gNUM, gNUM, gNUM)
    for si = 1:L # for num_source
        v_psx, v_a, v_L = imagesource(
            room_size,
            1.0 - inverse_sabine(0.5, room_size, c_speed),
            psx[si, :],
            2,
        )
        for (ii, x) in enumerate(evalpoint)
            for (jj, y) in enumerate(evalpoint)
                for (kk, z) in enumerate(evalpoint)
                    cur_r = [x, y, z]
                    for v_si = 1:v_L
                        u_true[ii, jj, kk] +=
                            1.0 * v_a[v_si] / (4π) *
                            exp(-1.0im * k * norm(cur_r - v_psx[v_si])) /
                            norm(cur_r - v_psx[v_si])
                    end
                end
            end
        end
    end

    # fig_u_true = heatmap(real.(transpose(u_true)), xlabel="x", ylabel="y", title="u_true"); # xlims=(-R, R), ylims=(-R,R), title="u_true");
    # savefig(fig_u_true, "result/u_true.svg"); 

    # estimate
    u_est = zeros(ComplexF64, gNUM, gNUM, gNUM)

    # get K
    # println("start K");
    K = fill(0.0 + 0.0im, M, M)
    for m1 = 1:M, m2 = 1:M
        # get K_{m1, m2}
        for ν1 = 0:νₘ[m1], μ1 = -ν1:ν1
            for ν2 = 0:νₘ[m2], μ2 = -ν2:ν2
                Tc::Complex{Float64} = 0.0
                for l = 1:L
                    ηv = [cos(η[l]), sin(η[l]), 0.0]
                    Tc +=
                        1.0 * a[l] / (C(β[l])) * translationj(
                            ν1,
                            μ1,
                            ν2,
                            μ2,
                            k * (r[m1, :] - r[m2, :]) + 1im * β[l] * ηv,
                        )
                end
                K[m1, m2] += conj(c[m1][(ν1, μ1)]) * c[m2][(ν2, μ2)] * Tc
            end
        end
    end

    # get α
    # println("start α");
    α = inv(K + λ * Σ) * s
    # println(α);

    # esitmate u
    # println("start u_est");
    for (ii, x) in enumerate(evalpoint)
        for (jj, y) in enumerate(evalpoint)
            for (kk, z) in enumerate(evalpoint)
                # @show ii, jj, kk
                for m = 1:M
                    r_rm = [x, y, z] - r[m, :]
                    u_est[ii, jj, kk] += α[m] * vm(k, m, r_rm, a, β, η)
                end
            end
        end
    end

    # println("start nmse");
    return NMSE(evalpoint, u_true, u_est)
end

define_mic();
test();
mkl_l1_test(500.0, 20.0)
# mkl_l1_given_η(200, 0.0);
