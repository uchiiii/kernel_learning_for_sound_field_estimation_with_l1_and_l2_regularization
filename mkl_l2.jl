using Distributions;
using LinearAlgebra;
using SpecialFunctions;
using ProgressBars;
using BenchmarkTools;
using Plots;
using Random;
using StaticArrays;
using CoordinateTransformations;
# plotly()
# Plots.PlotlyBackend()

pyplot()
Plots.PyPlotBackend()

# gr()
# Plots.GRBackend()

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

const L = 2; # The number of sources
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

    # setting for evaluation points
    global evalpoint, gNUM
    evalpoint = range(-R, R, step = 0.05)
    gNUM = length(evalpoint)

    # plot mean error
    global me_eval_x, me_eval_y
    me_eval_x = range(-1.0, 1.0, step = 0.01)
    me_eval_y = range(-1.0, 1.0, step = 0.01)
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

function define_signal(freq)
    λ = 1e-2

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed

    # plain wave setting
    # xinc = zeros(Float64, L, 3);
    # xinc[1,:] = [1.0, 0.0, 0.0];
    # xinc[2,:] = [cos(π/2), sin(π/2), 0.0];

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

    # point source
    psx = zeros(Float64, L, 3)
    psx[1, :] = [2.5, 0.0, 0.0]
    psx[2, :] = [0.0, 2.5, 1.0]

    Amp = zeros(ComplexF64, L)
    Amp[1] = 20.0
    Amp[2] = 20.0

    r_0 = [0.0, 0.0, 0.0]

    # received signals (point source, only supports pressure microphones)
    s = fill(0.0 + 0.0im, M)
    for i = 1:M
        for si = 1:L # for num_source
            s[i] +=
                Amp[si] / (4π) * exp(1.0im * k * norm(r[i, :] - psx[si, :])) /
                norm(r[i, :] - psx[si, :])
        end
    end

    # get true val for point sources
    u_true = zeros(ComplexF64, gNUM, gNUM, gNUM)
    for (ii, x) in enumerate(evalpoint)
        for (jj, y) in enumerate(evalpoint)
            for (kk, z) in enumerate(evalpoint)
                cur_r = [x, y, z]
                for l = 1:L
                    u_true[ii, jj, kk] +=
                        Amp[l] / (4π) * exp(1.0im * k * norm(cur_r - psx[l, :])) /
                        norm(cur_r - psx[l, :])
                end
            end
        end
    end

    # get true val for point sources (mean error)
    u_true_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y))
    for (ii, x) in enumerate(me_eval_x)
        for (jj, y) in enumerate(me_eval_y)
            cur_r = [x, y, 0.0]
            for l = 1:L
                u_true_me[ii, jj] +=
                    Amp[l] / (4π) * exp(1.0im * k * norm(cur_r - psx[l, :])) /
                    norm(cur_r - psx[l, :])
            end
        end
    end

    # # received signals (point source with reverberation)
    # room_size = [6.0, 4.0, 3.0];
    # s = fill(0.0+0.0im, M);
    # for si = 1:L # for num_source
    #     v_psx, v_a, v_L = imagesource(room_size, 1.0 - inverse_sabine(0.5, room_size, c_speed), psx[si, :], 2);
    #     for i = 1:M
    #         for v_si = 1:v_L
    #             s[i] += Amp[si] * v_a[v_si] / (4π) * exp(1.0im * k * norm(r[i, :] - v_psx[v_si])) / norm(r[i, :] - v_psx[v_si]);
    #         end
    #     end
    # end


    # # get true val for point source with reverberation
    # u_true = zeros(ComplexF64, gNUM, gNUM, gNUM);
    # for si = 1:L # for num_source
    #     v_psx, v_a, v_L = imagesource(room_size, 1.0 - inverse_sabine(0.5, room_size, c_speed), psx[si, :], 2);
    #     for (ii,x) = enumerate(evalpoint)
    #         for (jj,y) = enumerate(evalpoint)
    #             for (kk,z) = enumerate(evalpoint) 
    #                 cur_r = [x, y, z];
    #                 for v_si = 1:v_L
    #                     u_true[ii, jj, kk] += Amp[si] * v_a[v_si] / (4π) * exp(1.0im * k * norm(cur_r - v_psx[v_si])) / norm(cur_r - v_psx[v_si]);
    #                 end
    #             end
    #         end
    #     end
    # end

    # # get true val for point source with reverberation
    # u_true_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y));
    # for si = 1:L # for num_source
    #     v_psx, v_a, v_L = imagesource(room_size, 1.0 - inverse_sabine(0.5, room_size, c_speed), psx[si, :], 2);
    #     for (ii,x) = enumerate(me_eval_x)
    #         for (jj,y) = enumerate(me_eval_y)
    #             cur_r = [x, y, 0.0];
    #             for v_si = 1:v_L
    #                 u_true_me[ii, jj] += Amp[si] * v_a[v_si] / (4π) * exp(1.0im * k * norm(cur_r - v_psx[v_si])) / norm(cur_r - v_psx[v_si]);
    #             end
    #         end
    #     end
    # end


    # add noise
    Random.seed!(2022)
    noise =
        norm(s) / sqrt(M) * sqrt(λ) * (rand.(Normal(), M) + 1.0im * rand.(Normal(), M)) /
        sqrt(2)
    s = s + noise

    return s, u_true, u_true_me
end


function mkl_l2_given_β(freq, β::Real)

    ε = 1e-6

    # noise setting
    s, u_true, u_true_me = define_signal(freq)

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed

    D = 64

    η = fill(0.0, D)
    θ = fill(0.0, D)

    # 2D 
    for d = 1:D
        dnow = d - 1
        θ[d] = π / 2
        η[d] = 2π / D * dnow - π
    end

    # 3D
    # for d = 1:D
    #     cartecian_coord = SVector{3, Float64}(base64_r[d,1], base64_r[d,2], base64_r[d,3]);
    #     spherical_coord = SphericalFromCartesian()(cartecian_coord);
    #     θ[d] = spherical_coord.ϕ + π/2.0;
    #     η[d] = spherical_coord.θ;
    # end

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
                    ηv = [sin(θ[d]) * cos(η[d]), sin(θ[d]) * sin(η[d]), cos(θ[d])]
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

    q0 = fill(0.0, D)
    q = fill(1.0 / D, D)

    K0 = zeros(ComplexF64, M, M)
    for d = 1:D
        K0 += q[d] * K[d]
    end

    α1 = inv(K0 + λ * Σ) * s
    α2 = fill(0.0 + 0.0im, M)

    while norm(α1 - α2) > ε

        α2 = α1

        v = zeros(Float64, D)
        for d = 1:D
            v[d] = real(α2'K[d] * α2)
        end

        q = q0 + Λ * (v ./ norm(v))

        K_cur = zeros(ComplexF64, M, M)
        for d = 1:D
            K_cur += q[d] * K[d]
        end

        α1 = ratio * α2 + (1 - ratio) * inv(K_cur + λ * Σ) * s
    end

    # p1 = plot!(η, q, xlabel="Θ (rad)", ylabel="weight equivalent to γ", marker=:circle, label="Proposed (L₂, η:auto)", legendfontsize=9);
    # savefig(p1, "result/wegiht_η.pdf");

    # for si = 1:L
    #     p1 = histogram(ηs[BURN_IN:end,si], bins=100, title="η's distribution", xlabel="θ[rad]", ylabel="weight");
    #     savefig(p1, "result/mcmc_η_$si.svg");
    #     p2 = histogram(βs[BURN_IN:end,si], bins=100, title="β's distribution", ylabel="weight"); 
    #     savefig(p2, "result/mcmc_β_$si.svg");
    # end


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

    # u_est_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y));

    # for (ii, x) = enumerate(me_eval_x)
    #     for (jj, y) = enumerate(me_eval_y)
    #         # @show ii, jj, kk
    #         for m = 1:M
    #             r_rm = [x, y, 0.0] - r[m, :];
    #             u_est_me[ii,jj] += α[m] * vm(k, m, r_rm, q, fill(β,D), η);
    #         end
    #     end
    # end
    # me_plot = heatmap(me_eval_x, me_eval_y, real.(transpose(10.0 * log10.(abs2.(u_true_me - u_est_me) ./ abs2.(u_true_me)))), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Normalized error (dB)", aspect_ratio=:equal, clims=(-20, 0.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # # for ll = 1:L
    # #     scatter!([psx[ll, 1]], [psx[ll, 2]], legend=false, c=:red);
    # # end
    # savefig(me_plot, "result/mean_error_l2_η.pdf");

    # # amplitude
    # amp_plot = heatmap(me_eval_x, me_eval_y, real.(u_est_me), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Amplitude (real part)", aspect_ratio=:equal, clims=(-1.0, 1.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # savefig(amp_plot, "result/amp_l2_η.pdf");

    return NMSE(evalpoint, u_true, u_est)
end

function mkl_l2_given_η(freq, η::Real)

    ε = 1e-6

    # noise setting
    s, u_true, u_true_me = define_signal(freq)

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed


    D = 64

    # η = fill(0.0, D);
    # for d = 1:D
    #     η[d] = 2π / D * d - π;
    # end
    MX = 20.0
    β = fill(0.0, D)
    for d = 1:D
        dnow = d - 1
        β[d] = MX / D * dnow
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

    q0 = fill(0.0, D)
    q = fill(1.0 / D, D)

    K0 = zeros(ComplexF64, M, M)
    for d = 1:D
        K0 += q[d] * K[d]
    end

    α1 = inv(K0 + λ * Σ) * s
    α2 = fill(0.0 + 0.0im, M)

    while norm(α1 - α2) > ε

        α2 = α1

        v = zeros(Float64, D)
        for d = 1:D
            v[d] = real(α2'K[d] * α2)
        end

        q = q0 + Λ * (v ./ norm(v))

        K_cur = zeros(ComplexF64, M, M)
        for d = 1:D
            K_cur += q[d] * K[d]
        end

        α1 = ratio * α2 + (1 - ratio) * inv(K_cur + λ * Σ) * s
    end

    # p1 = plot!(β, q, xlabel="β", ylabel="weight equivalent to γ", marker=:circle, label="Proposed (L₂, β:auto)", legendfontsize=9);
    # savefig(p1, "result/wegiht_β.pdf");

    # p = plot(β, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="β");
    # savefig(p, "result/mkl2_intermidiate_$(floor(Int,freq)).svg");

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

    # u_est_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y));

    # for (ii, x) = enumerate(me_eval_x)
    #     for (jj, y) = enumerate(me_eval_y)
    #         # @show ii, jj, kk
    #         for m = 1:M
    #             r_rm = [x, y, 0.0] - r[m, :];
    #             u_est_me[ii,jj] += α[m] * vm(k, m, r_rm, q, β, fill(η, D));
    #         end
    #     end
    # end
    # me_plot = heatmap(me_eval_x, me_eval_y, real.(transpose(10.0 * log10.(abs2.(u_true_me - u_est_me) ./ abs2.(u_true_me)))), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Normalized error (dB)", aspect_ratio=:equal, clims=(-20, 0.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # # for ll = 1:L
    # #     scatter!([psx[ll, 1]], [psx[ll, 2]], legend=false, c=:red);
    # # end
    # savefig(me_plot, "result/mean_error_l2_β.pdf");

    # # amplitude
    # amp_plot = heatmap(me_eval_x, me_eval_y, real.(u_est_me), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Amplitude (real part)", aspect_ratio=:equal, clims=(-1.0, 1.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # savefig(amp_plot, "result/amp_l2_β.pdf");

    return NMSE(evalpoint, u_true, u_est)
end

function mkl_l2(freq)

    ε = 1e-6

    # noise setting
    s, u_true, u_true_me = define_signal(freq)

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed

    Dβ = 10
    Dη = 10
    D = Dβ * Dη

    # η = fill(0.0, D);
    # for d = 1:D
    #     η[d] = 2π / D * d - π;
    # end
    MX = 10.0
    β = fill(0.0, D)
    η = fill(0.0, D)
    for d = 1:D
        dnow = d - 1
        d1 = Int(floor(dnow / Dβ))
        d2 = dnow % Dη
        # β[d] = exp(MX/Dβ * d1 - MX/2.0);
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

    q0 = fill(0.0, D)
    q = fill(1.0 / D, D)

    K0 = zeros(ComplexF64, M, M)
    for d = 1:D
        K0 += q[d] * K[d]
    end

    α1 = inv(K0 + λ * Σ) * s
    α2 = fill(0.0 + 0.0im, M)

    while norm(α1 - α2) > ε

        α2 = α1

        v = zeros(Float64, D)
        for d = 1:D
            v[d] = real(α2'K[d] * α2)
        end

        q = q0 + Λ * (v ./ norm(v))

        K_cur = zeros(ComplexF64, M, M)
        for d = 1:D
            K_cur += q[d] * K[d]
        end

        α1 = ratio * α2 + (1 - ratio) * inv(K_cur + λ * Σ) * s
    end

    p = scatter(
        β,
        η,
        marker_z = q,
        markersize = 25,
        markershape = :rect,
        legend = false,
        colorbar = true,
    )
    savefig(p, "result/mkl_l2_weight/$(floor(Int,freq)).pdf")


    # p = plot(β, q, marker=:circle, linestyle=:dash, ylabel="weight", xlabel="β");
    # savefig(p, "result/mkl2_intermidiate_$(floor(Int,freq)).svg");

    # p1 = plot(η, q, title="Θ", marker=:circle);
    # savefig(p1, "result/multiple_kernel_learning_L2/$freq.svg");

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

    u_est_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y))

    for (ii, x) in enumerate(me_eval_x)
        for (jj, y) in enumerate(me_eval_y)
            # @show ii, jj, kk
            for m = 1:M
                r_rm = [x, y, 0.0] - r[m, :]
                u_est_me[ii, jj] += α[m] * vm(k, m, r_rm, q, β, η)
            end
        end
    end
    me_plot = heatmap(
        me_eval_x,
        me_eval_y,
        real.(transpose(10.0 * log10.(abs2.(u_true_me - u_est_me) ./ abs2.(u_true_me)))),
        xlabel = "x (m)",
        ylabel = "y (m)",
        c = :blues,
        colorbar_title = "Normalized error (dB)",
        aspect_ratio = :equal,
        clims = (-20, 0.0),
    )
    tmin = 0
    tmax = 4π
    tvec = range(tmin, tmax, length = 100)
    plot!(R * sin.(tvec), R * cos.(tvec), legend = false, c = :black, grid = false)
    # for ll = 1:L
    #     scatter!([psx[ll, 1]], [psx[ll, 2]], legend=false, c=:red);
    # end
    savefig(me_plot, "result/mean_error_l2.eps")

    # amplitude
    amp_plot = heatmap(
        me_eval_x,
        me_eval_y,
        real.(u_est_me),
        xlabel = "x (m)",
        ylabel = "y (m)",
        c = :blues,
        colorbar_title = "Amplitude (real part)",
        aspect_ratio = :equal,
        clims = (-1.0, 1.0),
        grid = false,
    )
    tmin = 0
    tmax = 4π
    tvec = range(tmin, tmax, length = 100)
    plot!(R * sin.(tvec), R * cos.(tvec), legend = false, c = :black, grid = false)
    savefig(amp_plot, "result/amp_l2.eps")

    return NMSE(evalpoint, u_true, u_est)
end

function mysinc(z)
    if z == 0.0
        return 1.0
    else
        return sin(z) / z
    end
end

function test()
    println("test starts....")


    freqs = range(100, 1500, step = 100)
    # freqs = [500.0];


    β = [0.0, 0.0]
    @assert size(β)[1] == L
    η = [0.0, 0.0]
    @assert size(η)[1] == L

    # freqs = [500.0];
    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = test_given_β_η(f, β, η)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        markershape = :utriangle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "β=0.0",
    )

    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = mkl_l2_given_η(f, 0.0);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE (dB)", xlabel="Frequency (Hz)", label="Proposed (L₂, β:auto)");


    # nmse = fill(0.0, length(freqs));
    # @time for (i, f) = ProgressBar(enumerate(freqs))
    #     nmse[i] = mkl_l2_given_β(f, 5.0);
    #     # println(nmse[i]);
    # end
    # pp = plot!(freqs, nmse, marker=:circle, ylabel="NMSE (dB)", xlabel="Frequency (Hz)", label="Proposed (L₂, η:auto)", legend=:bottomright);


    nmse = fill(0.0, length(freqs))
    @time for (i, f) in ProgressBar(enumerate(freqs))
        nmse[i] = mkl_l2(f)
        # println(nmse[i]);
    end
    pp = plot!(
        freqs,
        nmse,
        marker = :circle,
        ylabel = "NMSE (dB)",
        xlabel = "Frequency (Hz)",
        label = "Proposed (L₂)",
        legend = :bottomright,
    )

    savefig(pp, "result/nmse_multiple_kernel_learning.pdf")
end

function test_given_β_η(freq::Real, β::Vector{<:Real}, η::Vector{<:Real})

    # noise setting
    s, u_true, u_true_me = define_signal(freq)

    Σ = Matrix{Complex{Float64}}(1.0I, M, M)
    λ = 1e-2

    ω = 2 * π * freq # 500 Hz
    k = ω / c_speed


    # weight
    a = fill(1 / L, L)

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
                    # Tc += 1.0 * a[l] /(C(β[l])) * translationj(ν1, μ1, ν2, μ2, k * (r[m1,:]-r[m2,:]) + 1im * β[l] * ηv);
                    xx = k * (r[m1, :] - r[m2, :]) + 1im * β[l] * ηv
                    Tc += 1.0 / (4π) * a[l] / (C(β[l])) * mysinc(sqrt(transpose(xx) * xx))
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

    # u_est_me = zeros(ComplexF64, length(me_eval_x), length(me_eval_y));

    # for (ii, x) = enumerate(me_eval_x)
    #     for (jj, y) = enumerate(me_eval_y)
    #         # @show ii, jj, kk
    #         for m = 1:M
    #             r_rm = [x, y, 0.0] - r[m, :];
    #             u_est_me[ii,jj] += α[m] * vm(k, m, r_rm, a, β, η);
    #         end
    #     end
    # end
    # me_plot = heatmap(me_eval_x, me_eval_y, real.(transpose(10.0 * log10.(abs2.(u_true_me - u_est_me) ./ abs2.(u_true_me)))), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Normalized error (dB)", aspect_ratio=:equal, clims=(-20, 0.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # # for ll = 1:L
    # #     scatter!([psx[ll, 1]], [psx[ll, 2]], legend=false, c=:red);
    # # end
    # savefig(me_plot, "result/mean_error_cmp.pdf");

    # # amplitude
    # amp_plot = heatmap(me_eval_x, me_eval_y, real.(u_est_me), xlabel="x (m)", ylabel="y (m)", c=:blues, colorbar_title="Amplitude (real part)", aspect_ratio=:equal, clims=(-1.0, 1.0));
    # tmin = 0
    # tmax = 4π
    # tvec = range(tmin, tmax, length = 100)
    # plot!(R * sin.(tvec), R * cos.(tvec), legend=false, c=:black);
    # savefig(amp_plot, "result/amp_cmp.pdf");

    return NMSE(evalpoint, u_true, u_est)
end

define_mic();
# test();

# mkl_l2_given_β(700.0, 5.0);
# mkl_l2_given_η(700.0, 0.0);
println(mkl_l2(900.0));

# test_given_β_η(900.0, [0.0, 0.0], [0.0, 0.0]);
