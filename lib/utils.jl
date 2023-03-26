module utils

using LinearAlgebra
using SpecialFunctions
using DSP
using FFTW
using Base.Threads

function convert_x2polar(x::Vector{<:Real})
    @assert length(x) == 3
    _r = norm(x)
    θ = asin(x[3] / _r)
    if x[1] == 0.0 && x[2] == 0.0
        ϕ = 0
    else
        ϕ = atan(x[2], x[1])
    end
    return θ, ϕ
end

function convert_x2polar(x::Vector{Vector{<:Real}}, num_x::Int)
    θ = fill(0.0, num_x)
    ϕ = fill(0.0, num_x)
    for idx = 1:num_x
        _r = norm(x[idx])
        θ[idx] = asin(x[idx][3] / _r)
        if x[idx][1] == 0.0 && x[idx][2] == 0.0
            ϕ[idx] = 0
        else
            ϕ[idx] = atan(x[idx][2], x[idx][1])
        end
    end
    return θ, ϕ
end

_sphericalbesselj(ν, x) = √(π / 2x) * besselj(ν + 1 / 2, x)
sphericalbesselj(ν, x) = ifelse(x != 0.0, _sphericalbesselj(ν, x), 1.0)

create_dir(dir) = isdir(dir) || mkpath(dir)

function norm2(x::Vector{Float64})::Float64
    return sqrt(sum(x .* x))
end

function norm2(x::Vector{ComplexF64})::Float64
    return sqrt(sum(real(conj(x) .* x)))
end

function norm2(x::Vector{Vector{Float64}})::Vector{Float64}
    return map(norm2, x)
end

function norm2(x::Vector{Vector{ComplexF64}})::Vector{Float64}
    return map(norm2, x)
end

function norm2(x::Array{Float64,2})::Float64
    return sqrt(sum(x .* x))
end

function lowpass(sig::Array{Float64,2}, samplerate::Int, maxFreq::Float64 = 1500.0)
    # Evaluation signal
    h = digitalfilter(Lowpass(maxFreq, fs = samplerate), FIRWindow(hamming(64))) # 長さ64のローパスフィルタ
    sig = filtfilt(h, transpose(sig)) # hをかける
    return transpose(sig)
end

function downsample(sig::Array{Float64,2}, samplerate::Int, downsampling::Int = 6)
    pre_size_of_sigEval = size(sig)

    samplerate = samplerate ÷ downsampling
    sig = resample(sig, 1.0 / downsampling, dims = 2)
    @info "
    sigEval is resampled from $(pre_size_of_sigEval) to $(size(sig))
    samplerate : $(samplerate)
    "
    return sig, samplerate
end

function downsample(sig::Array{Float64,3}, samplerate::Int, downsampling::Int = 6)
    pre_size_of_sigEval = size(sig)

    samplerate = samplerate ÷ downsampling
    sig = resample(sig, 1.0 / downsampling, dims = 3)
    @info "
    sigEval is resampled from $(pre_size_of_sigEval) to $(size(sig))
    samplerate : $(samplerate)
    "
    return sig, samplerate
end


end
