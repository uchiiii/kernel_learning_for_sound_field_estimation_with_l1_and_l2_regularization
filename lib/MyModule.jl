module MyModule

using SpecialFunctions, FastTransforms, LinearAlgebra, Combinatorics

export sphbesselj_mod

function sphbesselj_mod(ν::Integer, z::Real)
    J = sqrt(π / 2) * besselj(ν + 1 / 2, abs(z)) / abs(z)^(ν + 1 / 2)
    if z == 0
        J = 1.0
        for n = 1:ν
            J /= 2n + 1
        end
    end
    return J
end

function sphbesselj_mod(ν::Integer, z::Complex)
    J = sqrt(π / 2) * besselj(ν + 1 / 2, z) / (z)^(ν + 1 / 2)
    if z == 0
        J = 1.0 + im * 0.0
        for n = 1:ν
            J /= 2n + 1
        end
    end
    return J
end

export sphbesselh

function sphbesselh(ν::Integer, z::Real)
    H = sqrt(π / 2) * besselh(ν + 1 / 2, z) / z^(1 / 2)
    return H
end

export harmonic_polynomial

function harmonic_polynomial(ν::Integer, μ::Integer, z)
    σ = sign(μ)
    m = abs(μ)

    (W, Ws) = (initialize_harmonic_polynomial(ν), 0.0)
    for mm = ν:-1:m+1
        (W, Ws) = (
            -((z[1]^2 + z[2]^2) * sqrt((ν - mm) * (ν + mm + 1)) * Ws + 2mm * z[3] * W) /
            sqrt((ν - mm + 1) * (ν + mm)),
            W,
        )
    end
    W *= σ^m * (z[1] + σ * im * z[2])^m
    return W
end

function initialize_harmonic_polynomial(ν::Integer)
    a = (-1)^ν * sqrt((2ν + 1) / 4π)
    for n = 1:ν
        a *= sqrt(1 - 1 / 2n)
    end
    return a
end

export normalize_gaunt

function normalize_gaunt(
    ν1::Integer,
    μ1::Integer,
    ν2::Integer,
    μ2::Integer,
    ν3::Integer,
    μ3::Integer,
)
    A = (-1)^μ3 * sqrt((2ν1 + 1) * (2ν2 + 1) / (4π * (2ν3 + 1)))
    A *= exp(
        (
            (logfactorial(ν1 - μ1) + logfactorial(ν2 - μ2) + logfactorial(ν3 - μ3)) -
            (logfactorial(ν1 + μ1) + logfactorial(ν2 + μ2) + logfactorial(ν3 + μ3))
        ) / 2,
    )
    return A
end

export sphwavefunj

function sphwavefunj(ν::Integer, μ::Integer, z)
    w = sqrt(z[1]^2 + z[2]^2 + z[3]^2)
    return (-im)^ν * sphbesselj_mod(ν, w) * harmonic_polynomial(ν, μ, z)
end

export translationj

function translationj(ν1::Integer, μ1::Integer, ν2::Integer, μ2::Integer, z)
    ν = ν1+ν2:-2:max(abs(ν1 - ν2), abs(μ1 - μ2))
    a = (-1)^μ2 * normalize_gaunt.(ν1, -μ1, ν2, μ2, ν, μ1 - μ2) .* gaunt(-μ1, ν1, μ2, ν2)
    b = [sphwavefunj(n, μ2 - μ1, z) for n in ν]
    return dot(a, b)
end

export imagesource

function imagesource(room_size::Vector{<:Real}, room_coef, r_src, N_reflect)
    N_im = div((2N_reflect + 1) * (2N_reflect^2 + 2N_reflect + 3), 3)
    r_im = fill(zeros(3), N_im)
    a_im = zeros(N_im)

    i = 0
    # |ix| + |iy| + |iz| <= N_reflect
    for iz = -N_reflect:N_reflect
        for iy = -N_reflect+abs(iz):N_reflect-abs(iz)
            for ix = -N_reflect+abs(iy)+abs(iz):N_reflect-abs(iy)-abs(iz)
                i += 1
                r_im[i] = room_size .* [ix, iy, iz] + r_src .* ((-1) .^ [ix, iy, iz])
                (ix1, ix2) = (
                    div(abs(ix), 2) + max(rem(ix, 2), 0),
                    div(abs(ix), 2) - min(rem(ix, 2), 0),
                ) # if ix > 0, ix1++ else ix2++;
                (iy1, iy2) = (
                    div(abs(iy), 2) + max(rem(iy, 2), 0),
                    div(abs(iy), 2) - min(rem(iy, 2), 0),
                )
                (iz1, iz2) = (
                    div(abs(iz), 2) + max(rem(iz, 2), 0),
                    div(abs(iz), 2) - min(rem(iz, 2), 0),
                )
                a_im[i] = prod(room_coef .^ [ix1, ix2, iy1, iy2, iz1, iz2]) + 0 * im
            end
        end
    end
    return r_im, a_im, N_im
end

export inverse_sabine

function inverse_sabine(rt60::Float64, room_size::Vector{<:Real}, c::Float64)
    sab_coef = 24
    V = prod(room_size)

    S = 0.0
    for x in room_size, y in room_size
        S += x * y
    end
    for x in room_size
        S -= x * x
    end

    return sab_coef * log(10) * V / (c * S * rt60)
end

export max_order
function max_order(rt60::Float64, room_size::Vector{<:Real}, c::Float64)
    R = zeros(0)
    for (x, y) in combinations(room_size, 2)
        append!(R, x * y / sqrt(x^2 + y^2))
    end
    return ceil(Int, c * rt60 / minimum(R) - 1)
end

end
