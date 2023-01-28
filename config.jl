module config


include("lib/tdesign.jl")
import .tdesign
inner = 0.40 * tdesign.base25_r
outer = 0.45 * tdesign.base25_r
const r = vcat(inner, outer)
export r

const RESULT_PATH = "./result"
const RAW_DATA_PATH = RESULT_PATH * "/raw"

export RESULT_PATH, RAW_DATA_PATH

const c_speed = 340.0
const rt60 = 0.5
const room_size = [6.0, 4.0, 3.0]

const L = 1
export L

# point source
psx = zeros(Float64, L, 3)
psx[1, :] = [2.5, 0.0, 0.0]
# psx[2, :] = [0.0, 2.5, 0.0]
export psx

export c_speed, rt60, room_size

const M = 50
export M

νₘ = fill(0, M)
@assert size(νₘ)[1] == M
export νₘ

const R = 0.4 # [m] radius
export R

@assert size(r) == (M, 3) # 3 dimentional
export c
const c = fill(Dict{Tuple{Int,Int},Complex{Float64}}(), M)
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
export evalpoint, gNUM
const evalpoint = range(-R, R, step = 0.05) # 0.05 or 0.4
const gNUM = length(evalpoint)

# plot mean error
export me_eval_x, me_eval_y
const me_eval_x = range(-0.6, 0.6, step = 0.01)
const me_eval_y = range(-0.6, 0.6, step = 0.01)

const UNWEIGHTED = 0
const UNIMODAL = 1
const MKL_L1 = 2
const MKL_L2 = 3
const SPLITTING_METHOD_1 = 4
const SPLITTING_METHOD_2 = 5
const SPLITTING_METHOD_2_NEW = 6
const PROXIMAL_GRADIENT = 7

const UNOPTIMIZED = [UNIMODAL, UNWEIGHTED]
const FREQUENCY_INDEPEDENT = [MKL_L1, MKL_L2, PROXIMAL_GRADIENT]
const FREQUENCY_DEPEDENT = [SPLITTING_METHOD_1, SPLITTING_METHOD_2, SPLITTING_METHOD_2_NEW]
end
