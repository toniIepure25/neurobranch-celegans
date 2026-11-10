"""Run the authors' Julia notebook pipeline for one Atanas recording.

This is a deliberately small, non-plotting extraction of cells 3-18 of
``code/simulations.ipynb``.  It keeps the notebook's 1-based indexing,
Savitzky-Golay preprocessing, replacement rules, ODE and Tsit5 solver, and
writes the truth/simulation traces for an independent Python comparison.

Usage:
    julia --project=/workspace/julia-depot/environments/v1.11 \
        scripts/reference/run_julia_reference.jl 2023-01-23-15
"""

using OrdinaryDiffEq
using JSON
using JLD2
using DataFrames
using CSV
using SavitzkyGolay
using DataInterpolations
using Statistics
using LinearAlgebra
using DelimitedFiles

const RECORDING = length(ARGS) >= 1 ? ARGS[1] : "2023-01-23-15"
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REF = joinpath(ROOT, "references", "external", "Celegans_premotor")
const OUT = joinpath(ROOT, "reports", "vienna_workbench", "julia_reference")
mkpath(OUT)

Relu(x) = x >= 0 ? x : 0

# The release's saved 205-neuron order.  It is the order used to fit
# A_10takes_avg/d_10takes_avg, so it must not be re-derived from a Set.
final_neurons = [strip(line) for line in readlines(joinpath(REF, "neuron_order.txt")) if !isempty(strip(line))]
N = length(final_neurons)
final_neuron_dict = Dict(name => i for (i, name) in enumerate(final_neurons))

df5 = DataFrame(CSV.File(joinpath(REF, "data", "White_data_processed", "df5.csv")))
EJ_mat = zeros(N, N)
S_mat = zeros(N, N)
for row in eachrow(df5)
    i = final_neuron_dict[row.Neuron1]
    j = final_neuron_dict[row.Neuron2]
    if row.Type == "EJ"
        EJ_mat[i, j] = row.Nbr
    elseif row.Type == "S"
        S_mat[i, j] = row.Nbr
    end
end
S_mat = S_mat'  # notebook transposes the chemical matrix for multiplication
S_mat = Matrix(S_mat)

@load joinpath(REF, "data", "model_weights", "avg_model_weights.jld2") A_10takes_avg d_10takes_avg

intrinsic_neurons = ["AVBL", "AVBR", "RIBL", "RIBR", "RID", "AVDL", "AVDR",
    "AVAL", "AVAR", "RIML", "RIMR", "AVEL", "AVER", "AIBL", "AIBR"]

# The notebook computes switchout_list from 30 recordings.  For one recording
# we apply the published twin rule to every intrinsic neuron instead; this is
# the same branch whenever the twin is present and avoids shipping 250 MB of
# unrelated recordings.

raw = JSON.parsefile(joinpath(REF, "data", "Atanas_data", "$(RECORDING).json"))
labs = raw["labeled"]
Fts1 = raw["trace_array"]
dt = round(raw["avg_timestep"]; digits=3)
dFts1dt = copy(0 * Fts1)

for i in 1:size(Fts1, 1)
    ts_tmp = savitzky_golay(Fts1[i], 11, 5).y
    ts_tmp_chop1 = ts_tmp[3:end]
    ts_tmp_chop2 = ts_tmp[1:end-2]
    dts_dt_tmp = (ts_tmp_chop1 - ts_tmp_chop2) ./ (2 * dt)
    ts_tmp_mid = ts_tmp[2:end-1]
    Fts1[i] = ts_tmp_mid
    dFts1dt[i] = dts_dt_tmp
end

Dict_nms = Dict{String,Int}()
for (key, meta) in labs
    Dict_nms[meta["label"]] = parse(Int64, key)  # Julia keys are 1-based
end
names_have = keys(Dict_nms)
ts_len = size(Fts1[1], 1)

ts_myneuros = zeros(N, ts_len)
dtsdt_myneuros = zeros(N, ts_len)
for (i, nm) in enumerate(final_neurons)
    if nm in names_have
        ts_myneuros[i, :] = Fts1[Dict_nms[nm]]
        dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[nm]]
    elseif (length(nm) == 4 || length(nm) == 5) && nm in intrinsic_neurons && nm[end] in ('L', 'R')
        # The notebook only reaches Data_Replacement for neurons in its
        # switchout list.  That list is built from all 30 recordings, so for a
        # single recording we apply the same twin rule to every intrinsic
        # neuron; this is exactly what the published function does whenever the
        # twin is present.
        twin = nm[1:end-1] * (nm[end] == 'L' ? "R" : "L")
        if twin in names_have
            ts_myneuros[i, :] = Fts1[Dict_nms[twin]]
            dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[twin]]
        elseif nm == "AVBL"
            donor = "RIBL" in names_have ? "RIBL" : ("RIBR" in names_have ? "RIBR" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        elseif nm == "AVBR"
            donor = "RIBR" in names_have ? "RIBR" : ("RIBL" in names_have ? "RIBL" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        elseif nm == "RIBL"
            donor = "AVBL" in names_have ? "AVBL" : ("AVBR" in names_have ? "AVBR" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        elseif nm == "RIBR"
            donor = "AVBR" in names_have ? "AVBR" : ("AVBL" in names_have ? "AVBL" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        elseif nm == "AIBL"
            donor = "RIML" in names_have ? "RIML" : ("RIMR" in names_have ? "RIMR" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        elseif nm == "AIBR"
            donor = "RIMR" in names_have ? "RIMR" : ("RIML" in names_have ? "RIML" : nothing)
            donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
        end
    elseif nm == "RID"
        donor = "AVBR" in names_have ? "AVBR" : ("AVBL" in names_have ? "AVBL" : nothing)
        donor !== nothing && (ts_myneuros[i, :] = Fts1[Dict_nms[donor]]; dtsdt_myneuros[i, :] = dFts1dt[Dict_nms[donor]])
    end
end

D = copy(ts_myneuros)
row_sum = sum(D, dims=2)
idxfull_list = Int64[i for i in 1:N if row_sum[i] != 0]
Dshort = D[idxfull_list, :]
EJ_mat_short = EJ_mat[idxfull_list, idxfull_list]
S_mat_short = S_mat[idxfull_list, idxfull_list]
N_short = size(idxfull_list, 1)
core_neurons = final_neurons[idxfull_list]
Ncore = size(core_neurons, 1)
ids_core = [final_neuron_dict[name] for name in core_neurons]
Dcore = D[ids_core, :]
core_neurons_dict = Dict(name => i for (i, name) in enumerate(core_neurons))

S_approx = A_10takes_avg
d_approx = d_10takes_avg
fp1, fp2, fp3 = -0.8, 0.1, 1.0
a, γ, β, τ = -2, 1.4, 10, 0.2
d = d_approx[ids_core]
A = S_approx[ids_core, ids_core]
W = EJ_mat[ids_core, ids_core]

ids_intrinsic = [core_neurons_dict[name] for name in intrinsic_neurons]
Nintrinsic = length(ids_intrinsic)
ids_signal = setdiff(collect(1:Ncore), ids_intrinsic)
Nsignals = length(ids_signal)

tstart = 5  # 1-based index: fifth smoothed sample
Dcore_cut = Dcore[:, tstart:end]
Dsignals = Dcore_cut[ids_signal, :]
Dintrinsic = Dcore_cut[ids_intrinsic, :]
u0 = Dintrinsic[:, 1]
num_steps = 1400
t_short = dt * collect(0:num_steps)

Dsigs_interp = [LinearInterpolation(Dsignals[i, 1:num_steps+1], t_short) for i in 1:Nsignals]
tspan = (0.0, t_short[end])

function lorenz!(du, u, p, t)
    γ, β, τ = p
    usig_tmp = [Dsigs_interp[j](t) for j in 1:Nsignals]
    utmp = zeros(Ncore)
    utmp[ids_intrinsic] = u
    utmp[ids_signal] = usig_tmp
    A_RELUu = A * Relu.(utmp)
    W_u = diag(W * (repeat(utmp, 1, Ncore) - transpose(repeat(utmp, 1, Ncore))))
    du_tmp = [(1 / τ) * (a * (utmp[j] - fp1) * (utmp[j] - fp2) * (utmp[j] - fp3) + d[j] + γ * A_RELUu[j] + β * W_u[j]) for j in 1:Ncore]
    du .= du_tmp[ids_intrinsic]
end

p = (γ, β, τ)
prob = ODEProblem(lorenz!, u0, tspan, p)
du0 = similar(u0)
lorenz!(du0, u0, p, 0.0)
println("checks A_sum=$(sum(A)) W_sum=$(sum(W)) d_sum=$(sum(d)) u0_sum=$(sum(u0)) maxA=$(maximum(abs.(A))) maxW=$(maximum(abs.(W)))")
println("du0=$(du0)")
writedlm(joinpath(OUT, "$(RECORDING)_A.csv"), A, ',')
writedlm(joinpath(OUT, "$(RECORDING)_W.csv"), W, ',')
writedlm(joinpath(OUT, "$(RECORDING)_core.csv"), hcat(intrinsic_neurons, string.(d[ids_intrinsic]), string.(u0), string.(du0)), ',')
ut0 = zeros(Ncore)
ut0[ids_intrinsic] = u0
ut0[ids_signal] = [itp(0.0) for itp in Dsigs_interp]
chem0 = A * Relu.(ut0)
gap0 = diag(W * (repeat(ut0, 1, Ncore) - transpose(repeat(ut0, 1, Ncore))))
writedlm(joinpath(OUT, "$(RECORDING)_ut0.csv"), hcat(ut0, chem0, gap0), ',')
sol = solve(prob, Tsit5(), saveat=t_short)

sim = Array(sol)
truth = Dintrinsic[:, 1:num_steps+1]
corr = [cor(truth[i, :], sim[i, :]) for i in 1:Nintrinsic]

function behavior_readout(matrix)
    idx = Dict(name => i for (i, name) in enumerate(intrinsic_neurons))
    # Notebook cell 20: F = mean(Dintrinsic[1:5]), R = mean(Dintrinsic[8:13]).
    forward = vec(mean(matrix[[idx["AVBL"], idx["AVBR"], idx["RIBL"], idx["RIBR"], idx["RID"]], :], dims=1))
    backward = vec(mean(matrix[[idx["AVAL"], idx["AVAR"], idx["RIML"], idx["RIMR"], idx["AVEL"], idx["AVER"]], :], dims=1))
    z = forward .- backward
    return [z[i] > 0.5 ? 1 : (z[i] < -0.5 ? -1 : 0) for i in eachindex(z)]
end

behavior_truth = behavior_readout(truth)
behavior_sim = behavior_readout(sim)
behavior_agreement = mean(behavior_truth .== behavior_sim)
println("recording=$(RECORDING) core=$(Ncore) intrinsic=$(Nintrinsic) signals=$(Nsignals) dt=$(dt)")
println("finite=$(all(isfinite.(sim))) max_abs=$(maximum(abs.(sim))) mean_corr=$(mean(corr)) behavior_agreement=$(behavior_agreement)")
for (i, name) in enumerate(intrinsic_neurons)
    println("corr $(name) $(corr[i])")
end

out = DataFrame(time=t_short)
for (i, name) in enumerate(intrinsic_neurons)
    out[!, Symbol("truth_$(name)")] = truth[i, :]
end
for (i, name) in enumerate(intrinsic_neurons)
    out[!, Symbol("sim_$(name)")] = sim[i, :]
end
out[!, :behavior_truth] = behavior_truth
out[!, :behavior_sim] = behavior_sim
CSV.write(joinpath(OUT, "$(RECORDING)_julia.csv"), out)
println("wrote $(joinpath(OUT, "$(RECORDING)_julia.csv"))")
