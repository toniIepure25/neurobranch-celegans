# Export NeuroPAL identities from the Baskoylu & Flavell JLD2 container.
#
# Julia is used only for this one-time conversion: the JLD2 string encoding is
# not readable by h5py (it stores a JLD2-specific variable-length string type).
# The output is a plain TSV with a documented 0-based trace index, so the Python
# pipeline has no Julia dependency at analysis time.

using JLD2

const JLD2_PATH = joinpath(@__DIR__, "..", "data", "raw", "baskoylu2026", "files", "NeuroPAL_labels.jld2")
const OUT_PATH = joinpath(@__DIR__, "..", "data", "raw", "baskoylu2026", "neuropal_labels.tsv")

labels = load(JLD2_PATH)["dict_match_dict"]
open(OUT_PATH, "w") do io
    println(io, "animal_id\ttrace_index\tlabel\tneuron_class\tconfidence")
    for animal in sort(collect(keys(labels)))
        by_index = labels[animal][1]
        for (one_based_index, meta) in sort(collect(by_index), by=first)
            # JLD2 stores 1-based row indices into the Julia trace matrix; the
            # Python adapter uses 0-based indices.
            println(
                io,
                animal, '\t',
                Int(one_based_index) - 1, '\t',
                string(meta["label"]), '\t',
                string(meta["neuron_class"]), '\t',
                Int(meta["confidence"]),
            )
        end
    end
end
println("wrote ", OUT_PATH)
