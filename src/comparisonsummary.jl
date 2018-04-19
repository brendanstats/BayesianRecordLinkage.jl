#abstract type ComparisonType end
#type Binary <: ComparisonType end
#type Ordinal <: ComparisonType end

#abstract type DataScope end
#type Partial <: DataScope end
#type Full <: DataScope end

#abstract type ComparisonSummary{C <: ComparisonType, S <: DataScope} end
#obsct -> obsvecct
#add obsct back
"""
Structure for summarizing comparison vectors for EM algorithm, penalized likelihood, and MCMC moves.
# Fields
* obsidx: mapping from record pair to comparison vector, indicates which column of obsvecs contains the original record pair
* obsvecs: comparison vectors observed in data, each column corresponds to an observed comparison vector
* obsvecct: number of observations of each comparison vector, length = size(obsvecs, 2)
* counts: vector storing counts of all comparisons for all levels of all fields, length = sum(nlevels)
* obsct: number of observations for each comparison field, length(obsct) = ncomp)
* misct: number of comparisons missing for each comparison, length(obsct) = ncomp, obsct[ii] + misct[ii] = npairs)
* nlevels: number of levels allowed for each comparison
* cmap: which comparison does counts[ii] correspond to
* levelmap: which level does counts[ii] correspond to
* cadj: cadj[ii] + jj yields the index in counts for the jjth level of comparison ii, length(cadj) = ncomp, use range(cadj[ii] + 1, nlevels[ii]) to get indexes for counts of comparison ii
* nrow: number of records in first data base, first dimension of input Array
* ncol: number of records in second data base, second dimension of input Array
* npairs: number of records pairs = nrow * ncol
* ncomp: number of comparisons, computed as third dimension of input Array
"""
struct ComparisonSummary{G <: Integer, T <: Integer}
    obsidx::Array{G, 2}
    obsvecs::Array{T, 2}
    obsvecct::Array{Int64, 1}
    counts::Array{Int64, 1}
    obsct::Array{Int64, 1}
    misct::Array{Int64, 1}
    nlevels::Array{Int64, 1}
    cmap::Array{Int64, 1}
    levelmap::Array{Int64, 1}
    cadj::Array{Int64, 1}
    nrow::Int64
    ncol::Int64
    npairs::Int64
    ncomp::Int64
end

"""
    ComparisonSummary(binary array of dimension 3) -> ComparisonSummary

Will construct a ComparisonSummary object from  a 3 dimension binary array.  The stored comparisons
"""
function ComparisonSummary{A <: AbstractArray{Bool, 3}}(comparisons::A)
    nrow, ncol, ncomp = size(comparisons)
    npairs = nrow * ncol

    nlevels = fill(2, ncomp)
    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]
    
    obsidx = Array{Int64}(nrow, ncol)
    obsvecct = Array{Int64}(0) 
    idxDict = Dict{BitArray{1}, Int64}()
    for jj in 1:ncol, ii in 1:nrow
        if haskey(idxDict, comparisons[ii, jj, :])
            idx = idxDict[comparisons[ii, jj, :]]
            obsidx[ii, jj] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[comparisons[ii, jj, :]] = length(obsvecct)
            obsidx[ii, jj] = length(obsvecct)
        end
    end
    
    obsvecs = fill(Int8(2), ncomp, length(obsvecct))
    for (key, value) in idxDict
        obsvecs[:, value] -= key
    end

    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if obsvecs[ii, jj] == zero(Int8)
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end

    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels,
                             cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function ComparisonSummary(comparisons::Array{Bool, 3})
    nrow, ncol, ncomp = size(comparisons)
    npairs = nrow * ncol

    nlevels = fill(2, ncomp)
    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]
    
    obsidx = Array{Int64}(nrow, ncol)
    obsvecct = Array{Int64}(0) 
    idxDict = Dict{BitArray{1}, Int64}()
    for jj in 1:ncol, ii in 1:nrow
        if haskey(idxDict, comparisons[ii, jj, :])
            idx = idxDict[comparisons[ii, jj, :]]
            obsidx[ii, jj] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[comparisons[ii, jj, :]] = length(obsvecct)
            obsidx[ii, jj] = length(obsvecct)
        end
    end
    
    obsvecs = fill(Int8(2), ncomp, length(obsvecct))
    for (key, value) in idxDict
        obsvecs[:, value] -= key
    end

    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if obsvecs[ii, jj] == zero(Int8)
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end

    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels,
                             cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end


function ComparisonSummary{G <: Integer}(comparisons::Array{G, 3}, nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1:2)))
    nrow, ncol, ncomp = size(comparisons)
    npairs = nrow * ncol

    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]
    
    obsidx = Array{Int64}(nrow, ncol)
    obsvecct = Array{Int64}(0) 
    idxDict = Dict{Array{G, 1}, Int64}()
    for jj in 1:ncol, ii in 1:nrow
        if haskey(idxDict, comparisons[ii, jj, :])
            idx = idxDict[comparisons[ii, jj, :]]
            obsidx[ii, jj] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[comparisons[ii, jj, :]] = length(obsvecct)
            obsidx[ii, jj] = length(obsvecct)
        end
    end
    
    obsvecs = Array{G}(ncomp, length(obsvecct))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end

    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if obsvecs[ii, jj] == zero(G)
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end

    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels,
                             cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

ComparisonSummary{G <: Integer}(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::ComparisonSummary) =
    ComparisonSummary(compsum.obsidx[rowperm, colperm], compsum.obsvecs, compsum.obsvecct, compsum.counts,
                      compsum.obsct, compsum.misct, compsum.nlevels, compsum.cmap, compsum.levelmap, compsum.cadj,
                      compsum.nrow, compsum.ncol, compsum.npairs, compsum.ncomp)

struct SparseComparisonSummary{G <: Integer, T <: Integer}
    obsidx::SparseMatrixCSC{G, Int64} #assume indexed by Int64s...
    obsvecs::Array{T, 2}
    obsvecct::Array{Int64, 1}
    counts::Array{Int64, 1}
    obsct::Array{Int64, 1}
    misct::Array{Int64, 1}
    nlevels::Array{Int64, 1}
    cmap::Array{Int64, 1}
    levelmap::Array{Int64, 1}
    cadj::Array{Int64, 1}
    nrow::Int64
    ncol::Int64
    npairs::Int64
    ncomp::Int64
end

#defined assuming each row of input is an observation, each column would be more efficient...
function SparseComparisonSummary{G <: Integer}(rows::Array{Int64, 1}, cols::Array{Int64, 1},
                                               comparisons::Array{G, 2},
                                               nrow::Integer = maximum(rows),
                                               ncol::Integer = maximum(cols),
                                               nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1)))
    npairs, ncomp = size(comparisons)

    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]

    obsidxvec = Array{Int64}(npairs)
    obsvecct = Array{Int64}(0) 
    idxDict = Dict{Array{G, 1}, Int64}()
    for ii in 1:npairs
        if haskey(idxDict, comparisons[ii, :])
            idx = idxDict[comparisons[ii, :]]
            obsidxvec[ii] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[comparisons[ii, :]] = length(obsvecct)
            obsidxvec[ii] = length(obsvecct)
        end
    end

    #defined later than for non-sparse
    obsidx = sparse(rows, cols, obsidxvec, nrow, ncol)
    
    obsvecs = Array{G}(ncomp, length(obsvecct))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end

    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if obsvecs[ii, jj] == zero(G)
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels,
                             cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

SparseComparisonSummary{G <: Integer}(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::SparseComparisonSummary) =
    SparseComparisonSummary(compsum.obsidx[rowperm, colperm], compsum.obsvecs, compsum.obsvecct, compsum.counts,
                      compsum.obsct, compsum.misct, compsum.nlevels, compsum.cmap, compsum.levelmap, compsum.cadj,
                      compsum.nrow, compsum.ncol, compsum.npairs, compsum.ncomp)

function counts_delta{G <: Integer, T <: Integer}(compsum::Union{ComparisonSummary{G, T}, SparseComparisonSummary{G, T}})
    deltas = zeros(Int8, length(compsum.counts), size(compsum.obsvecs, 2))
    for jj in 1:size(compsum.obsvecs, 2), ii in 1:size(compsum.obsvecs, 1)
        if compsum.obsvecs[ii, jj] != zero(T)
            deltas[compsum.cadj[ii]  + compsum.obsvecs[ii, jj], jj] = one(Int8)
        end
    end
    return deltas
end
