
"""
    ComparisonSummary{G <: Integer, T <: Integer}

Structure for summarizing comparison vectors for EM algorithm, penalized likelihood, and MCMC moves.



# Fields

* `obsidx::Array{T, 2}`: mapping from record pair to comparison vector, indicates which column of obsvecs contains the original record pair
* `obsvecs::Array{G, 2}`: comparison vectors observed in data, each column corresponds to an
observed comparison vector. Zero entries indicate a missing value for the comparison-feature pair.
* `obsvecct::Array{Int64, 1}`: number of observations of each comparison vector, length = size(obsvecs, 2)
* `counts::Array{Int64, 1}`: vector storing counts of all comparisons for all levels of all fields, length = sum(nlevels)
* `obsct::Array{Int64, 1}`: number of observations for each comparison field, length(obsct) = ncomp)
* `misct::Array{Int64, 1}`: number of comparisons missing for each comparison, length(obsct) = ncomp, obsct[ii] + misct[ii] = npairs)
* `nlevels::Array{Int64, 1}`: number of levels allowed for each comparison
* `cmap::Array{Int64, 1}`: which comparison does counts[ii] correspond to
* `levelmap::Array{Int64, 1}`: which level does counts[ii] correspond to
* `cadj::Array{Int64, 1}`: cadj[ii] + jj yields the index in counts for the jjth level of comparison ii, length(cadj) = ncomp, use range(cadj[ii] + 1, nlevels[ii]) to get indexes for counts of comparison ii
* `nrow::Int64`: number of records in first data base, first dimension of input Array
* `ncol::Int64`: number of records in second data base, second dimension of input Array
* `npairs::Int64`: number of records pairs = nrow * ncol
* `ncomp::Int64`: number of comparisons, computed as third dimension of input Array

# Constructors

    ComparisonSummary(comparisons::BitArray{3})
    ComparisonSummary(comparisons::Array{Bool, 3})
    ComparisonSummary(comparisons::Array{G, 3}, nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1:2))) where G <: Integer
    ComparisonSummary(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::ComparisonSummary) where G <: Integer

See also: [`SparseComparisonSummary`](@ref), [`mapping_variables`](@ref), [`comparison_variables`](@ref), [`count_variables`](@ref)
"""
struct ComparisonSummary{G <: Integer, T <: Integer}
    obsidx::Array{T, 2}
    obsvecs::Array{G, 2}
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

##Construct mapping variables
"""
    mapping_variables(nlevels::Array{Int64, 1})

Transform `nlevels` listing number of bins for each comparison field into set of mapping variables.

Using in the construction of `ComparisonSummary` and `SparseComparisonSummary` types.  Return a tuple
of cmap (index to comparison), levelmap (index to level), and cadj (comparison adjustment).

See also: [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`comparison_variables`](@ref), [`count_variables`](@ref)
"""
function mapping_variables(nlevels::Array{Int64, 1})
    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]
    return cmap, levelmap, cadj
end

##Construct
"""
    comparison_variables(comparisons::BitArray{3}, nrow::Int64, ncol::Int64, ncomp::Int64)
    comparison_variables(comparisons::Array{Bool, 3}, nrow::Int64, ncol::Int64, ncomp::Int64)
    comparison_variables(comparisons::Array{G, 3}, nrow::Int64, ncol::Int64, ncomp::Int64) where G <: Integer
    comparison_variables(rows::Array{Ti, 1}, cols::Array{Ti, 1}, comparisons::Array{G, 2}, nrow::Int64, ncol::Int64, ncomp::Int64, Tv::DataType = Int64) where {G <: Integer, Ti <: Integer}
    comparison_variables(comparisons::Array{G, 2}, nrow::Int64, ncol::Int64, ncomp::Int64; idxType::DataType = G, obsType::DataType = Int8) where G <: Integer

Helper function for generating `obsidx`, `obsvecs`, and `obsvecct` fields in `ComparisonSummary` and `SparseComparisonSummary` types.

If `comparisons` is of type `Array{3}` then `obsidx` is a full matrix and a `ComparisonSummary` will be generated.  If not then `obsidx` is a `SparseMatrixCSC` and a `SparseComparisonSummary` will be generated.  If seperate `rows` and `cols` variables are missing then the first two colums are assumed to correspond to the row and column indicies respectively.

See also: [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`mapping_variables`](@ref), [`count_variables`](@ref)
"""
function comparison_variables(comparisons::BitArray{3}, nrow::Int64, ncol::Int64, ncomp::Int64)
    obsidx = Array{Int64}(undef, nrow, ncol)
    obsvecct = Array{Int64}(undef, 0) 
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
    
    obsvecs = fill(Int8(2), ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] -= key
    end
    return obsidx, obsvecs, obsvecct
end

function comparison_variables(comparisons::Array{Bool, 3}, nrow::Int64, ncol::Int64, ncomp::Int64)
    obsidx = Array{Int64}(undef, nrow, ncol)
    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{Bool, 1}, Int64}()
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
    
    obsvecs = fill(Int8(2), ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] -= key
    end
    return obsidx, obsvecs, obsvecct
end

function comparison_variables(comparisons::Array{G, 3}, nrow::Int64, ncol::Int64, ncomp::Int64) where G <: Integer
    obsidx = Array{Int64}(undef, nrow, ncol)
    obsvecct = Array{Int64}(undef, 0) 
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

    obsvecs = zeros(G, ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    return obsidx, obsvecs, obsvecct
end

function comparison_variables(rows::Array{Ti, 1}, cols::Array{Ti, 1}, comparisons::Array{G, 2}, nrow::Int64, ncol::Int64, ncomp::Int64, Tv::DataType = Int64) where {G <: Integer, Ti <: Integer}
    
    obsidxvec = Array{Tv}(undef, size(comparisons, 1))
    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{G, 1}, Tv}()
    for ii in 1:size(comparisons, 1)
        obsvec = comparisons[ii, :]
        if haskey(idxDict, obsvec)
            idx = idxDict[obsvec]
            obsidxvec[ii] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[obsvec] = length(obsvecct)
            obsidxvec[ii] = length(obsvecct)
        end
    end

    #defined later than for non-sparse
    obsidx = sparse(rows, cols, obsidxvec, nrow, ncol)
    
    obsvecs = Array{Ti}(undef, ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    
    return obsidx, obsvecs, obsvecct
end

function comparison_variables(comparisons::Array{G, 2}, nrow::Int64, ncol::Int64, ncomp::Int64; idxType::DataType = G, obsType::DataType = Int8) where G <: Integer

    obsidxvec = Array{idxType}(undef, size(comparisons, 1))
    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{G, 1}, Int64}()
    for ii in 1:size(comparisons, 1)
        obsvec = comparisons[ii, 3:end]
        if haskey(idxDict, obsvec)
            idx = idxDict[obsvec]
            obsidxvec[ii] = idx
            obsvecct[idx] += 1
        else
            push!(obsvecct, 1)
            idxDict[obsvec] = length(obsvecct)
            obsidxvec[ii] = length(obsvecct)
        end
    end

    #defined later than for non-sparse
    obsidx = sparse(Int64.(comparisons[:, 1]), Int64.(comparisons[:, 2]), obsidxvec, nrow, ncol)
    
    obsvecs = Array{obsType}(undef, ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    
    return obsidx, obsvecs, obsvecct
end

##Count numbers of observations
"""
    count_variables(obsidx::Array{G, 2}, obsvecs::Array{T, 2}, obsvecct::Array{Int64, 1}, cmap::Array{Int64, 1}, cadj::Array{Int64, 1}) where {G <: Integer, T <: Integer}
    count_variables(obsidx::SparseMatrixCSC{Tv, Ti}, obsvecs::Array{G, 2}, obsvecct::Array{Int64, 1}, cmap::Array{Int64, 1}, cadj::Array{Int64, 1}) where {G <: Integer, Tv <: Integer, Ti <: Integer}

Helper function used to generate `npairs`, `counts`, `obsct`, and `misct` fields in `ComparisonSummary` and `SparseComparisonSummary` types.

See also: [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`mapping_variables`](@ref), [`comparison_variables`](@ref)
"""
function count_variables(obsidx::Array{G, 2}, obsvecs::Array{T, 2}, obsvecct::Array{Int64, 1}, cmap::Array{Int64, 1}, cadj::Array{Int64, 1}) where {G <: Integer, T <: Integer}
    ncomp = size(obsvecs, 1)
    npairs = size(obsidx, 1) * size(obsidx, 2)
    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if iszero(obsvecs[ii, jj])
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end
    return npairs, counts, obsct, misct
end

function count_variables(obsidx::SparseMatrixCSC{Tv, Ti}, obsvecs::Array{G, 2}, obsvecct::Array{Int64, 1}, cmap::Array{Int64, 1}, cadj::Array{Int64, 1}) where {G <: Integer, Tv <: Integer, Ti <: Integer}
    ncomp = size(obsvecs, 1)
    npairs = nnz(obsidx)
    counts = zeros(Int64, length(cmap))
    obsct = zeros(Int64, ncomp)
    misct = zeros(Int64, ncomp)
    for jj in 1:length(obsvecct), ii in 1:ncomp
        if iszero(obsvecs[ii, jj])
            misct[ii] += obsvecct[jj]
        else
            obsct[ii] += obsvecct[jj]
            counts[cadj[ii] + obsvecs[ii, jj]] += obsvecct[jj]
        end
    end
    return npairs, counts, obsct, misct
end

function ComparisonSummary(comparisons::BitArray{3})
    
    nrow, ncol, ncomp = size(comparisons)
    nlevels = fill(Int64(2), ncomp)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function ComparisonSummary(comparisons::Array{Bool, 3})

    nrow, ncol, ncomp = size(comparisons)
    nlevels = fill(Int64(2), ncomp)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)
    
    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function ComparisonSummary(comparisons::Array{G, 3}, nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1:2))) where G <: Integer

    nrow, ncol, ncomp = size(comparisons)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)
    
    return ComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

ComparisonSummary(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::ComparisonSummary) where G <: Integer =
    ComparisonSummary(compsum.obsidx[rowperm, colperm], compsum.obsvecs, compsum.obsvecct, compsum.counts,
                      compsum.obsct, compsum.misct, compsum.nlevels, compsum.cmap, compsum.levelmap, compsum.cadj,
                      compsum.nrow, compsum.ncol, compsum.npairs, compsum.ncomp)

"""
    SparseComparisonSummary{G <: Integer, Tv <: Integer, Ti <: Integer}

Structure for summarizing comparison vectors for EM algorithm, penalized likelihood, and MCMC moves.  Comparisons need not be included for all possible record pairs.

# Fields

* `obsidx::SparseMatrixCSC{Tv, Ti}`: mapping from record pair to comparison vector, indicates which column of obsvecs contains the original record pair
* `obsvecs::Array{G, 2}`: comparison vectors observed in data, each column corresponds to an
observed comparison vector.  Zero entries indicate a missing value for the comparison-feature pair.
* `obsvecct::Array{Int64, 1}`: number of observations of each comparison vector, length = size(obsvecs, 2)
* `counts::Array{Int64, 1}`: vector storing counts of all comparisons for all levels of all fields, length = sum(nlevels)
* `obsct::Array{Int64, 1}`: number of observations for each comparison field, length(obsct) = ncomp)
* `misct::Array{Int64, 1}`: number of comparisons missing for each comparison, length(obsct) = ncomp, obsct[ii] + misct[ii] = npairs)
* `nlevels::Array{Int64, 1}`: number of levels allowed for each comparison
* `cmap::Array{Int64, 1}`: which comparison does counts[ii] correspond to
* `levelmap::Array{Int64, 1}`: which level does counts[ii] correspond to
* `cadj::Array{Int64, 1}`: cadj[ii] + jj yields the index in counts for the jjth level of comparison ii, length(cadj) = ncomp, use range(cadj[ii] + 1, nlevels[ii]) to get indexes for counts of comparison ii
* `nrow::Int64`: number of records in first data base, first dimension of input Array
* `ncol::Int64`: number of records in second data base, second dimension of input Array
* `npairs::Int64`: number of records pairs = nrow * ncol
* `ncomp::Int64`: number of comparisons, computed as third dimension of input Array#

# Constructors

    SparseComparisonSummary(rows::Array{Ti, 1}, cols::Array{Ti, 1}, comparisons::Array{G, 2},
                            nrow::Integer = maximum(rows), ncol::Integer = maximum(cols),
                            nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1))) where {G <: Integer, Ti <: Integer}
    SparseComparisonSummary(comparisons::Array{G, 2},
                             nrow::Int64 = Int64.(maximum(comparisons[:, 1])), ncol::Int64 = Int64.(maximum(comparisons[:, 2])),
                             nlevels::Array{Int64, 1} = Int64.(vec(maximum(comparisons, 1))[3:end])) where G <: Integer
    SparseComparisonSummary(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::SparseComparisonSummary) where G <: Integer

See also: [`ComparisonSummary`](@ref), [`mapping_variables`](@ref), [`comparison_variables`](@ref), [`count_variables`](@ref)
"""
struct SparseComparisonSummary{G <: Integer, Tv <: Integer, Ti <: Integer}
    obsidx::SparseMatrixCSC{Tv, Ti}
    obsvecs::Array{G, 2}
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

function SparseComparisonSummary(rows::Array{Ti, 1}, cols::Array{Ti, 1}, comparisons::Array{G, 2},
                                 nrow::Integer = maximum(rows), ncol::Integer = maximum(cols),
                                 nlevels::Array{<:Integer, 1} = vec(maximum(comparisons, 1))) where {G <: Integer, Ti <: Integer}
    if length(rows) != length(cols)
        error("rows and columns must be the same length")
    end

    nunique = prod(nlevels .+ 1)
    if size(nunique, 1) < typemax(Int8)
        Tv = Int8
    elseif size(nunique, 1) < typemax(Int16)
        Tv = Int16
    elseif size(nunique, 1) < typemax(Int32)
        Tv = Int32
    else
        Tv = Int64
    end
    
    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(rows, cols, comparisons, nrow, ncol, ncomp, Tv)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function SparseComparisonSummary(comparisons::Array{G, 2},
                                 nrow::Int64 = Int64.(maximum(comparisons[:, 1])), ncol::Int64 = Int64.(maximum(comparisons[:, 2])),
                                 nlevels::Array{Int64, 1} = Int64.(vec(maximum(comparisons, 1))[3:end])) where G <: Integer

    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)    
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)
    
    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

"""
    merge_comparisonsummary(CS1::SparseComparisonSummary, CS2::SparseComparisonSummary)

Combine two `SparseComparisonSummary` variables, observations can have a non-trivial intersection.

`CS2.obsidx` are first converted into `CS1.obsidx` scales since `CS1.obsvecs` and `CS2.obsvecs` may not
have the same indexing scheme.  After converting `CS1.obsidx` and `CS2.obsidx` in this case comparison
indicies should now agree.  If they do not the combined compsum.obsidx is given an `obsidx` value of -1
to signify the disagreement.
"""
function merge_comparisonsummary(CS1::SparseComparisonSummary, CS2::SparseComparisonSummary)
    
    if CS1.ncomp != CS2.ncomp
        error("ncomp (number of comparisons) must match")
    end
    ncomp = CS1.ncomp
    nlevels = map(1:CS1.ncomp) do ii
        max(CS1.nlevels[ii], CS2.nlevels[ii])
    end
    nrow = max(CS1.nrow, CS2.nrow)
    ncol = max(CS1.ncol, CS2.ncol)
    cmap, levelmap, cadj = mapping_variables(nlevels)

    obsType = eltype(CS1.obsvecs)
    idxType = eltype(CS1.obsidx)
    
    ##Map for obsvecs in first comparison summary
    idxDict = Dict{Array{obsType, 1}, idxType}()
    for jj in 1:size(CS1.obsvecs, 2)
        idxDict[CS1.obsvecs[:, jj]] = jj
    end

    ##Map obsvecs in second comparison summary to first
    obsvecs = CS1.obsvecs
    nvecs = size(CS1.obsvecs, 2)
    oldidx2newidx = zeros(idxType, size(CS2.obsvecs, 2))
    for jj in 1:size(CS2.obsvecs, 2)
        obsvec = CS2.obsvecs[:, jj]
        if haskey(idxDict, obsvec)
            oldidx2newidx[jj] = idxDict[obsvec]
        else
            nvecs += 1
            oldidx2newidx[jj] = nvecs
            obsvecs = hcat(obsvecs, obsvec)
        end
    end

    function idx_combine(x::G, y::G) where G <: Real
        if x == y
            return x
        else
            return -one(G)
        end
    end
    
    ##Add observations from second comparison to those of first
    cols = [findnz(CS1.obsidx)[2]; findnz(CS2.obsidx)[2]]
    rows = [rowvals(CS1.obsidx); rowvals(CS2.obsidx)]
    vals = [nonzeros(CS1.obsidx); oldidx2newidx[nonzeros(CS2.obsidx)]]
    obsidx = sparse(rows, cols, vals, nrow, ncol, idx_combine) #combine using minimum since they should match, default is + for duplicates

    println(nnz(obsidx))
    rows = true
    cols = true
    vals = true
    println(nnz(obsidx))

    ##Combine Counts - must be recomputed from scratch since duplicates will be combined...
    obsvecct = zeros(Int64, nvecs)
    for idx in nonzeros(obsidx)
        obsvecct[idx] += one(Int64)
    end
    
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)
    
    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

SparseComparisonSummary(rowperm::Array{G, 1}, colperm::Array{G, 1}, compsum::SparseComparisonSummary) where G <: Integer =
    SparseComparisonSummary(compsum.obsidx[rowperm, colperm], compsum.obsvecs, compsum.obsvecct, compsum.counts,
                      compsum.obsct, compsum.misct, compsum.nlevels, compsum.cmap, compsum.levelmap, compsum.cadj,
                      compsum.nrow, compsum.ncol, compsum.npairs, compsum.ncomp)

"""
    get_obsidxcounts(compsum::Union{ComparisonSummary, SparseComparisonSummary})

Transform each observation (column) of `compsum.obsvecs` to binary vector of same length as `compsum.counts`.

The returned value makes it easier to compute overall comparison counts for each feature from
a count of comparison indices computed based soley on `compsum.obsidx` values.
"""
function get_obsidxcounts(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    deltas = zeros(Int, length(compsum.counts), size(compsum.obsvecs, 2))
    for jj in 1:size(compsum.obsvecs, 2), ii in 1:size(compsum.obsvecs, 1)
        if !iszero(compsum.obsvecs[ii, jj])
            deltas[compsum.cadj[ii]  + compsum.obsvecs[ii, jj], jj] = one(Int)
        end
    end
    return deltas
end

"""
    get_obsidxobs(compsum::Union{ComparisonSummary, SparseComparisonSummary})

Transform each observation (column) of `compsum.obsvecs` to binary vector of same length as `compsum.obsct`.

The returned value makes it easier to compute overall observation counts for each feature from
a count of comparison indices computed based soley on `compsum.obsidx` values.
"""
function get_obsidxobs(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    deltas = zeros(Int, size(compsum.obsvecs))
    for jj in 1:size(compsum.obsvecs, 2), ii in 1:size(compsum.obsvecs, 1)
        if !iszero(compsum.obsvecs[ii, jj])
            deltas[ii, jj] = one(Int)
        end
    end
    return deltas
end
