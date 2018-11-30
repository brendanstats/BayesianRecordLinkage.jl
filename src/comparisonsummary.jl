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
function mapping_variables(nlevels::Array{Int64, 1})
    cmap = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(nlevels))
    levelmap = mapreduce(x -> 1:x, vcat, nlevels)
    cadj = [0; cumsum(nlevels)[1:end-1]]
    return cmap, levelmap, cadj
end

##Construct 
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

function comparison_variables(filenames::Array{String, 1}, nrow::Int64, ncol::Int64, ncomp::Int64; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = '\t', verbose::Bool = true)

    nlines = mapreduce(countlines, +, filenames)
    if header
        nlines -= length(filenames)
    end
    
    rows = Array{Int64}(undef, nlines)
    cols = Array{Int64}(undef, nlines)
    obsidxvec = Array{idxType}(nlines)

    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{Int64, 1}, Int64}()
    
    idx = 0
    for f in filenames

        if verbose
            println(f)
        end
        
        ##read in file
        if header
            comp = readdlm(f, sep, Int64, header = header)[1]
        else
            comp = readdlm(f, sep, Int64, header = header)
        end
        
        for ii in 1:size(comp, 1)
            idx += 1
            rows[idx] = comp[ii, 1]
            cols[idx] = comp[ii, 2]
            obsvec = comp[ii, 3:end]
            if haskey(idxDict, obsvec)
                compid = idxDict[obsvec]
                obsidxvec[idx] = compid
                obsvecct[compid] += 1
            else
                push!(obsvecct, 1)
                idxDict[obsvec] = length(obsvecct)
                obsidxvec[idx] = length(obsvecct)
            end
        end
    end
    println(nlines)
    println(idx)
    println(maximum(rows))
    println(maximum(cols))
    
    obsidx = sparse(rows, cols, obsidxvec, nrow, ncol)

    rows = true
    cols = true
    obsidxvec = true
    
    #=
    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{Int64, 1}, Int64}()
    obsidx = sparse(Int64[], Int64[], idxType[], nrow, ncol)
    
    for f in filenames

        if verbose
            println(f)
        end
                    
        ##read in file
        if header
            comparisons = readdlm(f, sep, Int64, header = header)[1]
        else
            comparisons = readdlm(f, sep, Int64, header = header)
        end

        ##add contents to file
        obsidxvec = Array{idxType}(size(comparisons, 1))
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

        rows = comparisons[:, 1]
        cols = comparisons[:, 2]
        keep = trues(length(rows))
        for ii in length(rows)
            if !iszero(obsidx[rows[ii], cols[ii]])
                keep[ii] = false
            end
        end
        obsidx += sparse(rows[keep], cols[keep], obsidxvec[keep], nrow, ncol)
        #@time for (row, col, idx) in zip(rows, cols, obsidxvec)
        #    @time if iszero(obsidx[row, col])
        #        @time obsidx[row, col] = idx
        #    else
        #        if obsidx[row, col] != idx
        #            warn("multiple comparison vectors compute for row: $row and col: $col")
        #        end
        #    end
        #end
    end
    =#
    obsvecs = Array{obsType}(undef, ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    
    return obsidx, obsvecs, obsvecct
end


function comparison_variables(namelist::Array{Array{String, 1}, 1}, nrow::Int64, ncol::Int64, ncomp::Int64; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = 't', verbose::Bool = true)

    if length(namelist) == 1
        return comparison_variables(namelist[1], nrow, ncol, ncomp, idxType = idxType, obsTypeobsType, header = header, sep = sep, verbose = verbose)
    end

    obsvecct = Array{Int64}(undef, 0) 
    idxDict = Dict{Array{Int64, 1}, Int64}()
    obsidx = sparse(Int64[], Int64[], idxType[], nrow, ncol)
    
    for filenames in namelist

        ##List specific variables
        nlines = mapreduce(countlines, +, filenames)
        if header
            nlines -= length(filenames)
        end
        rows = Array{Int64}(undef, nlines)
        cols = Array{Int64}(undef, nlines)
        obsidxvec = Array{idxType}(nlines)

        idx = 0
        for f in filenames
            
            if verbose
                println(f)
            end
            
            ##read in file
            if header
                comp = readdlm(f, sep, Int64, header = header)[1]
            else
                comp = readdlm(f, sep, Int64, header = header)
            end
            
            for ii in 1:size(comp, 1)
                idx += 1
                rows[idx] = comp[ii, 1]
                cols[idx] = comp[ii, 2]
                obsvec = comp[ii, 3:end]
                if haskey(idxDict, obsvec)
                    compid = idxDict[obsvec]
                    obsidxvec[idx] = compid
                    obsvecct[compid] += 1
                else
                    push!(obsvecct, 1)
                    idxDict[obsvec] = length(obsvecct)
                    obsidxvec[idx] = length(obsvecct)
                end
            end
        end
        keep = trues(length(rows))
        for ii in length(rows)
            if !iszero(obsidx[rows[ii], cols[ii]])
                keep[ii] = false
            end
        end
        obsidx += sparse(rows[keep], cols[keep], obsidxvec[keep], nrow, ncol)
    end
    
    obsvecs = Array{obsType}(undef, ncomp, maximum(values(idxDict)))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    
    return obsidx, obsvecs, obsvecct
end

##Count numbers of observations
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


"""
    ComparisonSummary(binary array of dimension 3) -> ComparisonSummary

Will construct a ComparisonSummary object from  a 3 dimension binary array.  The stored comparisons
"""
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

#ncomp = length(nlevels)
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

struct SparseComparisonSummary{G <: Integer, Tv <: Integer, Ti <: Integer}
    obsidx::SparseMatrixCSC{Tv, Ti} #assume indexed by Int64s...
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

#defined assuming each row of input is an observation, each column would be more efficient...
function SparseComparisonSummary(rows::Array{Ti, 1},
                                 cols::Array{Ti, 1},
                                 comparisons::Array{G, 2},
                                 nrow::Integer = maximum(rows),
                                 ncol::Integer = maximum(cols),
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
                                               nrow::Int64 = Int64.(maximum(comparisons[:, 1])),
                                               ncol::Int64 = Int64.(maximum(comparisons[:, 2])),
                                               nlevels::Array{Int64, 1} = Int64.(vec(maximum(comparisons, 1))[3:end])) where G <: Integer

    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)    
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)
    
    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function SparseComparisonSummary(filename::String, nrow::Int64, ncol::Int64, nlevels::Array{Int64, 1}; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = '\t')

    ##Mapping variables
    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    
    ##Define variables for loading data
    obsidx = spzeros(idxType, nrow, ncol)
    idxDict = Dict{Array{obsType, 1}, idxType}()
    obsvecct = Int64[]
    
    ##read in file
    if header
        comparisons = readdlm(filename, sep, idxType, header = header)[1]
    else
        comparisons = readdlm(filename, sep, idxType, header = header)
    end
    obsidx, obsvecs, obsvecct = comparison_variables(comparisons, nrow, ncol, ncomp, idxType = idxType, obsType = obsType)
    #obsidx += sparse(rows, cols, comparisonIdx, nrow, ncol)
    
    ##Count numbers of observations
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function SparseComparisonSummary(filenames::Array{String, 1}, nrow::Int64, ncol::Int64, nlevels::Array{Int64, 1}; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = '\t')

    ##Mapping variables
    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(filenames, nrow, ncol, ncomp, idxType = idxType, obsType = obsType, header = header, sep = sep)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function SparseComparisonSummary(filenames::Array{Array{String, 1}, 1}, nrow::Int64, ncol::Int64, nlevels::Array{Int64, 1}; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = '\t')

    ##Mapping variables
    ncomp = length(nlevels)
    cmap, levelmap, cadj = mapping_variables(nlevels)
    obsidx, obsvecs, obsvecct = comparison_variables(filenames, nrow, ncol, ncomp, idxType = idxType, obsType = obsType, header = header, sep = sep)
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

function extract_integers(s::String, nobs::Integer, idxType::DataType, obsType::DataType, sep::Char)
    obs = Array{obsType}(undef, nobs)
    idx1 = start(s)
    n = endof(s)
    r = search(s, sep, idx1)
    j, k = first(r), nextind(s, last(r))

    row = parse(idxType, s[idx1:prevind(s, j)])

    idx1 = k
    r = search(s, sep, k)
    j, k = first(r), nextind(s,last(r))

    col = parse(idxType, s[idx1:prevind(s, j)])

    ii = 1
    while ii != ncomp
        idx1 = k
        r = search(s, sep, k)
        j, k = first(r), nextind(s, last(r))
        obs[ii] = parse(obsType, s[idx1:prevind(s, j)])
        ii += 1
    end
    obs[ii] = parse(obsType, s[k:n])
    return row, col, obs
end

function extract_integers!(row::G, col::G, obs::Array{T, 1}, s::String, nobs::Integer, sep::Char) where {G <: Integer, T <: Integer}
    idx1 = start(s)
    n = endof(s)
    r = search(s, sep, idx1)
    j, k = first(r), nextind(s, last(r))

    row = parse(G, s[idx1:prevind(s, j)])

    idx1 = k
    r = search(s, sep, k)
    j, k = first(r), nextind(s,last(r))

    col = parse(G, s[idx1:prevind(s, j)])

    ii = 1
    while ii != nobs
        idx1 = k
        r = search(s, sep, k)
        j, k = first(r), nextind(s, last(r))
        obs[ii] = parse(T, s[idx1:prevind(s, j)])
        ii += 1
    end
    obs[ii] = parse(T, s[k:n])
    return row, col, obs
end

function stream_comparisonsummary(filename::String, nrow::Int64, ncol::Int64, nlevels::Array{Int64, 1}; idxType::DataType = Int64, obsType::DataType = Int8, header::Bool = true, sep::Char = '\t')

    ##Mapping variables
    cmap, levelmap, cadj = mapping_variables(nlevels)
    ncomp = length(nlevels)
    
    ##Define variables for loading data
    obsidx = spzeros(idxType, nrow, ncol)
    idxDict = Dict{Array{obsType, 1}, idxType}()
    obsvecct = Int64[]

    ##Variables to fill
    row = zero(idxType)
    col = zero(idxType)
    obsvec = zeros(obsType, ncomp)
    
    ##read in file
    open(filename) do f
        if header #skip first line if header = true
            readline(f)
        end
        while !eof(f)
            row, col, obsvec = extract_integers!(row, col, obsvec, readline(f), ncomp, sep)
            #row, col, obsvec = extract_integers(readline(f), ncomp, idxType, obsType, sep)
            if haskey(idxDict, obsvec)
                idx = idxDict[obsvec]
                obsidx[row, col] = idx
                obsvecct[idx] += 1
            else
                push!(obsvecct, 1)
                idxDict[copy(obsvec)] = length(obsvecct)
                idx = idxDict[obsvec]
                obsidx[row, col] = idx
            end
        end
    end

    ##Combine observed vectors in array
    obsvecs = Array{obsType}(undef, ncomp, length(obsvecct))
    for (key, value) in idxDict
        obsvecs[:, value] = key
    end
    
    ##Count numbers of observations
    npairs, counts, obsct, misct = count_variables(obsidx, obsvecs, obsvecct, cmap, cadj)

    return SparseComparisonSummary(obsidx, obsvecs, obsvecct, counts, obsct, misct, nlevels, cmap, levelmap, cadj, nrow, ncol, npairs, ncomp)
end

#hcat(findnz(compsum.obsidx)...)
#sparse(I,J,V)
#obsvecs

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

function counts_delta(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    deltas = zeros(Int8, length(compsum.counts), size(compsum.obsvecs, 2))
    for jj in 1:size(compsum.obsvecs, 2), ii in 1:size(compsum.obsvecs, 1)
        if !iszero(compsum.obsvecs[ii, jj])
            deltas[compsum.cadj[ii]  + compsum.obsvecs[ii, jj], jj] = one(Int8)
        end
    end
    return deltas
end

function obs_delta(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    deltas = zeros(Int8, size(compsum.obsvecs))
    for jj in 1:size(compsum.obsvecs, 2), ii in 1:size(compsum.obsvecs, 1)
        if !iszero(compsum.obsvecs[ii, jj])
            deltas[ii, jj] = one(Int8)
        end
    end
    return deltas
end
