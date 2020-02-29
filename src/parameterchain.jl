"""
    ParameterChain{G <: Integer, T <: AbstractFloat}

Store either MCMC parameter traces or parameter evolution produced by a penalized likelihood estimator.

# Fields

* `C::Array{G, 2}`: Either row and column indices paired with link counts or indicies paired with steps / iterations in which they appeared.
* `nlinks::Union{Array{G, 1}, Array{G, 2}}`: Total number of links at each step / iteration.
* `pM::Array{T, 2}`: M parameters at each iteration, size(pM, 1) == nsteps
* `pU::Array{T, 2}`: U parameters at each iteration, size(pU, 1) == nsteps
* `nsteps::Int`: Total number of steps / iterations
* `linktrace::Bool`: Indicator if `C` contains link counts or a trace of the entire link structure.

If `linktrace == true` then each row of `C` will correspond to an interval in which a link persisted in the form
* C[ii, 1]: link row.
* C[ii, 2]: link column.
* C[ii, jj] where jj > 2 and jj < size(C, 2) - 1: other fields (flexible).
* C[ii, end - 1]: first step / iteration link was present.
* C[ii, end]: last step / iteration link was present.
This allows both the frequency with which each link appeared to be reconstructed as well as the exact LinkMatrix
at any iteration to be reconstructed.

If `linktrace == false` then each row of `C` will correspond to a link and a count of the number of times it appeared in the form
* C[ii, 1]: link row.
* C[ii, 2]: link column.
* C[ii, jj] where jj > 2 and jj < size(C, 2): other fields (flexible).
* C[ii, end]: frequency count of link appaearance.
"""
struct ParameterChain{G <: Integer, T <: AbstractFloat}
    C::Array{G, 2}
    nlinks::Union{Array{G, 1}, Array{G, 2}}
    pM::Array{T, 2}
    pU::Array{T, 2}
    nsteps::Int
    linktrace::Bool
end

"""
    counts2indicies(A::SparseMatrixCSC{G, T}) where {G <: Integer, T <: Integer}
    counts2indicies(A::Array{G, 2}) where G <: Integer
    counts2indicies(A::Array{G, 3}) where G <: Integer

Transform an array of counts to [rows cols counts] array storing only indicies and counts.
"""
counts2indicies(A::SparseMatrixCSC{G}) where {G <: Integer} = hcat(findnz(A)...)

function counts2indicies(A::Array{G, 2}) where G <: Integer
    rows = Int[]
    cols = Int[]
    vals = Int[]
    for jj in 1:size(A, 2)
        for ii in 1:size(A, 1)
            if !iszero(A[ii, jj])
                push!(rows, ii)
                push!(cols, jj)
                push!(vals, A[ii, jj])
            end
        end
    end
    return [rows cols vals]
end

function counts2indicies(A::Array{G, 3}) where G <: Integer
    rows = Int[]
    cols = Int[]
    stages = Int[]
    vals = Int[]
    for kk in 1:size(A, 3)
        for jj in 1:size(A, 2)
            for ii in 1:size(A, 1)
                if !iszero(A[ii, jj, kk])
                    push!(rows, ii)
                    push!(cols, jj)
                    push!(stages, kk)
                    push!(vals, A[ii, jj, kk])
                end
            end
        end
    end
    return [rows cols stages vals]
end

"""
    add_linkcounts!(pchain::ParameterChain, ctDict::DefaultDict{Tuple{G, G}, G} = DefaultDict{Tuple{Int, Int}, Int}(zero(Int)), burnin::Integer = 0) where G <: Integer

Add pairwise link counts from a `ParameterChain` to a dictionary mapping tuples for record pairs to existing counts.  Mainly an internal function for `get_linkcounts`.
"""
function add_linkcounts!(ctDict::DefaultDict{Tuple{G, G}, G}, pchain::ParameterChain, burnin::Integer = 0) where G <: Integer
        for ii in 1:size(pchain.C, 1)
            if pchain.C[ii, end] > burnin
                if pchain.C[ii, end - 1] > burnin
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2])] += pchain.C[ii, end] + one(Int) - pchain.C[ii, end - 1]
                else
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2])] += pchain.C[ii, end] - burnin
                end
            end
        end
    return ctDict
end

"""
    get_linkcounts(pchain::ParameterChain, burnin::Integer = 0)

Return pairwise link counts from a `ParameterChain` in form of [rows cols counts].  If `burnin` is set then the counts include only steps with a value strictly greater than `burnin`. 
"""
function get_linkcounts(pchain::ParameterChain, burnin::Integer = 0)
    if burnin < 0
        @error "burnin cannot be negative"
    end
    if pchain.linktrace
        ctDict = DefaultDict{Tuple{Int, Int}, Int}(zero(Int))
        for ii in 1:size(pchain.C, 1)
            if pchain.C[ii, end] > burnin
                if pchain.C[ii, end - 1] > burnin
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2])] += pchain.C[ii, end] + one(Int) - pchain.C[ii, end - 1]
                else
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2])] += pchain.C[ii, end] - burnin
                end
            end
        end
        outC = Array{Int, 2}(undef, length(ctDict), 3)
        ii = 0
        for (ky, vl) in pairs(ctDict)
            ii += 1
            outC[ii, 1] = ky[1]
            outC[ii, 2] = ky[2]
            outC[ii, 3] = vl
        end
           return outC 
    else
        if burnin > 0
            @error "burnin cannot be applied to a parameter with only stored counts"
        end
        return pchain.C
    end
end

"""
    get_groupidcounts_column(pchain::ParameterChain, colgroupid::Array{G, 1}) where G <: Integer

Count the number of occurences of a set of ids defined for each record pair for each mcmc step.

See also: [`get_groupidcounts_row`](@ref), [`get_groupidcounts_pair`](@ref)
"""
function get_groupidcounts_row(pchain::ParameterChain, rowgroupid::Array{G, 1}) where G <: Integer
    if !pchain.linktrace
        @error "pchain.linktrace must be true to extract trace counts"
    end

    if maximum(pchain.C[:, 1]) > length(rowgroupid)
        @error "Maximum value for first column of pchain.C greater than number of entries in rowgroupid"
    end

    minid, maxid = extrema(rowgroupid)
    if minid < 0
        @error "negative values observed in rowgroupid"
    elseif iszero(minid)
        @warn "zero values in rowgroupid will be ignored"
    end

    out = zeros(Int, pchain.nsteps, maxid)
    
    for ii in 1:size(pchain.C, 1)
        groupid = rowgroupid[pchain.C[ii, 1]]
        if !iszero(groupid)
            for jj in pchain.C[ii, end-1]:pchain.C[ii, end]
                out[jj, groupid] += 1
            end
        end
    end
    return out
end

"""
    get_groupidcounts_column(pchain::ParameterChain{G, T}, colgroupid::Array{G, 1}) where G <: Integer

Count the number of occurences of a set of ids defined for each record pair for each mcmc step.

See also: [`get_groupidcounts_row`](@ref), [`get_groupidcounts_pair`](@ref)
"""
function get_groupidcounts_column(pchain::ParameterChain, colgroupid::Array{G, 1}) where G <: Integer
    if !pchain.linktrace
        @error "pchain.linktrace must be true to extract trace counts"
    end

    if maximum(pchain.C[:, 2]) > length(colgroupid)
        @error "Maximum value for second column of pchain.C greater than length of colgroupid"
    end

    minid, maxid = extrema(colgroupid)
    if minid < 0
        @error "negative values observed in colgroupid"
    elseif iszero(minid)
        @warn "zero values in colgroupid will be ignored"
    end

    out = zeros(Int, pchain.nsteps, maxid)
    
    for ii in 1:size(pchain.C, 1)
        groupid = colgroupid[pchain.C[ii, 2]]
        if !iszero(groupid)
            for jj in pchain.C[ii, end-1]:pchain.C[ii, end]
                out[jj, groupid] += 1
            end
        end
    end
    return out
end

"""
    get_groupidcounts_pair(pchain::ParameterChain, pairgroupid::Union{SparseMatrixCSC{G}, Array{G, 2}}) where G <: Integer

Count the number of occurences of a set of defined for each record pair for each mcmc step.

See also: [`get_groupidcounts_row`](@ref), [`get_groupidcounts_column`](@ref)
"""
function get_groupidcounts_pair(pchain::ParameterChain, pairgroupid::Union{SparseMatrixCSC{G}, Array{G, 2}}) where G <: Integer
    if !pchain.linktrace
        @error "pchain.linktrace must be true to extract trace counts"
    end

    if maximum(pchain.C[:, 1]) > size(pairgroupid, 1)
        @error "Maximum value for first column of pchain.C greater than number of rows in pairgroupid"
    elseif maximum(pchain.C[:, 2]) > size(pairgroupid, 2)
        @error "Maximum value for second column of pchain.C greater than number of columns in pairgroupid"
    end

    if issparse(pairgroupid)
        minid, maxid = extrema(pairgroupid.nzval)
        minid = min(minid, zero(G))
    else
        minid, maxid = extrema(pairgroupid)
    end
    
    if minid < 0
        @error "negative values observed in pairgroupid"
    elseif iszero(minid)
        @warn "zero values in pairgroupid will be ignored"
    end

    out = zeros(Int, pchain.nsteps, maxid)
    
    for ii in 1:size(pchain.C, 1)
        groupid = pairgroupid[pchain.C[ii, 1], pchain.C[ii, 2]]
        if !iszero(groupid)
            for jj in pchain.C[ii, end-1]:pchain.C[ii, end]
                out[jj, groupid] += 1
            end
        end
    end
    return out
end

"""
    get_linkstagecounts(pchain::ParameterChain, burnin::Integer = 0)

Return pairwise link counts from a `ParameterChain` in form of [rows cols counts].  If `burnin` is set then the counts include only steps with a value strictly greater than `burnin`. 
"""
function get_linkstagecounts(pchain::ParameterChain, burnin::Integer = 0)
    if pchain.linktrace
        ctDict = DefaultDict{Tuple{Int, Int, Int}, Int}(zero(Int))
        for ii in 1:size(pchain.C, 1)
            if pchain.C[ii, end] > burnin
                if pchain.C[ii, end - 1] > burnin
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2], pchain.C[ii, 3])] += pchain.C[ii, end] + one(Int) - pchain.C[ii, end - 1]
                else
                    ctDict[(pchain.C[ii, 1], pchain.C[ii, 2], pchain.C[ii, 3])] += pchain.C[ii, end] - burnin
                end
            end
        end
        outC = Array{Int, 2}(undef, length(ctDict), 4)
        ii = 0
        for (ky, vl) in pairs(ctDict)
            ii += 1
            outC[ii, 1] = ky[1]
            outC[ii, 2] = ky[2]
            outC[ii, 3] = ky[3]
            outC[ii, 4] = vl
        end
        return outC 
    else
        return pchain.C
    end
end

"""
    get_steplinks(n::Integer, pchain::ParameterChain)

Return the set of rows and columns linked in step `n` as a tuple (rows, cols)
"""
function get_steplinks(n::Integer, pchain::ParameterChain)
    if !pchain.linktrace
        error("pchain.linktrace must equal true" )
    end
    keep = (pchain.C[:, (end - 1)] .<= n) .* (pchain.C[:, end] .>= n)
    return pchain.C[keep, 1], pchain.C[keep, 2]
end

"""
    get_segmentlinks(nstart::Integer, nstop::Integer, pchain::ParameterChain)

Return the set of rows and columns linked for all steps (inclusive) from nstart through nstop.
"""
function get_segmentlinks(nstart::Integer, nstop::Integer, pchain::ParameterChain)
    if !pchain.linktrace
        error("pchain.linktrace must equal true" )
    elseif nstart > nstop
        error("nstart must be <= nstop" )
    end
    keep = (pchain.C[:, (end - 1)] .<= nstart) .* (pchain.C[:, end] .>= nstop)
    return pchain.C[keep, 1], pchain.C[keep, 2]
end
