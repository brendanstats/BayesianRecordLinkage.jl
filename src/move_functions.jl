"""
    idx2pair(idx::Integer, nrow::Integer) -> (row, col)

Transform matrix index to row, column pair given number of rows.

See also: [`pair2idx`](@ref)
"""
function idx2pair(idx::Integer, nrow::Integer)
    col, row = divrem(idx, nrow)
    if iszero(row)
        row = typeof(row)(nrow)
    else
        col += one(typeof(col))
    end
    return row, col
end

"""
    pair2idx(row::G, col::G, nrow::Integer) where G <: Integer -> idx

Transform row, column pair to index of column dominante array.

See also: [`idx2pair`](@ref)
"""
function pair2idx(row::G, col::G, nrow::Integer) where G <: Integer
    return (col - one(G)) * nrow + row
end

"""
    sample_proposal_full(nrow::G, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat} -> (row, col, logweight)
    sample_proposal_full(rowlabs::Array{G, 1}, collabs::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat} -> (row, col, logweight)

Draw a weighted sample from `length(logweights)` and transform to row, col pair using idx2pair label mappings are then applied if present.

# Arguments

* `rowlabs::Array{G, 1}`: Row label or mapping applied via rowlabs[row]
* `collabs::Array{G, 1}`: Col label or mapping applied via collabs[col]
* `logweights::Array{T, 1}`: log of weights with which index should be sampled.
* `logsumweights::T`: log(sum(exp.(logweights))) computed using `logsumexp` to avoid under / overflow issues.

See also: [`idx2pair`](@ref), [`logsumexp`](@ref), [`sample_proposal_sparse`](@ref)
"""
function sample_proposal_full(nrow::G, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}
    idx = sample(Weights(exp.(logweights .- logsumweights)))
    row, col = idx2pair(idx, nrow)
    return row, col, logweights[idx]
end

function sample_proposal_full(rowlabs::Array{G, 1}, collabs::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}
    idx = sample(Weights(exp.(logweights .- logsumweights)))
    ridx, cidx = idx2pair(idx, length(rowlabs))
    return rowlabs[ridx], collabs[cidx], logweights[idx]
end

"""
    sample_proposal_sparse(rowlabs::Array{G, 1}, collabs::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}

Draw a weighted sample from `length(logweights)` and then return rowlabs[idx], collabs[idx], logweights[idx] for sampled idx.

# Arguments

* `rowlabs::Array{G, 1}`: Row label or mapping applied via rowlabs[idx]
* `collabs::Array{G, 1}`: Col label or mapping applied via collabs[idx]
* `logweights::Array{T, 1}`: log of weights with which index should be sampled.
* `logsumweights::T`: log(sum(exp.(logweights))) computed using `logsumexp` to avoid under / overflow issues.

See also: [`idx2pair`](@ref), [`logsumexp`](@ref), [`sample_proposal_full`](@ref)
"""
function sample_proposal_sparse(rowlabs::Array{G, 1}, collabs::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}
    idx = sample(Weights(exp.(logweights .- logsumweights)))
    return rowlabs[idx], collabs[idx], logweights[idx]
end


"""
    get_loglik(row::Integer, col::Integer, compsum::ComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where {T <: AbstractFloat}
    get_loglik(row::Integer, col::Integer, compsum::SparseComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where {T <: AbstractFloat}

Return loglikRatios[compsum.obsidx[row, col]] if compsum.obsidx[row, col] is non-zero otherwise return loglikMissing.  Checking for zero values only performed if a SparseComparisonSummary is supplied.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`loglik_colswitch`](@ref), [`loglik_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref)
"""
get_loglik(row::Integer, col::Integer, compsum::ComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where {T <: AbstractFloat} = loglikRatios[compsum.obsidx[row, col]]

function get_loglik(row::Integer, col::Integer, compsum::SparseComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    if iszero(compsum.obsidx[row, col])
        return loglikMissing
    else
        return loglikRatios[compsum.obsidx[row, col]]
    end
end

"""
    loglik_add(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}

Compute loglikelihood ratio for adding a link at row, col to `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`get_loglik`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`loglik_colswitch`](@ref), [`loglik_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`LinkMatrix`](@ref)
"""
function loglik_add(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""
    loglik_remove(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}

Compute loglikelihood ratio for removing the link at row, col from `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`get_loglik`](@ref), [`loglik_add`](@ref), [`loglik_rowswitch`](@ref), [`loglik_colswitch`](@ref), [`loglik_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`LinkMatrix`](@ref)
"""
function loglik_remove(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    loglikratio = -get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""
    loglik_rowswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}

Compute loglikelihood ratio for moving the link at C.col2row[col], col to row, col within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`get_loglik`](@ref), [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_colswitch`](@ref), [`loglik_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`LinkMatrix`](@ref)
"""
function loglik_rowswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing) - get_loglik(C.col2row[col], col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""
    loglik_colswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}

Compute loglikelihood ratio for moving the link at row, C.row2col[row] to row, col within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`get_loglik`](@ref), [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`loglik_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`LinkMatrix`](@ref)
"""
function loglik_colswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios) - get_loglik(row, C.row2col[row], compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""
    function loglik_doubleswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}

Compute loglikelihood ratio for moving the links at row, C.row2col[row] and C.col2row[col], col to row, col and C.col2row[col], C.row2col[row] within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`get_loglik`](@ref), [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`loglik_colswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`LinkMatrix`](@ref)
"""
function loglik_doubleswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf))  where {T <: AbstractFloat}
    rowalt = C.col2row[col]
    colalt = C.row2col[row]
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    loglikratio += get_loglik(rowalt, colalt, compsum, loglikRatios, loglikMissing)
    loglikratio -= get_loglik(row, colalt, compsum, loglikRatios, loglikMissing)
    loglikratio -= get_loglik(rowalt, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""
    get_counts(row::Integer, col::Integer, compsum::ComparisonSummary, obsidxCounts::Array{<:Integer, 2})
    get_counts(row::Integer, col::Integer, compsum::SparseComparisonSummary, obsidxCounts::Array{<:Integer, 2})

Return the increase in counts vector from adding a link at row, col.

This is stored in the idxth column of `obsidxCounts`, returns `obsidxCounts[:, compsum.obsidx[row, col]]` if `compsum.obsidx[row, col]` is non-zero.  If zero then a vector of zeros is returned.  Checking for zero values only performed if a SparseComparisonSummary is supplied.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`counts_add`](@ref), [`counts_remove`](@ref), [`counts_rowswitch`](@ref), [`counts_colswitch`](@ref), [`counts_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
get_counts(row::Integer, col::Integer, compsum::ComparisonSummary, obsidxCounts::Array{<:Integer, 2}) = obsidxCounts[:, compsum.obsidx[row, col]]


function get_counts(row::Integer, col::Integer, compsum::SparseComparisonSummary, obsidxCounts::Array{<:Integer, 2})
    if iszero(compsum.obsidx[row, col])
        return zeros(T, size(obsidxCounts, 1))
    else
        return obsidxCounts[:, compsum.obsidx[row, col]]
    end
end

"""
    counts_add(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute increase in binned comparison counts from adding a link at row, col to `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`get_counts`](@ref), [`counts_remove`](@ref), [`counts_rowswitch`](@ref), [`counts_colswitch`](@ref), [`counts_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
function counts_add(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    countsdelta = get_counts(row, col, compsum, obsidxCounts)
    return countsdelta
end

"""
    counts_remove(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute increase in binned comparison counts from removing the link at row, col from `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`get_counts`](@ref), [`counts_add`](@ref), [`counts_rowswitch`](@ref), [`counts_colswitch`](@ref), [`counts_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
function counts_remove(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    countsdelta = -get_counts(row, col, compsum, obsidxCounts)
    return countsdelta
end

"""
    counts_rowswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute increase in binned comparison counts from  moving the link at C.col2row[col], col to row, col within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`get_counts`](@ref), [`counts_add`](@ref), [`counts_remove`](@ref), [`counts_colswitch`](@ref), [`counts_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
function counts_rowswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    countsdelta = get_counts(row, col, compsum, obsidxCounts) - get_counts(C.col2row[col], col, compsum, obsidxCounts)
    return countsdelta
end

"""
    counts_colswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute increase in binned comparison counts from moving the link at row, C.row2col[row] to row, col within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`get_counts`](@ref), [`counts_add`](@ref), [`counts_remove`](@ref), [`counts_rowswitch`](@ref), [`counts_doubleswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
function counts_colswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    countsdelta = get_counts(row, col, compsum, obsidxCounts) - get_counts(row, C.row2col[row], compsum, obsidxCounts)
    return countsdelta
end

"""
    counts_doubleswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute compute increase in binned comparison counts from moving the links at row, C.row2col[row] and C.col2row[col], col to row, col and C.col2row[col], C.row2col[row] within `C`.

    counts_colswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})

Compute increase in binned comparison counts from moving the link at row, C.row2col[row] to row, col within `C`.

# Arguments

* `row::Integer`: Row of compsum.obsidx to be referrenced.
* `col::Integer`: Column of compsum.obsidx to be referrenced.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().

See also: [`get_counts`](@ref), [`counts_add`](@ref), [`counts_remove`](@ref), [`counts_rowswitch`](@ref), [`counts_colswitch`](@ref), [`ComparisonSummary`](@ref), [`SparseComparisonSummary`](@ref), [`get_obsidxcounts`](@ref)
"""
function counts_doubleswitch(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    rowalt = C.col2row[col]
    colalt = C.row2col[row]
    countsdelta = get_counts(row, col, compsum, obsidxCounts)
    countsdelta += get_counts(rowalt, colalt, compsum, obsidxCounts)
    countsdelta -= get_counts(row, colalt, compsum, obsidxCounts)
    countsdelta -= get_counts(rowalt, col, compsum, obsidxCounts)
    return countsdelta
end

"""
    logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}

Return log(P(C.nlink + 1) / P(C.nlink)) given either a vecotr of log ratios or a function.

If a function is supplied then `logpCRatio(one(G), C)` is returned.  Such a function can be
constructed from `exppenalty_logratiopn` or `betabipartite_logratiopn` with defined function
parameters. If a vector then the `logpCRatio[C.nlink + one(G)]` is returned unless
`C.nlink >= length(logpCRatio)` in which case `-T(Inf)` is returned.

# Arguments

* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios

See also: [`logpCRatios_remove`](@ref), [`exppenalty_logratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    return logpCRatio(one(G), C)
end

function logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    if C.nlink >= length(logpCRatio)
        return -T(Inf)
    else
        return logpCRatio[C.nlink + one(G)]
    end
end

"""
    logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}

Compute log(prior(C')) - log(prior(C)) where C' has one more link than C.

If a function is supplied then `logpCRatio(-one(G), C)` is returned.  Such a function can be
constructed from `exppenalty_logratiopn` or `betabipartite_logratiopn` with defined function
parameters. If a vector then the `logpCRatio[C.nlink - one(G)]` is returned unless
`iszero(C.nlink)` in which case `-T(Inf)` is returned.

# Arguments

* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios

See also: [`logpCRatios_add`](@ref), [`exppenalty_logratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    return logpCRatio(-one(G), C)
end

function logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    if iszero(C.nlink)
        return -T(Inf)
    else
        return -logpCRatio[C.nlink]
    end
end

"""
    randomwalk1_move!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Performes an update to C where a link at (row, col) is added.

Updates are as follows:
* If C.row2col[row] == 0 and C.col2row[col] == 0 then a link is added at (row, col)
* If C.row2col[row] == 0 and C.col2row[col] != 0 then the link at C.col2row[col], col is moved to row, col
* If C.row2col[row] != 0 and C.col2row[col] == 0 then the link at row, C.row2col[row] is moved to row, col
* If C.row2col[row] != 0 and C.col2row[col] != 0 and C.row2col[row] == col then the link at row, col is deleted.
* If C.row2col[row] != 0 and C.col2row[col] != 0 and C.row2col[row] != col then the links at row, C.row2col[row] and C.col2row[col], col are move to row, col and C.col2row[col], C.row2col[row]

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.

See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link!`](@ref), [`randomwalk1_inverse`](@ref)
"""
function randomwalk1_move!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            add_link!(row, col, C)
        else ##single switch move I
            rowswitch_link!(row, col, C)
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            colswitch_link!(row, col, C)
        elseif C.col2row[col] == row ##delete move
            remove_link!(row, col, C)
        else ##double switch move
            doubleswitch_link!(row, col, C)
        end
    end
    return C
end

"""
    randomwalk1_inverse(row::G, col::G, C::LinkMatrix{G}) where G <: Integer -> invrow, invcol

Find the inverse move for a randomwalk1 move.

Inverse is defined as after calling `randomwalk1_move!` but determined beforehand. So calling
`randomwalk1_move!` and then calling `randomwalk1_move!` again on the inverse move computed
before the first call will return the original `LinkMatrix`.

```julia
C0 = deepcopy(C)
invrow, invcol = randomwalk1_inverse(row, col, C)
randomwalk1_move!(row, col, C)
randomwalk1_move!(invrow, invcol, C)
C == C0
```

See also: [`randomwalk1_move!`](@ref), [`randomwalk1_update!`](@ref), [`randomwalk1_locally_balanced_update!`](@ref)
"""
function randomwalk1_inverse(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(C.row2col[row])
        if iszero(C.col2row[col])
            return row, col
        else
            return C.col2row[col], col
        end
    elseif iszero(C.col2row[col]) #row not zero, col zero
        return  row, C.row2col[row]
    elseif C.row2col[row] == col
        return row, col
    else
        return row, C.row2col[row] #could also do C.col2row[col], col
    end
end

"""
    randomwalk1_loglikpCratio(row::Integer, col::Integer, C::LinkMatrix,
                              compsum::Union{ComparisonSummary, SparseComparisonSummary},
                              logpCRatioAdd::T, logpCRatioRemove::T,
                              loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where T <: AbstractFloat -> loglikratio, doubleswitch

Return log(likelihood(C') * prior(C') / (likelihood(C) * prior(C))) where C' = `randomwalk1_move!(row, col, C)`.

Returns a log likelihood ratio as well as a bool indicating if the move is a doubleswitch move (true).

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `logpCRatioAdd::T`: Log prior ratio if C' contains one additional link typically calculated with `logpCRatios_add`.
* `logpCRatioRemove::T`: Log prior ratio if C' contains one fewer links typically calculated with `logpCRatios_remove`.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: See also: [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`loglik_colswitch`](@ref), [`loglik_doubleswitch`](@ref), [`logpCRatios_add`](@ref), [`logpCRatios_remove`](@ref)
"""
function randomwalk1_loglikpCratio(row::Integer, col::Integer, C::LinkMatrix,
                                   compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                   logpCRatioAdd::T, logpCRatioRemove::T,
                                   loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where T <: AbstractFloat
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatioAdd, false
        else ##single switch move I
            return loglik_rowswitch(row, col, C, compsum, loglikRatios, loglikMissing), false
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return loglik_colswitch(row, col, C, compsum, loglikRatios, loglikMissing), false
        elseif C.col2row[col] == row ##delete move
            return loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatioRemove, false
        else ##double switch move
            return loglik_doubleswitch(row, col, C, compsum, loglikRatios, loglikMissing), true
        end
    end
end

"""
    randomwalk1_countsdelta(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2}) -> countdelta

Compute the change in binned comparison vector counts for links in `C` from performing `randomwalk1_move!(row, col, C)`

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
"""
function randomwalk1_countsdelta(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return counts_add(row, col, C, compsum, obsidxCounts)
        else ##single switch move I
            return counts_rowswitch(row, col, C, compsum, obsidxCounts)
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return counts_colswitch(row, col, C, compsum, obsidxCounts)
        elseif C.col2row[col] == row ##delete move
            return counts_remove(row, col, C, compsum, obsidxCounts)
        else ##double switch move
            return counts_doubleswitch(row, col, C, compsum, obsidxCounts)
        end
    end
end

"""
    randomwalk1_log_movecount(nrow::Integer, ncol::Integer, nlink::Integer)

Log of the number of distinct moves possible from randomwalk1.  This is equal to `nrow * ncol`
minus the number of distinct doubleswitch moves which can be sampled in two ways.
"""
function randomwalk1_log_movecount(nrow::Integer, ncol::Integer, nlink::G) where G <: Integer
    return log(nrow * ncol - div(nlink * (nlink - one(G)), G(2)))
end

"""
    randomwalk1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                        C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                        loglikRatios::Array{T, 1}, obsidxCounts::Array{A, 2},
                        logpCRatio::Union{Function, Array{T, 1}}, log_balance_function::Function = identity,
                        loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Integer} -> C, countsdelta, update

Perform an update to `C` according to randomwalk1 drawn uniformly from the set of possible moves.

Returns an updated version of C, a change in the binned comparison counts, and a boolean
variable indicating if `C` has changed.  Update is drawn uniformly from the cartesian product
of `rows` and `cols`.  This double weights doubleswitch moves, to account for this doubleswitch
moves are retained with p = 0.5 so make the sampling uniform.  A standard metropolis-hastings
accept / reject update is made based on likelihood, prior probability, and move counts.  Despite
uniform sampling the number of unique potential moves can change with the update (since
doubleswitch moves are double counted) therefore the sampling kernel density is accounted for.

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `log_balance_function::Function`: Balancing function used for locally balanced move.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`randomwalk1_move!`](@ref), [`randomwalk2_update!`](@ref), [`randomwalk1_countsdelta`](@ref), [`randomwalk1_log_movecount`](@ref)
"""
function randomwalk1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                             C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                             loglikRatios::Array{T, 1}, obsidxCounts::Array{A, 2},
                             logpCRatio::Union{Function, Array{T, 1}}, log_balance_function::Function = identity,
                             loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Integer}
    
    row = sample(rows)
    col = sample(cols)

    #Resample if missing recordpair sampled
    if iszero(compsum.obsidx[row, col])
        return randomwalk1_update!(rows, cols, C, compsum, obsidxCounts, loglikRatios, logpCRatio, loglikMissing)
    end
    
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            loglik = loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_add(C, logpCRatio)
            loglik += randomwalk1_log_movecount(length(rows), length(cols), C.nlink) - randomwalk1_log_movecount(length(rows), length(cols), C.nlink + one(G))
        else ##single switch move I
            loglik = loglik_rowswitch(row, col, C, compsum, loglikRatios, loglikMissing)
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            loglik = loglik_colswitch(row, col, C, compsum, loglikRatios, loglikMissing)
        elseif C.col2row[col] == row ##delete move
            loglik = loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
            loglik += randomwalk1_log_movecount(length(rows), length(cols), C.nlink) - randomwalk1_log_movecount(length(rows), length(cols), C.nlink - one(G))
        else ##double switch move
            #only sample double switches with p = 0.5 to make sampling uniform over moves
            if rand() < 0.5
                return randomwalk1_update!(rows, cols, C, compsum, obsidxCounts, loglikRatios, logpCRatio, loglikMissing)
            end
            loglik = loglik_doubleswitch(row, col, C, compsum, loglikRatios, loglikMissing)
        end
    end

    if rand() < exp(loglik)
        countsdelta = randomwalk1_countsdelta(row, col, C, compsum, obsidxCounts)
        C = randomwalk1_move!(row, col, C)
        return C, countsdelta, true
    else
        return C, zeros(A, size(obsidxCounts, 1)), false
    end
end

"""
    randomwalk1_log_move_weights(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                 C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 logpCRatioAdd::T, logpCRatioRemove::T,
                                 loglikRatios::Array{T, 1}, log_balance_function::Function = identity,
                                 loglikMissing::T = -T(Inf)) where T <: AbstractFloat -> logweights, logsum(logweights)

Compute (unnormalized) logged locally balanced kernel sampling weights.

# Arguments

* `rows::Array{<:Integer, 1}`: rows defining subregion of C to update, typically correspond to a block of a `PosthocBlock`.
* `cols::Array{<:Integer, 1}`: columns defining subregion of C to update, typically correspond to a block of a `PosthocBlock`.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `logpCRatioAdd::T`: Log prior ratio if C' contains one additional link typically calculated with `logpCRatios_add`.
* `logpCRatioRemove::T`: Log prior ratio if C' contains one fewer links typically calculated with `logpCRatios_remove`.
* `log_balance_function::Function`: Balancing function used for locally balanced move.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`randomwalk1_loglikpCratio`](@ref), [`randomwalk1_log_move_weights_sparse`](@ref)
"""
function randomwalk1_log_move_weights(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                      C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                      logpCRatioAdd::T, logpCRatioRemove::T,
                                      loglikRatios::Array{T, 1}, log_balance_function::Function = identity,
                                      loglikMissing::T = -T(Inf)) where T <: AbstractFloat
    
    lmoveweights = zeros(T, length(cols) * length(rows))
    idx = 1
    for jj in 1:length(cols), ii in 1:length(rows)
        lp, doubleswitch = randomwalk1_loglikpCratio(rows[ii], cols[jj], C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, loglikMissing)
        if doubleswitch
            lmoveweights[idx] = log_balance_function(lp) + loghalf
        else
            lmoveweights[idx] = log_balance_function(lp)
        end
        idx += 1
    end
    lsummw = logsumexp(lmoveweights)
    return lmoveweights, lsummw
end

"""
    randomwalk1_log_move_weights_sparse(cols::Array{<:Integer, 1}, nzobs::Integer,
                                        C::LinkMatrix, compsum::SparseComparisonSummary,
                                        logpCRatioAdd::T, logpCRatioRemove::T,
                                        loglikRatios::Array{T, 1}, log_balance_function::Function = identity,
                                        loglikMissing::T = -T(Inf)) where T <: AbstractFloat

Compute (unnormalized) logged locally balanced kernel sampling weights.

Differs from in that `randomwalk1_log_move_weights` in that sparsity is used to reduce the
number of moves which must be considered.  All entries in the sparse matrix for which the
column is contained in `cols` are considered in the set of moves consisered.  To limit
`compsum` to contain only entries within `PosthocBlocks` see `dropoutside!` or `dropoutside`.

# Arguments

* `cols::Array{<:Integer, 1}`: columns defining subregion of C to update, typically correspond to a block of a `PosthocBlock`.
* `nzobs::Integer` Number of non-zero entries
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum:: SparseComparisonSummary`: Summary of record pair comparisons used in model fitting.
* `logpCRatioAdd::T`: Log prior ratio if C' contains one additional link typically calculated with `logpCRatios_add`.
* `logpCRatioRemove::T`: Log prior ratio if C' contains one fewer links typically calculated with `logpCRatios_remove`.
* `log_balance_function::Function`: Balancing function used for locally balanced move.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`randomwalk1_loglikpCratio`](@ref), [`randomwalk1_log_move_weights`](@ref)
"""
function randomwalk1_log_move_weights_sparse(cols::Array{<:Integer, 1}, nzobs::Integer,
                                             C::LinkMatrix, compsum::SparseComparisonSummary,
                                             logpCRatioAdd::T, logpCRatioRemove::T,
                                             loglikRatios::Array{T, 1}, log_balance_function::Function = identity,
                                             loglikMissing::T = -T(Inf)) where T <: AbstractFloat
    
    rows = rowvals(compsum.obsidx)

    obsrows = zeros(eltype(rows), nzobs)
    obscols = zeros(eltype(cols), nzobs)
    lmoveweights = zeros(T, nzobs)
    idx = 1
    for col in cols
        for ii in nzrange(compsum.obsidx, col)
            obsrows[idx] = rows[ii]
            obscols[idx] = col
            lp, doubleswitch = randomwalk1_loglikpCratio(rows[ii], cols[jj], C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, loglikMissing)
            if doubleswitch
                lmoveweights[idx] = log_balance_function(lp) + loghalf
            else
                lmoveweights[idx] = log_balance_function(lp)
            end
            idx += 1
        end
    end
    lsummw = logsumexp(lmoveweights)
    return obsrows, obscols, lmoveweights, lsummw
end



"""
    randomwalk1_locally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                         C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                         loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                         logpCRatio::Union{Function, Array{T, 1}}, log_balance_function::Function = identity,
                                         loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Perform locally balanced (informed) Metropolis-Hastings update to `C`. Calculations performed in logspace to avoid under/over flow issues.

Update design based on Zanella (2019) "Informed Proposals for Local MCMC in Discrete Spaces".  

# Arguments

* `rows::Array{<:Integer, 1}`: rows defining subregion of C to update, typically correspond to a block of a `PosthocBlock`.
* `cols::Array{<:Integer, 1}`: columns defining subregion of C to update, typically correspond to a block of a `PosthocBlock`.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `log_balance_function::Function`: Balancing function used for locally balanced move.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also [`randomwalk1_log_move_weights`](@ref), [`sample_proposal_full`](@ref), [`randomwalk1_move!`](@ref), [`randomwalk1_countsdelta`](@ref)
"""
function randomwalk1_locally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                              C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                              loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                              logpCRatio::Union{Function, Array{T, 1}}, log_balance_function::Function = identity,
                                              loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

    ##Move weights
    logpCRatioAdd = logpCRatios_add(C, logpCRatio)
    logpCRatioRemove = logpCRatios_remove(C, logpCRatio)  
    lmoveweights, lsummw = randomwalk1_log_move_weights(rows, cols, C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, log_balance_function, loglikMissing)
    
    ##Sample move
    moverow, movecol, lmove = sample_proposal_full(rows, cols, lmoveweights, lsummw)
    invrow, invcol = randomwalk1_inverse(moverow, movecol, C)
    loglikpCratio, doubleswitch = randomwalk1_loglikpCratio(moverow, movecol, C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, loglikMissing)

    #check that empty move has not been sampled
    if iszero(compsum.obsidx[moverow, movecol])
        @warn "Missing record pair linked ($moverow, $movecol), debugging needed..."
    end
    
    #Assume move to compute reverse weights
    randomwalk1_move!(moverow, movecol, C)
    loginvpCRatioAdd = logpCRatios_add(C, logpCRatio)
    loginvpCRatioRemove = logpCRatios_remove(C, logpCRatio)
    
    ##Proposal weights and reverse move
    linverseweights, lsuminvw = randomwalk1_log_move_weights(rows, cols, C, compsum, loginvpCRatioAdd, loginvpCRatioRemove, loglikRatios, log_balance_function, loglikMissing)

    if doubleswitch
        linvmove = log_balance_function(-loglikpCratio) + loghalf
    else
        linvmove = log_balance_function(-loglikpCratio)
    end
    
    ##Compute ratio
    lmoveratio = linvmove - lmove + lsummw - lsuminvw

    #also return change in likelihood
    if rand() < exp(loglikpCratio + lmoveratio)
        countsdelta = -randomwalk1_countsdelta(invrow, invcol, C, compsum, obsidxCounts)
        return C, countsdelta, true
    else
        randomwalk1_move!(invrow, invcol, C)
        countsdelta = zeros(G, size(obsidxCounts, 1))
        return C, countsdelta, false
    end
end

"""
    randomwalk1_globally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                          C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                          loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                          logpCRatio::Union{Function, Array{T, 1}},
                                          loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Wrapper around `randomwalk1_locally_balanced_update` using the identity balancing function to perform globally balanced updates on `C`.

See also: [`randomwalk1_locally_balanced_update!`](@ref)
"""
function randomwalk1_globally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                               C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                               loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                               logpCRatio::Union{Function, Array{T, 1}},
                                               loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, identity, loglikMissing)
end

"""
    randomwalk1_locally_balanced_sqrt_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                              C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                              loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                              logpCRatio::Union{Function, Array{T, 1}},
                                              loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Wrapper around `randomwalk1_locally_balanced_update` using a sqrt balancing function to perform locally balanced updates on `C`.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`sqrt`](@ref), [`randomwalk1_locally_balanced_barker_update!`](@ref)
"""
function randomwalk1_locally_balanced_sqrt_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                   C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                   loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                                   logpCRatio::Union{Function, Array{T, 1}},
                                                   loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lsqrt, loglikMissing)
end

"""
    randomwalk1_locally_balanced_barker_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                                logpCRatio::Union{Function, Array{T, 1}},
                                                loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Wrapper around `randomwalk1_locally_balanced_update` using the balancing function `t / (t + 1)` (barker)to perform locally balanced updates on `C`.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`lbarker`](@ref), [`randomwalk1_locally_balanced_sqrt_update!`](@ref)
"""
function randomwalk1_locally_balanced_barker_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                     C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                     loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                                     logpCRatio::Union{Function, Array{T, 1}},
                                                     loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lbarker, loglikMissing)
end

"""
    randomwalk1_locally_balanced_min1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                              C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                              loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                              logpCRatio::Union{Function, Array{T, 1}},
                                              loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Wrapper around `randomwalk1_locally_balanced_update` using the balancing function `min(t, 1)` to perform locally balanced updates on `C`.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`lmin1`](@ref), [`randomwalk1_locally_balanced_max1_update!`](@ref)
"""
function randomwalk1_locally_balanced_min1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                   C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                   loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                                   logpCRatio::Union{Function, Array{T, 1}},
                                                   loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lmin1, loglikMissing)
end

"""
    randomwalk1_locally_balanced_max1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                              C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                              loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                              logpCRatio::Union{Function, Array{T, 1}},
                                              loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Wrapper around `randomwalk1_locally_balanced_update` using the balancing function `max(t, 1)` to perform locally balanced updates on `C`.

See also: [`randomwalk1_locally_balanced_update!`](@ref), [`lmax1`](@ref), [`randomwalk1_locally_balanced_min1_update!`](@ref)
"""
function randomwalk1_locally_balanced_max1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                   C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                   loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                                                   logpCRatio::Union{Function, Array{T, 1}},
                                                   loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lmax1, loglikMissing)
end

"""
    randomwalk2_move!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Performes an update to C where a link at (row, col) is added.

Updates are as follows:
* If C.row2col[row] == 0 and C.col2row[col] == 0 then a link is added at (row, col)
* If C.row2col[row] == 0 and C.col2row[col] != 0 - should not be sampled under randomwalk2 and a warning is returned
* If C.row2col[row] != 0 and C.col2row[col] == 0 then the link at row, C.row2col[row] is moved to row, col
* If C.row2col[row] != 0 and C.col2row[col] != 0 and C.row2col[row] == col then the link at row, col is deleted.
* If C.row2col[row] != 0 and C.col2row[col] != 0 and C.row2col[row] != col - should not be sampled under randomwalk2 and a warning is returned

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.

See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`randomwalk2_inverse`](@ref)
"""
function randomwalk2_move!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            add_link!(row, col, C)
        else
            @warn "incorrect draw for randomwalk 2"
        end
    else
        if iszero(C.col2row[col]) ##switch move
            colswitch_link!(row, col, C)
        elseif C.col2row[col] == row ##delete move
            remove_link!(row, col, C)
        else ##double switch move
            @warn "incorrect draw for randomwalk 2 (double switch)"
        end
    end
    
    return C
end

"""
    randomwalk2_inverse(row::G, col::G, C::LinkMatrix{G}) where G <: Integer -> invrow, invcol

Find the inverse move for a randomwalk2 move.

Inverse is defined as after calling `randomwalk2_move!` but determined beforehand. So calling
`randomwalk1_move!` and then calling `randomwalk2_move!` again on the inverse move computed
before the first call will return the original `LinkMatrix`.

```julia
C0 = deepcopy(C)
invrow, invcol = randomwalk2_inverse(row, col, C)
randomwalk2_move!(row, col, C)
randomwalk2_move!(invrow, invcol, C)
C == C0
```

See also: [`randomwalk2_move!`](@ref), [`randomwalk2_update!`](@ref), [`randomwalk2_locally_balanced_update!`](@ref)
"""
function randomwalk2_inverse(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(C.row2col[row])
        if iszero(C.col2row[col])
            return row, col
        else
            @warn "incorrect draw for randomwalk 2"
        end
    elseif iszero(C.col2row[col]) #row not zero, col zero
        return  row, C.row2col[row]
    elseif C.row2col[row] == col
        return row, col
    else
        @warn "incorrect draw for randomwalk 2 (double switch)"
    end
end

"""
    randomwalk2_loglikpCratio(row::Integer, col::Integer, C::LinkMatrix,
                              compsum::Union{ComparisonSummary, SparseComparisonSummary},
                              logpCRatioAdd::T, logpCRatioRemove::T,
                              loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where T <: AbstractFloat -> loglikratio, doubleswitch

Return log(likelihood(C') * prior(C') / (likelihood(C) * prior(C))) where C' = `randomwalk1_move!(row, col, C)`.

Returns a log likelihood ratio as well as a bool indicating if the move is a doubleswitch move (true).

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `logpCRatioAdd::T`: Log prior ratio if C' contains one additional link typically calculated with `logpCRatios_add`.
* `logpCRatioRemove::T`: Log prior ratio if C' contains one fewer links typically calculated with `logpCRatios_remove`.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: See also: [`loglik_add`](@ref), [`loglik_remove`](@ref), [`loglik_rowswitch`](@ref), [`logpCRatios_add`](@ref), [`logpCRatios_remove`](@ref)
"""
function randomwalk2_loglikpCratio(row::T, col::T, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, logpCRatioAdd::T, logpCRatioRemove::T, loglikRatios::Array{T, 1}, loglikMissing::T = -T(Inf)) where T <: Integer
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatioAdd, false
        else ##single switch move I
            @warn "incorrect draw for randomwalk 2"
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return loglik_colswitch(row, col, C, compsum, loglikRatios, loglikMissing), false
        elseif C.col2row[col] == row ##delete move
            return loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatioRemove, false
        else ##double switch move
            @warn "incorrect draw for randomwalk 2"
        end
    end
end


"""
    randomwalk2_countsdelta(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{<:Integer, 2}) -> countdelta

Compute the change in binned comparison vector counts for links in `C` from performing `randomwalk2_move!(row, col, C)`

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
"""
function randomwalk2_countsdelta(row::T, col::T,
                                 C::LinkMatrix{T},
                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 obsidxCounts::Array{G, 2}) where {T <: Integer, G <: Integer}
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return get_counts(row, col, compsum, obsidxCounts)
        else ##single switch move I
            @warn "incorrect draw for randomwalk 2"
            return countsdelta
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return get_counts(row, col, compsum, obsidxCounts) - get_counts(row, C.row2col[row], compsum, obsidxCounts)
        elseif C.col2row[col] == row ##delete move
            return -get_counts(row, col, compsum, obsidxCounts)
        else ##double switch move
            @warn "incorrect draw for randomwalk 2 (double switch)"
            return zeros(G, size(obsidxCounts, 1))
        end
    end
end

"""
    randomwalk2_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                        C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                        loglikRatios::Array{T, 1}, obsidxCounts::Array{A, 2},
                        logpCRatio::Union{Function, Array{T, 1}}, p::AbstractFloat = 0.5,
                        loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Integer} -> C, countsdelta, update

Perform an update to `C` according to randomwalk2 drawn uniformly from the set of possible moves.

Returns an updated version of C, a change in the binned comparison counts, and a boolean
variable indicating if `C` has changed.  Should only be used if `length(rows) >= length(cols)`.
Update is drawn as follows:

1. A row is sampled uniformly
2. Column is selected:
    * If the row is unlinked then a column is sampled uniformly from the unlinked columns
    * If the row is linked:
        * If there are unlinked columns then the link containing row is removed
        * If there are no unlinked columns:
            * With probability p the link is removed
            * With probability 1 - p the link is moved to (row, newcol) where newcol is drawn uniformly from the unlinked columns.

A standard metropolis-hastings accept / reject update is made based on likelihood, prior
probability, and move counts.

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `p::AbstractFloat`: probability of link removal vs. move.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`randomwalk2_move!`](@ref), [`randomwalk1_update!`](@ref), [`randomwalk2_countsdelta`](@ref), [`randomwalk2_log_movecount`](@ref)
"""
function randomwalk2_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                             C::LinkMatrix,
                             compsum::Union{ComparisonSummary, SparseComparisonSummary},
                             loglikRatios::Array{T, 1}, obsidxCounts::Array{A, 2},
                             logpCRatio::Union{Function, Array{T, 1}}, p::AbstractFloat = 0.5,
                             loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Integer}

    row = sample(rows)
    opencols = cols[findall(iszero, C.col2row[cols])]
    
    if iszero(C.row2col[row]) #add move
        if length(opencols) == 0 #this should only happen if ncol < nrow in which case a different kernel should be used
            return randomwalk2_update!(rows, cols, C, compsum, obsidxCounts, loklikMargin, logpCRatio, p, loglikMissing)
        end
        col = sample(opencols)
        loglik = loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_add(C, logpCRatio)
        if length(opencols) > 1
            loglik += log(p) + log(length(opencols))
        end
    elseif length(opencols) == 0
        col = C.row2col[row]
        loglik = loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
        #lmoveratio = zero(T)
    elseif rand() < p  #remove move
        col = C.row2col[row]
        loglik = loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
        loglik += -log(length(opencols) + 1) - log(p)
    else
        col = sample(opencols)
        loglik = loglik_colswitch(row, col, C, compsum, loglikRatios, loglikMissing)
        #lmoveratio = zero(T)
    end
    
    if rand() < exp(loglik)
        countsdelta = randomwalk2_countsdelta(row, col, C, compsum, obsidxCounts)
        C = randomwalk2_move!(row, col, C)
        return C, countsdelta, true
    else
        return C, zeros(A, size(obsidxCounts, 1)), false
    end
end

"""
    singleton_gibbs!(row::Integer, col::Integer, C::LinkMatrix,
                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                     loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                     logpCRatio::Union{Function, Array{T, 1}}, 
                     loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Perform a gibbs update on the link structure.

Operates on a single row, column pair sampling between the two options (link and non-link).
Returns an update `C`, the change in the matched binned comparison vector counts, and a
boolean variable indicating in the sampled structure was the same as the input or not.

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair  comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`randomwalk1_update!`](@ref), [`randomwalk2_update!`](@ref), [`randomwalk1_locally_balanced_update!`](@ref), [`singlerow_gibbs!`](@ref), [`singlecol_gibbs!`](@ref), [`gibbs_MU_draw`](@ref)
"""
function singleton_gibbs!(row::Integer, col::Integer, C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                          logpCRatio::Union{Function, Array{T, 1}},
                          loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    #logaddexp
    if iszero(C.row2col[row])
        loglik = loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_add(C, logpCRatio)
        if rand() < exp(loglik - logaddexp(loglik, 1.0))
            countsdelta = counts_add(row, col, C, compsum, obsidxCounts)
            add_link!(row, col, C)
            move = true
        else
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    else
        loglik = loglik_remove(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
        if rand() < exp(loglik - logaddexp(loglik, 1.0))
            countsdelta = counts_remove(row, col, C, compsum, obsidxCounts)
            remove_link!(row, col, C)
            move = true
        else
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    end
    return C, countsdelta, move
end

"""
    singlerow_gibbs!(row::Integer, cols::Array{<:Integer, 1}, C::LinkMatrix,
                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                     loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                     logpCRatio::Union{Function, Array{T, 1}},
                     loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Perform a gibbs update on a subregion of `C` consisting of a single row and multiple columns.

# Arguments

* `row::Integer`: row of C which link will be added to or removed from.
* `cols::Array{<:Integer, 1}`: columns of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair  comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`singleton_gibbs!`](@ref), [`singlecol_gibbs!`](@ref), [`randomwalk1_update!`](@ref), [`randomwalk2_update!`](@ref), [`randomwalk1_locally_balanced_update!`](@ref), [`gibbs_MU_draw`](@ref)
"""
function singlerow_gibbs!(row::Integer, cols::Array{<:Integer, 1}, C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                          logpCRatio::Union{Function, Array{T, 1}},
                          loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

    logweights = zeros(T, length(cols) + 1) #last entry corresponds to unlinked
    if iszero(C.row2col[row])
        for ii in 1:length(cols)
            logweights[ii] = loglik_add(row, cols[ii], C, compsum, loglikRatios, loglikMissing)
        end
        logweights[end] = -logpCRatios_add(C, logpCRatio) #subtracting from unlinked entry equivalent to adding to all linked entries

        softmax!(logweights) #these are now unlogged
        linkidx = sample(Weights(logweights))

        if linkidx <= length(cols) #add link
            col = cols[linkidx]
            countsdelta = counts_add(row, col, C, compsum, obsidxCounts)
            add_link!(row, col, C)
            move = true
        else #no addition
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    else
        for ii in 1:length(cols)
            if cols[ii] != C.row2col[row]
                logweights[ii] = loglik_rowswitch(row, cols[ii], C, compsum, loglikRatios, loglikMissing)
            else
                logweights[end] = loglik_remove(row, cols[ii], C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
            end
        end
        
        softmax!(logweights) #these are now unlogged
        linkidx = sample(Weights(logweights))

        if linkidx > length(cols) #remove link
            col = C.row2col[row]
            countsdelta = counts_remove(row, col, C, compsum, obsidxCounts)
            remove_link!(row, col, C)
            move = true
        elseif cols[linkidx] != C.row2col[row] #rowswitch move
            col = cols[linkidx]
            countsdelta = counts_rowswitch(row, col, C, compsum, obsidxCounts)
            rowswitch_link!(row, col, C)
            move = true
        else #no move
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    end
        
    return C, countsdelta, move
end

"""
    singlecol_gibbs!(rows::Array{<:Integer, 1}, cols::Integer, C::LinkMatrix,
                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                     loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                     logpCRatio::Union{Function, Array{T, 1}},
                     loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

Perform a gibbs update on a subregion of `C` consisting of a multiple rows and a single column.

# Arguments

* `rows::Array{<:Integer, 1}`: rows of C which link will be added to or removed from.
* `col::Integer`: column of C which link will be added to or removed from.
* `C::LinkMatrix`: Parameter containing current state of linkage structure.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair  comparisons used in model fitting.
* `loglikRatios::Array{<:AbstractFloat, 1}` loglikelihood ratio for each comparison vector index.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also: [`singleton_gibbs!`](@ref), [`singlecol_gibbs!`](@ref), [`randomwalk1_update!`](@ref), [`randomwalk2_update!`](@ref), [`randomwalk1_locally_balanced_update!`](@ref), [`gibbs_MU_draw`](@ref)
"""
function singlecol_gibbs!(rows::Array{<:Integer, 1}, col::Integer, C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                          logpCRatio::Union{Function, Array{T, 1}},
                          loglikMissing::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

    logweights = zeros(T, length(rows) + 1) #last entry corresponds to unlinked
    if iszero(C.col2row[col])
        for ii in 1:length(rows)
            logweights[ii] = loglik_add(rows[ii], col, C, compsum, loglikRatios, loglikMissing)
        end
        logweights[end] = -logpCRatios_add(C, logpCRatio) #subtracting from unlinked entry equivalent to adding to all linked entries

        softmax!(logweights) #these are now unlogged
        linkidx = sample(Weights(logweights))

        if linkidx <= length(rows) #add link
            row = rows[linkidx]
            countsdelta = counts_add(row, col, C, compsum, obsidxCounts)
            add_link!(row, col, C)
            move = true
        else #no addition
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    else
        for ii in 1:length(rows)
            if rows[ii] != C.col2row[col]
                logweights[ii] = loglik_rowswitch(rows[ii], col, C, compsum, loglikRatios, loglikMissing)
            else
                logweights[end] = loglik_remove(rows[ii], col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_remove(C, logpCRatio)
            end
        end
        
        softmax!(logweights) #these are now unlogged
        linkidx = sample(Weights(logweights))

        if linkidx > length(rows) #remove link
            col = C.row2col[row]
            countsdelta = counts_remove(row, col, C, compsum, obsidxCounts)
            remove_link!(row, col, C)
            move = true
        elseif cols[linkidx] != C.row2col[row] #rowswitch move
            row = rows[linkidx]
            countsdelta = counts_rowswitch(row, col, C, compsum, obsidxCounts)
            rowswitch_link!(row, col, C)
            move = true
        else #no move
            countsdelta = zeros(G, size(obsidxCounts, 1))
            move = false
        end
    end

    return C, countsdelta, move
end
    
"""
    dirichlet_draw(matchcounts::Array{<:Integer, 1},
                   compsum::Union{ComparisonSummary, SparseComparisonSummary},
                   priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                   priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))

Compute the conditional dirchlet distribution for updating M and U parameters.

Takes as observed multinomial counts for the M distribution `matchcounts + priorM` and
`compsum.counts - matchcounts + priorU`.  These counts are then partitoned into draws
for each individual comparison variable.  The resulting draws are then appended together
so that the returned format is a tuple containing two vectors.

# Arguments

* `matchcounts::Array{<:Integer, 1}`: Binned comparions vector counts such as that returned by the `counts_matches` function.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair  comparisons used in model fitting.
* `priorM::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `priorU::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.

See also: [`counts_matches`](@ref), [`gibbs_MU_draw`](@ref)
"""
function dirichlet_draw(matchcounts::Array{<:Integer, 1},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary},
                        priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                        priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))
    nonmatchcounts = compsum.counts - matchcounts
    paramM = matchcounts + priorM
    paramU = nonmatchcounts + priorU

    pM = Array{Float64}(undef, length(priorM))
    pU = Array{Float64}(undef, length(priorU))
    
    startidx = 1
    for ii in 1:length(compsum.nlevels)
        rng = range(startidx, length = compsum.nlevels[ii])
        startidx += compsum.nlevels[ii]
        pM[rng] = rand(Dirichlet(paramM[rng]))
        pU[rng] = rand(Dirichlet(paramU[rng]))
    end
    return pM, pU
end

"""
    gibbs_MU_draw(matchcounts::Array{<:Integer, 1},
                  compsum::Union{ComparisonSummary, SparseComparisonSummary},
                  obsidxCounts::Array{<:Integer, 2} = counts_delta(compsum),
                  priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                  priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))

# Arguments

* `matchcounts::Array{<:Integer, 1}`: Binned comparions vector counts such as that returned by the `counts_matches` function.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair  comparisons used in model fitting.
* `obsidxCounts::Array{<:Integer, 1}`: Mapping from compsum.obsidx to compsum.counts as generated by get_obsidxcounts().
* `priorM::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `priorU::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.

See also: [`counts_matches`](@ref), [`gibbs_MU_draw`](@ref)
"""
function gibbs_MU_draw(matchcounts::Array{<:Integer, 1},
                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                       obsidxCounts::Array{<:Integer, 2} = get_obsidxcounts(compsum),
                       priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                       priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikRatios = obsidxCounts' * logDiff
    return pM, pU, loglikRatios
end
