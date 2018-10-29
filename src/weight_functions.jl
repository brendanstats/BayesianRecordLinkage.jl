"""
Find the smallest margin between sorted entries in a vector
"""
function minimum_margin(x::Array{<:Real, 1})
    s = sort(unique(round.(x, 10)))
    margin = minimum(s[2:end] - s[1:end-1])
    return margin
end

"""
    count_matches(matchedrows, matchedcols, comparisonSummary) -> matchcounts, matchobs

Returns vectors comparable to comparisonSummary.counts and comparisonSummary.obsct
corresponding to 
"""
function counts_matches(mrows::Array{G, 1},
                        mcols::Array{G, 1},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer
    
    #count occurences of each observation in obsvecs
    matchvecct = zeros(Int64, length(compsum.obsvecct))
    for (ii, jj) in zip(mrows, mcols)
        matchvecct[compsum.obsidx[ii, jj]] += 1
    end

    #map observation occurences to counts
    matchcounts = zeros(Int64, length(compsum.counts))
    matchobs = zeros(Int64, compsum.ncomp)
    for (jj, ct) in enumerate(IndexLinear(), matchvecct)
        if ct > 0
            for ii in 1:compsum.ncomp
                if compsum.obsvecs[ii, jj] != 0
                    matchobs[ii] += ct
                    matchcounts[compsum.cadj[ii] + compsum.obsvecs[ii, jj]] += ct
                end
            end
        end
    end
    return matchcounts, matchobs
end

function counts_matches(C::LinkMatrix{G},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer
    
    #count occurences of each observation in obsvecs
    matchvecct = zeros(Int64, length(compsum.obsvecct))
    if C.nrow < C.ncol
        for (ii, jj) in enumerate(IndexLinear(), C.row2col)
            if jj != zero(G)
                matchvecct[compsum.obsidx[ii, jj]] += 1
            end
        end
    else
        for (jj, ii) in enumerate(IndexLinear(), C.col2row)
            if ii != zero(G)
                matchvecct[compsum.obsidx[ii, jj]] += 1
            end
        end
    end

    #map observation occurences to counts
    matchcounts = zeros(Int64, length(compsum.counts))
    matchobs = zeros(Int64, compsum.ncomp)

    for (jj, ct) in enumerate(IndexLinear(), matchvecct)
        if ct > 0
            for ii in 1:compsum.ncomp
                if compsum.obsvecs[ii, jj] != 0
                    matchobs[ii] += ct
                    matchcounts[compsum.cadj[ii] + compsum.obsvecs[ii, jj]] += ct
                end
            end
        end
    end
    return matchcounts, matchobs
end

function weights_vector(pM::Array{T, 1},
                        pU::Array{T, 1},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary},
                        comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightinc = log.(pM) - log.(pU)
    weightvec = zeros(Float64, length(compsum.obsvecct))

    for jj in 1:length(weightvec), ii in comps
        if compsum.obsvecs[ii, jj] != 0
            weightvec[jj] += weightinc[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
        end
    end
    
    return weightvec
end

function penalized_weights_vector(pM::Array{T, 1},
                                  pU::Array{T, 1},
                                  compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                  penalty::AbstractFloat = 0.0,
                                  comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum, comps)
    for ii in 1:length(weightvec)
        if weightvec[ii] > penalty
            weightvec[ii] -= penalty
        else
            weightvec[ii] = 0.0
        end
    end
    return weightvec
end
    
function weights_vector_integer(pM::Array{T, 1},
                                pU::Array{T, 1},
                                compsum::ComparisonSummary,
                                comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightinc = log.(pM) - log.(pU)
    weightvec = zeros(Float64, length(compsum.obsvecct))

    for jj in 1:length(weightvec), ii in comps
        if compsum.obsvecs[ii, jj] != 0
            weightvec[jj] += weightinc[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
        end
    end
    return Int64.(round.(weightinc .* 1.0e14))
end

"""
    weights_matrix(pM, pU, comparisonSummary) -> weightArray

Compute weights = log(p(γ|M) / p(γ|U)) for each comparison vector.  This is done
efficiently by computing the weight once for each observed comparison and then mapping
based on the storred array indicies.  Missing do not contribute to the weights assuming
ignorability.
"""
function weights_matrix(pM::Array{T, 1},
                        pU::Array{T, 1},
                        compsum::ComparisonSummary,
                        comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum, comps)
    return weightvec[compsum.obsidx]
end

function weights_matrix(pM::Array{T, 1},
                        pU::Array{T, 1},
                        compsum::SparseComparisonSummary,
                        comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum, comps)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])
end

function weights_matrix(weightvec::Array{T, 1},
                        compsum::ComparisonSummary) where T <: Real
    return weightvec[compsum.obsidx]
end

function weights_matrix(weightvec::Array{T, 1},
                        compsum::SparseComparisonSummary) where T <: Real
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])
end

"""
    maximum_weights_matrix(pM, pU, comparisonSummary) -> weightArray

Compute weights = log(p(γ|M) / p(γ|U)) for each comparison vector.  This is done over
the range of provided parameter values.  The maximum is then taken for each different
comparison vector.  This is done efficiently by computing the weight once for each
observed comparison and then mapping based on the storred array indicies.  Missing do
not contribute to the weights assuming ignorability.
"""
function maximum_weights_vector(pM::Array{T, 2},
                                pU::Array{T, 2},
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightinc = log.(pM) - log.(pU)
    weightmat = zeros(T, length(compsum.obsvecct), size(weightinc, 1))

    for jj in 1:length(compsum.obsvecct), ii in comps
        if compsum.obsvecs[ii, jj] != 0
            for kk in 1:size(weightinc, 1)
                weightmat[jj, kk] += weightinc[kk, compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
            end
        end
    end
    weightvec = vec(maximum(weightmat, 2))
    return weightvec
end

function maximum_weights_matrix(pM::Array{T, 2},
                                pU::Array{T, 2},
                                compsum::ComparisonSummary,
                                comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: Real
    weightvec = maximum_weights_vector(pM, pU, compsum, comps)
    return weightvec[compsum.obsidx]
end

function maximum_weights_matrix(pM::Array{T, 2},
                                pU::Array{T, 2},
                                compsum::SparseComparisonSummary,
                                comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: Real
    weightvec = maximum_weights_vector(pM, pU, compsum, comps)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])
end

function maximum_weights_matrix(weightvec::Array{T, 1},
                                compsum::ComparisonSummary) where T <: AbstractFloat
    return weightvec[compsum.obsidx]
end

"""
    penalized_weights_matrix(pM, pU, comparisonSummary) -> weightArray

Compute penalized weight = log(p(γ|M) / p(γ|U)) - penalty for each comparison vector.
Calculation is done efficiently by computing the weight once for each observed
comparison and then mapping based on the storred array indicies.  Missing values do not
contribute to the weights assuming ignorability.
"""
function penalized_weights_matrix(pweightvec::Array{T, 1}, compsum::ComparisonSummary) where T <: Real
    positiveweight = pweightvec .> zero(T)
    n = sum(compsum.obsvecct[positiveweight])
    
    rows = Array{Int64}(n)
    cols = Array{Int64}(n)
    pweights = Array{T}(n)

    ii = 0
    
    for col in 1:compsum.ncol
        for row in 1:compsum.nrow
            if positiveweight[compsum.obsidx[row, col]]
                ii += 1
                rows[ii] = row
                cols[ii] = col
                pweights[ii] = pweightvec[compsum.obsidx[row, col]]
            end
        end
    end

    return sparse(rows[1:ii], cols[1:ii], pweights[1:ii], compsum.nrow, compsum.ncol)
end

function penalized_weights_matrix(pweightvec::Array{T, 1}, compsum::SparseComparisonSummary) where T <: Real
    positiveweight = pweightvec .> zero(T)
    n = sum(compsum.obsvecct[positiveweight])

    idxvals = nonzeros(compsum.obsidx)
    idxrows = rowvals(compsum.obsidx)
    
    rows = Array{Int64}(n)
    cols = Array{Int64}(n)
    pweights = Array{T}(n)

    ii = 0
    
    for jj in 1:compsum.ncol
        for matidx in nzrange(compsum.obsidx, jj)
            if positiveweight[idxvals[matidx]]
                ii += 1
                rows[ii] = idxrows[matidx]
                cols[ii] = jj
                pweights[ii] = pweightvec[idxvals[matidx]]
            end
        end
    end

    return sparse(rows[1:ii], cols[1:ii], pweights[1:ii], compsum.nrow, compsum.ncol)
end

function penalized_weights_matrix(weightvec::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, penalty::AbstractFloat) where T <: AbstractFloat
    pweightvec = max.(weightvec .- penalty, 0.0)
    return penalized_weights_matrix(pweightvec, compsum)
end

function penalized_weights_matrix(pM::Array{T, 1},
                                  pU::Array{T, 1},
                                  compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                  penalty::AbstractFloat = 0.0,
                                  comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum, comps)
    pweightvec = max.(weightvec .- penalty, 0.0)
    return penalized_weights_matrix(pweightvec, compsum)
end


"""
    indicator_weights_matrix(pM, pU, comparisonSummary) -> boolArray

Compute indicator for log(p(γ|M) / p(γ|U)) > penalty for each comparison vector.
Calculation is done efficiently by computing the weight once for each observed
comparison and then mapping based on the storred array indicies.  Missing values do not
contribute to the weights assuming ignorability.
"""
function indicator_weights_matrix(pM::Array{T, 1},
                                  pU::Array{T, 1},
                                  compsum::ComparisonSummary,
                                  penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum)
    iweightvec = weightvec .> penalty
    return iweightvec[compsum.obsidx]
end

function indicator_weights_matrix(pM::Array{T, 1},
                                  pU::Array{T, 1},
                                  compsum::SparseComparisonSummary,
                                  penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum)
    iweightvec = weightvec .> penalty
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, collect(iweightvec[compsum.obsidx.nzval]))
end

function indicator_weights_matrix(weightvec::Array{T, 1},
                                  compsum::ComparisonSummary,
                                  penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    iweightvec = weightvec .> penalty
    return iweightvec[compsum.obsidx]
end

function indicator_weights_matrix(weightvec::Array{T, 1},
                                  compsum::SparseComparisonSummary,
                                  penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    iweightvec = weightvec .> penalty
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, collect(iweightvec[compsum.obsidx.nzval]))
end


"""
    compute_costs(pM, pU, comparisonSummary, penalty) -> costArray, maxcost

Compuate a cost matrix to transform maximization problem into minimiation problem.  
"""
function compute_costs(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::ComparisonSummary,
                       penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum) .- penalty
    maxcost = maximum(weightvec)
    costvec = fill(maxcost, length(weightvec))
    for ii in 1:length(weightvec)
        if weightvec[ii] > 0.0
            costvec[ii] = maxcost - weightvec[ii]
        end
    end
    return costvec[compsum.obsidx], maxcost
end

function compute_costs(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::SparseComparisonSummary,
                       penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    weightvec = weights_vector(pM, pU, compsum) .- penalty
    maxcost = maximum(weightvec)
    costvec = fill(maxcost, length(weightvec))
    for ii in 1:length(weightvec)
        if weightvec[ii] > 0.0
            costvec[ii] = maxcost - weightvec[ii]
        end
    end
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, costvec[compsum.obsidx.nzval]), maxcost
end


#SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, pweightvec[compsum.obsidx.nzval])

function compute_costs_shrunk(pM::Array{T, 1},
                              pU::Array{T, 1},
                              compsum::ComparisonSummary,
                              penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    costmatrix, maxcost = compute_costs(pM, pU, compsum, penalty)
    return costmatrix .- minimum(costmatrix, 2), maxcost
end

function compute_costs_integer(pM::Array{T, 1},
                                                   pU::Array{T, 1},
                                                   compsum::ComparisonSummary,
                                                   penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    weightvec = Int64.(round.((weights_vector(pM, pU, compsum) .- penalty) .* 1.0e14))
    maxcost = ceil(maximum(weightvec))
    costvec = fill(maxcost, length(weightvec))
    for ii in 1:length(weightvec)
        if weightvec[ii] > 0.0
            costvec[ii] = maxcost - weightvec[ii]
        end
    end
    return costvec[compsum.obsidx], maxcost
end

function bayesrule_posterior(w::Array{T, 1}, p::T) where T <: AbstractFloat
    return logistic.(logit(p) .+ w)
end

function threshold_sensitivity(pM::Array{T, 2},
                                                   pU::Array{T, 2},
                                                   compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <: AbstractFloat
    obsW = sort(maximum_weights_vector(pM, pU, compsum))
    breaksW = get_mids(obsW)
    maxW = maximum_weights_matrix(pM, pU, compsum)
    ccsummary = mapreduce(vcat, breaksW) do w
        rowLabels, colLabels, maxLabel = bipartite_cluster(maxW, w)
        cc = ConnectedComponents(rowLabels, colLabels, maxLabel)
        summarize_components(cc)'
    end
    return [obsW[1:end-1] obsW[2:end] ccsummary]
end
