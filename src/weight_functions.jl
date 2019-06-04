"""
Find the smallest margin between sorted entries in a vector
"""
function minimum_margin(x::Array{<:Real, 1})
    s = sort(unique(round.(x, digits=10)))
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
    for (jj, ct) in pairs(IndexLinear(), matchvecct)
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

#change to count_matches(findn(C.row2col)..., compsum)?
function counts_matches(row2col::Array{G, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer
    
    #count occurences of each observation in obsvecs
    matchvecct = zeros(Int64, length(compsum.obsvecct))
    for (ii, jj) in pairs(IndexLinear(), row2col)
        if jj != zero(G)
            matchvecct[compsum.obsidx[ii, jj]] += 1
        end
    end

    #map observation occurences to counts
    matchcounts = zeros(Int64, length(compsum.counts))
    matchobs = zeros(Int64, compsum.ncomp)

    for (jj, ct) in pairs(IndexLinear(), matchvecct)
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

counts_matches(C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer = counts_matches(C.row2col, compsum)

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function weights_vector(pM::Array{T, 1},
                        pU::Array{T, 1},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary},
                        comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    weightinc = log.(pM) - log.(pU)
    weightvec = zeros(Float64, length(compsum.obsvecct))

    for jj in 1:length(weightvec), ii in 1:compsum.ncomp
        if compsum.obsvecs[ii, jj] != 0
            weightvec[jj] += weightinc[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
        end
    end
    
    return weightvec
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function shrink_weights(x::G, threshold::G) where G <: Real
    return x > threshold ? x - threshold : zero(G)
end

function shrink_weights!(x::Array{G, 1}, threshold::G) where G <: Real
    for ii in 1:length(x)
        x[ii] = shrink_weights(x[ii], threshold)
    end
    return x
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function penalized_weights_vector(pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    return shrink_weights!(weights_vector(pM, pU, compsum), penalty)
end

"""
    weights_matrix(pM, pU, comparisonSummary) -> weightArray

Compute weights = log(p(γ|M) / p(γ|U)) for each comparison vector.  This is done
efficiently by computing the weight once for each observed comparison and then mapping
based on the storred array indicies.  Missing do not contribute to the weights assuming
ignorability.
"""
weights_matrix(weightvec::Array{T, 1}, compsum::ComparisonSummary) where T <: Real = weightvec[compsum.obsidx]

weights_matrix(weightvec::Array{T, 1}, compsum::SparseComparisonSummary) where T <: Real = SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])

weights_matrix(pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <: AbstractFloat = weights_matrix(weights_vector(pM, pU, compsum), compsum)


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
                                compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <: AbstractFloat
    weightinc = log.(pM) - log.(pU)
    weightmat = zeros(T, length(compsum.obsvecct), size(weightinc, 1))

    for jj in 1:length(compsum.obsvecct), ii in 1:compsum.ncomp
        if compsum.obsvecs[ii, jj] != 0
            for kk in 1:size(weightinc, 1)
                weightmat[jj, kk] += weightinc[kk, compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
            end
        end
    end
    weightvec = vec(maximum(weightmat, 2))
    return weightvec
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
maximum_weights_matrix(pM::Array{T, 2}, pU::Array{T, 2}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <: Real = maximum_weights_matrix(maximum_weights_vector(pM, pU, compsum), compsum)

maximum_weights_matrix(weightvec::Array{T, 1}, compsum::ComparisonSummary) where T <: AbstractFloat = weightvec[compsum.obsidx]

maximum_weights_matrix(weightvec::Array{T, 1}, compsum::SparseComparisonSummary) where T <: AbstractFloat = SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])

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
    
    rows = Array{Int64}(undef, n)
    cols = Array{Int64}(undef, n)
    pweights = Array{T}(undef, n)

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
    
    rows = Array{Int64}(undef, n)
    cols = Array{Int64}(undef, n)
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

penalized_weights_matrix(weightvec::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, penalty::T) where T <: AbstractFloat = penalized_weights_matrix(shrink_weights(weightvec, penalty), compsum)


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
indicator_weights_matrix(weightvec::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, penalty::AbstractFloat = 0.0) where T <: AbstractFloat = penalized_weights_matrix(map(w -> w > penalty, weightvec), compsum)

indicator_weights_matrix(pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union(ComparisonSummary, SparseComparisonSummary), penalty::AbstractFloat = 0.0) where T <: AbstractFloat = indicator_weights_matrix(weights_vector(pM, pU, compsum), compsum, penalty)

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
"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function compute_costs_shrunk(pM::Array{T, 1},
                              pU::Array{T, 1},
                              compsum::ComparisonSummary,
                              penalty::AbstractFloat = 0.0) where T <: AbstractFloat
    costmatrix, maxcost = compute_costs(pM, pU, compsum, penalty)
    return costmatrix .- minimum(costmatrix, 2), maxcost
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
bayesrule_posterior(w::Array{T, 1}, p::T) where T <: AbstractFloat = logistic.(logit(p) .+ w)


"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function threshold_sensitivity(pM::Array{T, 2},
                               pU::Array{T, 2},
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               comps::Array{Int64, 1} = collect(1:compsum.ncomp)) where T <: AbstractFloat
    obsW = sort(maximum_weights_vector(pM, pU, compsum, comps))
    breaksW = get_mids(obsW)
    maxW = maximum_weights_matrix(pM, pU, compsum, comps)
    ccsummary = mapreduce(vcat, breaksW) do w
        rowLabels, colLabels, maxLabel = bipartite_cluster(maxW, w)
        cc = ConnectedComponents(rowLabels, colLabels, maxLabel)
        summarize_components(cc)'
    end
    return [obsW[1:end-1] obsW[2:end] ccsummary]
end
