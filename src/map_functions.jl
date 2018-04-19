"""
    count_matches(matchedrows, matchedcols, comparisonSummary) -> matchcounts, matchobs

Returns vectors comparable to comparisonSummary.counts and comparisonSummary.obsct
corresponding to 
"""
function counts_matches{G <: Integer}(mrows::Array{G, 1},
                                      mcols::Array{G, 1},
                                      compsum::Union{ComparisonSummary, SparseComparisonSummary})
    
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

function counts_matches{G <: Integer}(C::LinkMatrix{G},
                                      compsum::Union{ComparisonSummary, SparseComparisonSummary})
    
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

"""
    max_MU(matchRows, matchColumns, comparisonSummary, pseudoM, pseudoU) -> pM, pU

compute posterior maximums for M and U probabilities assuming a dirichlet prior.
"""
function max_MU{G <: Integer, T <: Real}(mrows::Array{G, 1},
                                         mcols::Array{G, 1},
                                         compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                         pseudoM::Array{T, 1},
                                         pseudoU::Array{T, 1})
    #Count Match / Non-match observations
    matchcounts, matchobs = counts_matches(mrows, mcols, compsum)
    nonmatchcounts = compsum.counts - matchcounts
    nonmatchobs = compsum.obsct - matchobs

    #Add regularization to generate denominators
    matchtotals = promote_type(G, T).(matchobs)
    nonmatchtotals = promote_type(G, T).(nonmatchobs)
    for (ii, pm, pu) in zip(compsum.cmap, pseudoM, pseudoU)
        matchtotals[ii] += pm
        nonmatchtotals[ii] += pu
    end

    #Generate probabilities
    pM = (T.(matchcounts) + pseudoM) ./ matchtotals[compsum.cmap]
    pU = (T.(nonmatchcounts) + pseudoU) ./ nonmatchtotals[compsum.cmap]
    
    return pM, pU
end

function weights_vector{T <: AbstractFloat}(pM::Array{T, 1},
                                            pU::Array{T, 1},
                                            compsum::Union{ComparisonSummary, SparseComparisonSummary})
    weightinc = log.(pM) - log.(pU)
    weightvec = zeros(Float64, length(compsum.obsvecct))

    for jj in 1:length(weightvec), ii in 1:compsum.ncomp
        if compsum.obsvecs[ii, jj] != 0
            weightvec[jj] += weightinc[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
        end
    end
    
    return weightvec
end

function penalized_weights_vector{T <: AbstractFloat}(pM::Array{T, 1},
                                                      pU::Array{T, 1},
                                                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                      penalty::AbstractFloat = 0.0)
    weightvec = weights_vector(pM, pU, compsum)
    for ii in 1:length(weightvec)
        if weightvec[ii] > penalty
            weightvec[ii] -= penalty
        else
            weightvec[ii] = 0.0
        end
    end
    return weightvec
end
    
function weights_vector_integer{T <: AbstractFloat}(pM::Array{T, 1},
                                                    pU::Array{T, 1},
                                                    compsum::ComparisonSummary)
    weightinc = log.(pM) - log.(pU)
    weightvec = zeros(Float64, length(compsum.obsvecct))

    for jj in 1:length(weightvec), ii in 1:compsum.ncomp
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
function weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                            pU::Array{T, 1},
                                            compsum::ComparisonSummary)
    weightvec = weights_vector(pM, pU, compsum)
    return weightvec[compsum.obsidx]
end

function weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                            pU::Array{T, 1},
                                            compsum::SparseComparisonSummary)
    weightvec = weights_vector(pM, pU, compsum)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])
end

function weights_matrix{T <: Real}(weightvec::Array{T, 1},
                                            compsum::ComparisonSummary)
    return weightvec[compsum.obsidx]
end

function weights_matrix{T <: Real}(weightvec::Array{T, 1},
                                            compsum::SparseComparisonSummary)
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
function maximum_weights_vector{T <: AbstractFloat}(pM::Array{T, 2},
                                                    pU::Array{T, 2},
                                                    compsum::Union{ComparisonSummary, SparseComparisonSummary})
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

function maximum_weights_matrix{T <: Real}(pM::Array{T, 2},
                                           pU::Array{T, 2},
                                           compsum::ComparisonSummary)
    weightvec = maximum_weights_vector(pM, pU, compsum)
    return weightvec[compsum.obsidx]
end

function maximum_weights_matrix{T <: Real}(pM::Array{T, 2},
                                           pU::Array{T, 2},
                                           compsum::SparseComparisonSummary)
    weightvec = maximum_weights_vector(pM, pU, compsum)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, weightvec[compsum.obsidx.nzval])
end

function maximum_weights_matrix{T <: AbstractFloat}(weightvec::Array{T, 1},
                                                    compsum::ComparisonSummary)
    return weightvec[compsum.obsidx]
end

"""
    penalized_weights_matrix(pM, pU, comparisonSummary) -> weightArray

Compute penalized weight = log(p(γ|M) / p(γ|U)) - penalty for each comparison vector.
Calculation is done efficiently by computing the weight once for each observed
comparison and then mapping based on the storred array indicies.  Missing values do not
contribute to the weights assuming ignorability.
"""
function penalized_weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                                      pU::Array{T, 1},
                                                      compsum::ComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    weightvec = weights_vector(pM, pU, compsum)
    pweightvec = max.(weightvec .- penalty, 0.0)
    return pweightvec[compsum.obsidx]
end

function penalized_weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                                      pU::Array{T, 1},
                                                      compsum::SparseComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    weightvec = weights_vector(pM, pU, compsum)
    pweightvec = max.(weightvec .- penalty, 0.0)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, pweightvec[compsum.obsidx.nzval])
end

function penalized_weights_matrix{T <: AbstractFloat}(weightvec::Array{T, 1},
                                                      compsum::ComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    pweightvec = max.(weightvec .- penalty, 0.0)
    return pweightvec[compsum.obsidx]
end

function penalized_weights_matrix{T <: AbstractFloat}(weightvec::Array{T, 1},
                                                      compsum::SparseComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    pweightvec = max.(weightvec .- penalty, 0.0)
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, pweightvec[compsum.obsidx.nzval])
end

"""
    indicator_weights_matrix(pM, pU, comparisonSummary) -> boolArray

Compute indicator for log(p(γ|M) / p(γ|U)) > penalty for each comparison vector.
Calculation is done efficiently by computing the weight once for each observed
comparison and then mapping based on the storred array indicies.  Missing values do not
contribute to the weights assuming ignorability.
"""
function indicator_weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                                       pU::Array{T, 1},
                                                       compsum::ComparisonSummary,
                                                       penalty::AbstractFloat = 0.0)
    weightvec = weights_vector(pM, pU, compsum)
    iweightvec = weightvec .> penalty
    return iweightvec[compsum.obsidx]
end

function indicator_weights_matrix{T <: AbstractFloat}(pM::Array{T, 1},
                                                       pU::Array{T, 1},
                                                       compsum::SparseComparisonSummary,
                                                       penalty::AbstractFloat = 0.0)
    weightvec = weights_vector(pM, pU, compsum)
    iweightvec = weightvec .> penalty
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, collect(iweightvec[compsum.obsidx.nzval]))
end

function indicator_weights_matrix{T <: AbstractFloat}(weightvec::Array{T, 1},
                                                      compsum::ComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    iweightvec = weightvec .> penalty
    return iweightvec[compsum.obsidx]
end

function indicator_weights_matrix{T <: AbstractFloat}(weightvec::Array{T, 1},
                                                      compsum::SparseComparisonSummary,
                                                      penalty::AbstractFloat = 0.0)
    iweightvec = weightvec .> penalty
    return SparseMatrixCSC(compsum.obsidx.m, compsum.obsidx.n, compsum.obsidx.colptr, compsum.obsidx.rowval, collect(iweightvec[compsum.obsidx.nzval]))
end


"""
    compute_costs(pM, pU, comparisonSummary, penalty) -> costArray, maxcost

Compuate a cost matrix to transform maximization problem into minimiation problem.  
"""
function compute_costs{T <: AbstractFloat}(pM::Array{T, 1},
                                           pU::Array{T, 1},
                                           compsum::ComparisonSummary,
                                           penalty::AbstractFloat = 0.0)
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

function compute_costs{T <: AbstractFloat}(pM::Array{T, 1},
                                           pU::Array{T, 1},
                                           compsum::SparseComparisonSummary,
                                           penalty::AbstractFloat = 0.0)
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

function compute_costs_shrunk{T <: AbstractFloat}(pM::Array{T, 1},
                                                  pU::Array{T, 1},
                                                  compsum::ComparisonSummary,
                                                  penalty::AbstractFloat = 0.0)
    costmatrix, maxcost = compute_costs(pM, pU, compsum, penalty)
    return costmatrix .- minimum(costmatrix, 2), maxcost
end

function compute_costs_integer{T <: AbstractFloat}(pM::Array{T, 1},
                                                   pU::Array{T, 1},
                                                   compsum::ComparisonSummary,
                                                   penalty::AbstractFloat = 0.0)
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

"""
    max_C(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns
"""
function max_C{T <: AbstractFloat}(pM::Array{T, 1},
                                   pU::Array{T, 1},
                                   compsum::ComparisonSummary,
                                   penalty::AbstractFloat = 0.0)

    if compsum.nrow <= compsum.ncol
        wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)
        sol = clue.solve_LSAP(RObject(wpenalized), maximum = RObject(true))
        mcols = rcopy(Array{Int64, 1}, sol)
        keep = falses(mcols)
        for (rr, cc) in enumerate(IndexLinear(), mcols)
            if wpenalized[rr, cc] > 0.0
                keep[rr] = true
            end
        end
        return find(keep), mcols[keep]
    else
        wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)'
        sol = clue.solve_LSAP(RObject(wpenalized), maximum = RObject(true))
        mrows = rcopy(Array{Int64, 1}, sol)
        keep = falses(mrows)
        for (rr, cc) in enumerate(IndexLinear(), mrows)
            if wpenalized[rr, cc] > 0.0
                keep[rr] = true
            end
        end
        return mrows[keep], find(keep)
    end
end

function max_C{T <: AbstractFloat}(pM::Array{T, 1},
                                   pU::Array{T, 1},
                                   compsum::SparseComparisonSummary,
                                   penalty::AbstractFloat = 0.0)

    if compsum.nrow <= compsum.ncol
        wpenalized = full(penalized_weights_matrix(pM, pU, compsum, penalty)) #makes it non-sparse do not want to use in most cases
        sol = clue.solve_LSAP(RObject(wpenalized), maximum = RObject(true))
        mcols = rcopy(Array{Int64, 1}, sol)
        keep = falses(mcols)
        for (rr, cc) in enumerate(IndexLinear(), mcols)
            if wpenalized[rr, cc] > 0.0
                keep[rr] = true
            end
        end
        return find(keep), mcols[keep]
    else
        wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)'
        sol = clue.solve_LSAP(RObject(wpenalized), maximum = RObject(true))
        mrows = rcopy(Array{Int64, 1}, sol)
        keep = falses(mrows)
        for (rr, cc) in enumerate(IndexLinear(), mrows)
            if wpenalized[rr, cc] > 0.0
                keep[rr] = true
            end
        end
        return mrows[keep], find(keep)
    end
end

"""
    max_C_complete(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns, keep, costs
"""
function max_C_offsets{T <: AbstractFloat}(pM::Array{T, 1},
                                           pU::Array{T, 1},
                                           compsum::ComparisonSummary,
                                           penalty::AbstractFloat = 0.0;
                                           verbose::Bool = false)
    
    costs, maxcost = compute_costs(pM, pU, compsum, penalty)
    #costs, maxcost = compute_costs_integer(pM, pU, compsum, penalty)
    #costs, maxcost = compute_costs_shrunk(pM, pU, compsum, penalty)
    shrunkcosts = costs .- minimum(costs, 2)
    #rows2cols, rowOffsets, colOffsets = lsap_solver(shrunkcosts)
    rows2cols, rowOffsets, colOffsets = lsap_solver_tracking(shrunkcosts, verbose = verbose)
    keep = falses(compsum.nrow)
    colassigned = falses(compsum.ncol)
    for (rr, cc) in enumerate(IndexLinear(), rows2cols)
        if rows2cols[rr] != 0 && costs[rr, cc] < maxcost
            keep[rr] = true
            colassigned[cc] = true
        end
    end
    #for rr in 1:length(rowOffsets)
    #    if !keep[rr] && !iszero(rowOffsets[rr])
    #        rowOffsets[rr] = 0.0
    #    end
    #end
    #for cc in 1:length(colOffsets)
    #    if !colassigned[cc] && !iszero(colOffsets[cc])
    #        colOffsets[cc] = 0.0
    #    end
    #end
    mrows = find(keep)
    mcols = rows2cols[mrows]
    return mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost
end

function max_C_initialized!{T <: AbstractFloat, G <: Real}(pM::Array{T, 1},
                                                           pU::Array{T, 1},
                                                           compsum::ComparisonSummary,
                                                           penalty::AbstractFloat,
                                                           rowInitial::Array{<:Integer, 1},
                                                           rowOffsets::Array{G, 1},
                                                           colOffsets::Array{G, 1},
                                                           maxcost0::Real;
                                                           verbose::Bool = false)
    
    costs, maxcost = compute_costs(pM, pU, compsum, penalty)
    #costs, maxcost = compute_costs_shrunk(pM, pU, compsum, penalty)
    #costs, maxcost = compute_costs_integer(pM, pU, compsum, penalty)
    #rowOffsets -= (maxcost0 - maxcost)
    shrunkcosts = costs .- minimum(costs, 2)
    #rows2cols, rowOffsets, colOffsets = lsap_solver!(shrunkcosts, rowOffsets, colOffsets, rowInitial, verbose = verbose)
    rows2cols, rowOffsets, colOffsets = lsap_solver_tracking!(shrunkcosts, rowOffsets, colOffsets, rowInitial, verbose = verbose)
    #if minimum(adjusted_cost(shrunkcosts, rowOffsets, colOffsets)) < 0.0
    #    return max_C_offsets(pM, pU, compsum, penalty)
    #end
    keep = falses(compsum.nrow)
    colassigned = falses(compsum.ncol)
    for (rr, cc) in enumerate(IndexLinear(), rows2cols)
        if cc != 0 && costs[rr, cc] < maxcost
            keep[rr] = true
            colassigned[cc] = true
        end
    end
    #for rr in 1:length(rowOffsets)
    #    if !keep[rr] && !iszero(rowOffsets[rr])
    #        rowOffsets[rr] = 0.0
    #    end
    #end
    #for cc in 1:length(colOffsets)
    #    if !colassigned[cc] && !iszero(colOffsets[cc])
    #        colOffsets[cc] = 0.0
    #    end
    #end
    mrows = find(keep)
    mcols = rows2cols[mrows]
    return mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost
end

"""
    maximizes by first 
"""
function max_C_cluster{T <: AbstractFloat}(pM::Array{T, 1},
                                           pU::Array{T, 1},
                                           compsum::ComparisonSummary,
                                           penalty::AbstractFloat = 0.0;
                                           verbose::Bool = false)
    
    ##Run clustering algorithm to split LSAP
    aboveThreshold = indicator_weights_matrix(pM, pU, compsum, penalty)
    rowLabels, colLabels, maxLabel = bipartite_cluster(aboveThreshold)

    ##Mark clusters for rows
    rowCluster = falses(compsum.nrow, maxLabel)
    for (ii, label) in enumerate(IndexLinear(), rowLabels)
        if label > 0
            rowCluster[ii, label] = true
        end
    end

    ##Mark clusters for columns
    colCluster = falses(compsum.ncol, maxLabel)
    for (ii, label) in enumerate(IndexLinear(), colLabels)
        if label > 0
            colCluster[ii, label] = true
        end
    end

    ##Compute cost matrix
    costs, maxcost = compute_costs(pM, pU, compsum, penalty)
    rows2cols = zeros(Int64, compsum.nrow)
    for kk in 1:maxLabel
        if verbose
            println("Cluster: $kk of $maxLabel")
        end
        clusterCols = find(colCluster[:, kk])
        clusterRows = find(rowCluster[:, kk])
        assignment = lsap_solver_tracking(costs[clusterRows, clusterCols])[1]
        for (row, colidx) in zip(clusterRows, assignment)
            if !iszero(colidx)
                rows2cols[row] = clusterCols[colidx]
            end
        end
    end
    
    keep = falses(compsum.nrow)
    for (ii, jj) in enumerate(IndexLinear(), rows2cols)
        if jj != 0 && (costs[ii, jj] < maxcost)
            keep[ii] = true
        end
    end
    mrows = find(keep)
    mcols = rows2cols[mrows]
    
    return mrows, mcols
end

function max_C_cluster{T <: AbstractFloat}(pM::Array{T, 1},
                                           pU::Array{T, 1},
                                           compsum::SparseComparisonSummary,
                                           penalty::AbstractFloat = 0.0;
                                           verbose::Bool = false)
    
    ##Run clustering algorithm to split LSAP
    aboveThreshold = indicator_weights_matrix(pM, pU, compsum, penalty)
    rowLabels, colLabels, maxLabel = bipartite_cluster_sparseblock(aboveThreshold)

    ##Mark clusters for rows
    rowCluster = falses(compsum.nrow, maxLabel)
    for (ii, label) in enumerate(IndexLinear(), rowLabels)
        if label > 0
            rowCluster[ii, label] = true
        end
    end

    ##Mark clusters for columns
    colCluster = falses(compsum.ncol, maxLabel)
    for (ii, label) in enumerate(IndexLinear(), colLabels)
        if label > 0
            colCluster[ii, label] = true
        end
    end

    ##Compute cost matrix
    costs, maxcost = compute_costs(pM, pU, compsum, penalty)
    rows2cols = zeros(Int64, compsum.nrow)
    for kk in 1:maxLabel
        clusterCols = find(colCluster[:, kk])
        clusterRows = find(rowCluster[:, kk])

        if verbose
            println("Cluster: $kk of $maxLabel")
        end
        
        #Matrix should technically already be full here, assuming blocks
        assignment = lsap_solver_tracking(full(costs[clusterRows, clusterCols]))[1]
        for (row, colidx) in zip(clusterRows, assignment)
            if !iszero(colidx)
                rows2cols[row] = clusterCols[colidx]
            end
        end
    end
    
    keep = falses(compsum.nrow)
    for (ii, jj) in enumerate(IndexLinear(), rows2cols)
        if jj != 0 && (costs[ii, jj] < maxcost)
            keep[ii] = true
        end
    end
    mrows = find(keep)
    mcols = rows2cols[mrows]
    
    return mrows, mcols
end

function next_penalty{T <: AbstractFloat}(pM::Array{T, 1},
                                          pU::Array{T, 1},
                                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                          penalty0::AbstractFloat)
    wv = sort!(weights_vector(pM, pU, compsum))
    ii = findfirst(x -> x > penalty0, wv)
    if ii == 0
        return wv[end], 0
    elseif ii < length(wv)
        return 0.95 * wv[ii] + 0.05 * wv[ii + 1], length(wv) - ii + 1
        #return mean(wv[ii:ii + 1]), length(wv) - ii + 1
    else
        return wv[ii], 1
    end
end

function lsap_checkoptimal{G <: Integer, T <: AbstractFloat}(mrows::Array{G, 1},
                                                             mcols::Array{G, 1},
                                                             costs::Array{T, 2},
                                                             tol::AbstractFloat = 1.0e-8)
    if size(costs, 1) > size(costs, 2)
        lsap_checkoptimal(mcols, mrows, costs')
    end
    #rowmins = vec(minimum(costs, 2))
    #colmins = vec(minimum(costs, 1))
    #u = zeros(Float64, size(costs, 1))
    u = vec(minimum(costs, 2))
    v = zeros(Float64, size(costs, 2))
    vcols = falses(size(costs, 2))
    for (ii, jj) in zip(mrows, mcols)
        if cmin < costs[ii, jj]
            v[jj] = cmin - costs[ii, jj]
            u[ii] = costs[ii, jj] - v[jj]
            vcols[jj] = true
        else
            u[ii] = costs[ii, jj]
        end
    end
    #for (ii, jj) in zip(mrows, mcols)
    #    if !isapprox(costs[ii, jj], u[ii] + v[jj])
    #        return isapprox(costs[ii, jj], u[ii] + v[jj]), v, u
    #    end
    #end
    #primal = sum(costs[CartesianIndex.(mrows, mcols)])
    #dual = sum(v) + sum(u)
    #return isapprox(primal, dual), v, u
    for jj in 1:size(costs, 2), ii in 1:size(costs, 1)
        if vcols[jj]
            if costs[ii, jj] < u[ii] + v[jj]
                println(ii, " ", jj, " constraint violation")
                return false, v, u
            end
        else
            if costs[ii, jj] < u[ii]
                println(ii, " ", jj, " constraint violation")
                return false, v, u
            end
        end
    end
    return true, v, u
end

"""
    map_solver(pM0, pU0, comparisonSummary, [priorM], [priorU], penalty; maxIter) -> matchRows, matchColumns, pM, pU, iterations
"""
function map_solver{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                   pU0::Array{G, 1},
                                                   compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                   priorM::Array{T, 1} = ones(T, length(compsum.counts)),
                                                   priorU::Array{T, 1} = ones(T, length(compsum.counts)),
                                                   penalty::AbstractFloat = 0.0;
                                                   maxIter::Integer = 100,
                                                   verbose::Bool = false)
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    currmrows, currmcols = max_C(pM0, pU0, compsum, penalty)
    iter = 0
    while iter < maxIter
        iter += 1
        if verbose
            println("Iteration: $iter")
            nmatch = length(currmrows)
            println("Matches: $nmatch")
        end
        pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
        newmrows, newmcols = max_C(pM, pU, compsum, penalty)
        if length(newmrows) == length(currmrows)
            if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                return currmrows, currmcols, pM, pU, iter
            end
        end
        currmrows = newmrows
        currmcols = newmcols
    end
    println("Maximum number of iterations reached")
    return currmrows, currmcols, pM, pU, iter
end

"""
    map_solver(pM0, pU0, comparisonSummary, [priorM], [priorU], penalty; maxIter) -> matchRows, matchColumns, pM, pU, iterations
"""
function map_solver_initialize{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                              pU0::Array{G, 1},
                                                              compsum::ComparisonSummary{<:Integer, <:Integer},
                                                              priorM::Array{T, 1} = ones(T, length(compsum.counts)),
                                                              priorU::Array{T, 1} = ones(T, length(compsum.counts)),
                                                              penalty::AbstractFloat = 0.0;
                                                              maxIter::Integer = 100,
                                                              verbose::Bool = false)
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    currmrows, currmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM0, pU0, compsum, penalty)
    iter = 0
    while iter < maxIter
        iter += 1
        if verbose
            println("Iteration: $iter")
            nmatch = length(currmrows)
            println("Matches: $nmatch")
        end
        pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
        newmrows, newmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
        if length(newmrows) == length(currmrows)
            if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                return currmrows, currmcols, pM, pU, iter
            end
        end
        currmrows = newmrows
        currmcols = newmcols
    end
    println("Maximum number of iterations reached")
    return currmrows, currmcols, pM, pU, iter
end

function map_solver_cluster{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                           pU0::Array{G, 1},
                                                           compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                           priorM::Array{T, 1},
                                                           priorU::Array{T, 1},
                                                           penalty::AbstractFloat = 0.0;
                                                           maxIter::Integer = 100,
                                                           verbose::Bool = false)
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    currmrows, currmcols = max_C_cluster(pM0, pU0, compsum, penalty)
    pM = copy(pM0)
    pU = copy(pU0)
    iter = 0
    while iter < maxIter
        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $pM")
            nmatch = length(currmrows)
            println("Matches: $nmatch")
        end
        pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
        newmrows, newmcols = max_C_cluster(pM, pU, compsum, penalty)
        if length(newmrows) == length(currmrows)
            if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                return currmrows, currmcols, pM, pU, iter
            end
        end
        currmrows = newmrows
        currmcols = newmcols
    end
    println("Maximum number of iterations reached")
    return currmrows, currmcols, pM, pU, iter
end


"""
    map_solver_iter(pM0, pU0, comparisonSummary, [priorM], [priorU], penaltyRng; maxIter) -> matchRows, matchColumns, pM, pU, iterations
"""
function map_solver_iter{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                        pU0::Array{G, 1},
                                                        compsum::ComparisonSummary{<:Integer, <:Integer},
                                                        priorM::Array{T, 1},
                                                        priorU::Array{T, 1},
                                                        penaltyRng::Range;
                                                        maxIter::Integer = 100,
                                                        verbose::Bool = false)
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, penaltyRng[1], maxIter = maxIter)
    outM[1, :] = pM
    outU[1, :] = pU
    outIter[1] = iter
    outMatches = Dict(1 => (mrows, mcols))
    
    for ii in 2:npenalty
        if verbose
            penalty = penaltyRng[ii]
            println("Step: $ii, Penalty $penalty")
        end
        mrows, mcols, pM, pU, iter = map_solver(outM[ii - 1, :], outU[ii - 1, :], compsum, priorM, priorU, penaltyRng[ii], maxIter = maxIter)
        if length(mrows)  .== 0
            warn("No matches found with penalty of $(penaltyRng[ii])")
            return outMatches, outM, outU, outIter
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end

function map_solver_iter_cluster{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                                pU0::Array{G, 1},
                                                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                                priorM::Array{T, 1},
                                                                priorU::Array{T, 1},
                                                                penaltyRng::Range;
                                                                maxIter::Integer = 100,
                                                                verbose::Bool = false)
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver_cluster(pM0, pU0, compsum, priorM, priorU, penaltyRng[1], maxIter = maxIter, verbose = verbose)
    outM[1, :] = pM
    outU[1, :] = pU
    outIter[1] = iter
    outMatches = Dict(1 => (mrows, mcols))
    
    for ii in 2:npenalty
        if verbose
            penalty = penaltyRng[ii]
            println("Step: $ii, Penalty $penalty")
        end
        mrows, mcols, pM, pU, iter = map_solver_cluster(outM[ii - 1, :], outU[ii - 1, :], compsum, priorM, priorU, penaltyRng[ii], maxIter = maxIter, verbose = verbose)
        if length(mrows)  .== 0
            warn("No matches found with penalty of $(penaltyRng[ii])")
            return outMatches, outM, outU, outIter
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end


function map_solver_iter_initialize{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                                   pU0::Array{G, 1},
                                                                   compsum::ComparisonSummary{<:Integer, <:Integer},
                                                                   priorM::Array{T, 1},
                                                                   priorU::Array{T, 1},
                                                                   penaltyRng::Range;
                                                                   maxIter::Integer = 100,
                                                                   verbose::Bool = false)
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    pM = copy(pM0)
    pU = copy(pU0)
    
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    outMatches = Dict{Int64,Tuple{Array{Int64,1},Array{Int64,1}}}()
    #mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM, pU, compsum, penaltyRng[1])
    currmrows, currmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM, pU, compsum, penaltyRng[1])
    for (ii, penalty) in enumerate(IndexLinear(), penaltyRng)
        if verbose
            println("Step: $ii, Penalty $penalty")
        end
        iter = 0
        while iter < maxIter
            iter += 1
            pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
            #npM, npU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
            #if all(npM .== pM) && all(npU .== pU)
            #    break
            #else
            #    pM = npM
            #    pU = npU
            #end
            newmrows, newmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
            #mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
            if length(newmrows) == length(currmrows)
                if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                    break
                end
            end
            currmrows = newmrows
            currmcols = newmcols
        end
        if iter >= maxIter
            println("Maximum number of iterations reached")
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (currmrows, currmcols)
        #outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end

function map_solver_search{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                          pU0::Array{G, 1},
                                                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                          priorM::Array{T, 1},
                                                          priorU::Array{T, 1},
                                                          penalty0::Real = 0.0;
                                                          maxIter::Integer = 100,
                                                          verbose::Bool = false)
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, penalty0, maxIter = maxIter)
    outM = copy(pM)
    outU = copy(pU)
    outIter = [iter]
    penalties = [penalty0]
    outMatches = Dict(1 => (mrows, mcols))
    penalty, nabove = next_penalty(pM, pU, compsum, penalty0)
    ii = 1
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        ii += 1
        mrows, mcols, pM, pU, iter = map_solver(pM, pU, compsum, priorM, priorU, penalty, maxIter = maxIter)
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (mrows, mcols)
        push!(penalties, penalty)
        penalty, nabove = next_penalty(pM, pU, compsum, penalty)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        keep = [w[compsum.obsidx[row, col]] > 0.0 for (row, col) in zip(mrows, mcols)]
        mrows = mrows[keep]
        mcols = mcols[keep]
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
    end
    return outMatches, outM', outU', penalties, outIter
end

function map_solver_search_cluster{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                                  pU0::Array{G, 1},
                                                                  compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                                  priorM::Array{T, 1},
                                                                  priorU::Array{T, 1},
                                                                  penalty0::Real = 0.0;
                                                                  maxIter::Integer = 100,
                                                                  verbose::Bool = false)
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver_cluster(pM0, pU0, compsum, priorM, priorU, penalty0, maxIter = maxIter, verbose = verbose)
    outM = copy(pM)
    outU = copy(pU)
    outIter = [iter]
    penalties = [penalty0]
    outMatches = Dict(1 => (mrows, mcols))
    penalty, nabove = next_penalty(pM, pU, compsum, penalty0)
    ii = 1
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        ii += 1
        mrows, mcols, pM, pU, iter = map_solver_cluster(pM, pU, compsum, priorM, priorU, penalty, maxIter = maxIter, verbose = verbose)
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (mrows, mcols)
        push!(penalties, penalty)
        penalty, nabove = next_penalty(pM, pU, compsum, penalty)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        keep = [w[compsum.obsidx[row, col]] > 0.0 for (row, col) in zip(mrows, mcols)]
        mrows = mrows[keep]
        mcols = mcols[keep]
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    end
    return outMatches, outM', outU', penalties, outIter
end

function map_solver_search_initialize{G <: AbstractFloat, T <: Real}(pM0::Array{G, 1},
                                                                     pU0::Array{G, 1},
                                                                     compsum::ComparisonSummary{<:Integer, <:Integer},
                                                                     priorM::Array{T, 1},
                                                                     priorU::Array{T, 1},
                                                                     penalty0::Real = 0.0;
                                                                     maxIter::Integer = 100,
                                                                     verbose::Bool = false)
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    pM = copy(pM0)
    pU = copy(pU0)
    penalty = copy(penalty0)
    
    #Solver for first value
    currmrows, currmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM, pU, compsum, penalty)
    outM = Array{T}(length(pM), 0)
    outU = Array{T}(length(pU), 0)
    outIter = Array{Int}(0)
    penalties = Array{typeof(penalty)}(0)
    outMatches = Dict{Int,Tuple{Array{Int,1},Array{Int,1}}}()

    nabove = count(weights_vector(pM, pU, compsum) .> penalty)
    ii = 0
    while nabove > 1
        ii += 1
        iter = 0
        while iter < maxIter
            iter += 1
            pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
            newmrows, newmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
            if length(newmrows) == length(currmrows)
                if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                    break
                end
            end
            currmrows = newmrows
            currmcols = newmcols
        end
        
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (currmrows, currmcols)
        push!(penalties, penalty)
        if verbose
            println("penalty: $penalty, matches: $(length(currmrows)), nabove: $nabove")
        end
        penalty, nabove = next_penalty(pM, pU, compsum, penalty)
    end
    return outMatches, outM', outU', penalties, outIter
end
