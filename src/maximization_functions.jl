"""
    max_MU(matchRows, matchColumns, comparisonSummary, pseudoM, pseudoU) -> pM, pU

compute posterior maximums for M and U probabilities assuming a dirichlet prior.
"""
function max_MU(mrows::Array{G, 1},
                mcols::Array{G, 1},
                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                pseudoM::Array{T, 1},
                pseudoU::Array{T, 1}) where {G <: Integer, T <: Real}
    #Count Match / Non-match observations
    matchcounts, matchobs = counts_matches(mrows, mcols, compsum)
    nonmatchcounts = compsum.counts - matchcounts
    nonmatchobs = compsum.obsct - matchobs

    #Add regularization to generate denominators
    matchtotals = promote_type(Int64, T).(matchobs)
    nonmatchtotals = promote_type(Int64, T).(nonmatchobs)
    for (ii, pm, pu) in zip(compsum.cmap, pseudoM, pseudoU)
        matchtotals[ii] += pm
        nonmatchtotals[ii] += pu
    end

    #Generate probabilities
    pM = (T.(matchcounts) + pseudoM) ./ matchtotals[compsum.cmap]
    pU = (T.(nonmatchcounts) + pseudoU) ./ nonmatchtotals[compsum.cmap]
    
    return pM, pU
end

function max_MU(rows2cols::Array{G, 1},
                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                pseudoM::Array{T, 1},
                pseudoU::Array{T, 1}) where {G <: Integer, T <: Real}
    #Count Match / Non-match observations
    matchcounts, matchobs = counts_matches(rows2cols, compsum)
    nonmatchcounts = compsum.counts - matchcounts
    nonmatchobs = compsum.obsct - matchobs

    #Add regularization to generate denominators
    matchtotals = promote_type(Int64, T).(matchobs)
    nonmatchtotals = promote_type(Int64, T).(nonmatchobs)
    for (ii, pm, pu) in zip(compsum.cmap, pseudoM, pseudoU)
        matchtotals[ii] += pm
        nonmatchtotals[ii] += pu
    end

    #Generate probabilities
    pM = (T.(matchcounts) + pseudoM) ./ matchtotals[compsum.cmap]
    pU = (T.(nonmatchcounts) + pseudoU) ./ nonmatchtotals[compsum.cmap]
    
    return pM, pU
end

"""
    max_C(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns
"""
function max_C(pM::Array{T, 1},
               pU::Array{T, 1},
               compsum::Union{ComparisonSummary, SparseComparisonSummary},
               penalty::AbstractFloat = 0.0) where T <: AbstractFloat

    #assignment algorithm should handle case where nrow > ncol 
    wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)
    astate = auction_assignment_padasfr2(wpenalized)[1]
    keep = astate.r2c .<= compsum.ncol
    return findall(keep), astate.r2c[keep]
end

function max_C_cluster(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                       penalty::AbstractFloat = 0.0;
                       verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)
    rowLabels, colLabels, maxLabel = bipartite_cluster(wpenalized, 0.0)

    ##Mark clusters for rows
    block2rows = label2dict(rowLabels)
    block2cols = label2dict(colLabels)
    
    if verbose
        maxsize = maximum([length(block2rows[ii]) * length(block2cols[ii]) for ii in 1:maxLabel])
        println("Maximum Cluster Size: $maxsize")
    end
    
    ##Compute cost matrix
    rows2cols = zeros(Int64, compsum.nrow)
    for kk in 1:maxLabel

        if verbose
            println("Cluster: $kk of $maxLabel")
        end
        
        #Matrix should technically already be full here, assuming blocks
        if length(block2rows[kk]) == 1 && length(block2cols[kk]) == 1
            rows2cols[block2rows[kk][1]] = block2cols[kk][1]
        else
            astate = auction_assignment_padasfr2(wpenalized[block2rows[kk], block2cols[kk]])[1]
            for ridx in 1:astate.nrow
                if !iszero(astate.r2c[ridx]) && astate.r2c[ridx] < length(block2cols[kk])
                    rows2cols[block2rows[kk][ridx]] = block2cols[kk][astate.r2c[ridx]]
                end
            end
        end
    end
    
    keep = .!iszero.(rows2cols)
    return findall(keep), rows2cols[keep]
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
function max_C_auction(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                       penalty::AbstractFloat = 0.0;
                       lambda0::T = zero(T), epsi0::T = -one(T), epsiscale::T = T(0.2),
                       minmargin::T = zero(T), digits::Integer = 5) where T <: AbstractFloat
    #Compute weights
    #wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = weights_matrix(w, compsum)

    #Determine error levels for complete assignment
    margin = minimum_margin(w, minmargin, digits)
    if epsi0 < zero(T)
        epsi0 = margin * epsiscale
    end
    epsitol = margin / min(compsum.nrow, compsum.ncol)
    #epsitol::T
    #Solve assignment problem
    astate, lambda = auction_assignment_padasfr2(wpenalized)
    #astate, lambda = auction_assignment_padasfr2(wpenalized, astate = astate, lambda0 = lambda, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale)
    #r2c, c2r, rowCosts, colCosts, λ = scaling_forward_backward(weightMat, ε0, εfinal, εscale)
    #λ = minimum(colCosts[.!iszero.(c2r)])

    #Convert assignment format, should change the convention here eventually
    #mrows = Int[]
    #for ii in 1:length(r2c)
    #    if !iszero(r2c[ii]) && weightMat[ii, r2c[ii]] > 0.0
    #        push!(mrows, ii)
    #    end
    #end
    #mcols = r2c[mrows]
    
    return astate, w, lambda, epsilon
end

#astate::AssignmentState

#=
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
function max_C_cluster(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::ComparisonSummary,
                       penalty::AbstractFloat = 0.0;
                       verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    aboveThreshold = indicator_weights_matrix(pM, pU, compsum, penalty)
    rowLabels, colLabels, maxLabel = bipartite_cluster(aboveThreshold) #diff

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

        assignment = lsap_solver_tracking(costs[clusterRows, clusterCols])[1] #diff
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
=#

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
function max_C_cluster2(pM::Array{T, 1},
                        pU::Array{T, 1},
                        compsum::ComparisonSummary,
                        penalty::AbstractFloat = 0.0;
                        verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = dropzeros(sparse(weights_matrix(w, compsum)))
    rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat)

    concomp = ConnectedComponents(rowLabels, colLabels, maxLabel)

    if verbose
        maxsize = maximum(concomp.rowcounts[2:end] .* concomp.colcounts[2:end])
        println("Maximum Cluster Size: $maxsize")
    end
    
    ##Compute cost matrix
    costs, maxcost = compute_costs(pM, pU, compsum, penalty)
    rows2cols = zeros(Int64, compsum.nrow)
    for kk in 1:maxLabel
        clusterRows, clusterCols = get_component(kk, concomp)

        if verbose
            println("Cluster: $kk of $maxLabel")
        end

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

function max_C_auction(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::ComparisonSummary,
                       rowCosts::Array{T, 1},
                       colCosts::Array{T, 1},
                       prevw::Array{T, 1},
                       prevε::T,
                       λ::T,
                       penalty::AbstractFloat = 0.0,
                       εscale::T = 0.2) where T <: AbstractFloat
    #Compute weights
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = weights_matrix(w, compsum)

    #Determine error levels for complete assignment
    mininc, maxinc = extrema(w - prevw)
    maxmove = maxinc - mininc
    if maxinc > 0.0
        if compsum.nrow <= compsum.ncol
            rowCosts += maxinc
        else
            colCosts += maxinc
        end
    end
    ε0 = εscale * (prevε + maxmove)
    margin = minimum_margin(w)
    εfinal = margin / min(compsum.nrow, compsum.ncol)
    
    #Solve assignment problem
    r2c, c2r, rowCosts, colCosts, λ = scaling_forward_backward(weightMat, rowCosts, colCosts, ε0, εfinal, εscale, λ)
    if compsum.nrow <= compsum.ncol
        λ = minimum(colCosts[.!iszero.(c2r)])
    else
        λ = minimum(rowCosts[.!iszero.(r2c)])
    end
    
    #Convert assignment format, should change the convention here eventually
    mrows = Int[]
    for ii in 1:length(r2c)
        if !iszero(r2c[ii]) && weightMat[ii, r2c[ii]] > 0.0
            push!(mrows, ii)
        end
    end
    mcols = r2c[mrows]
    
    return mrows, mcols, rowCosts, colCosts, w, εfinal, λ
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
function max_C_auction_cluster(pM::Array{T, 1},
                               pU::Array{T, 1},
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               penalty::AbstractFloat = 0.0,
                               εscale::AbstractFloat = 0.2,
                               verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = dropzeros(sparse(penalized_weights_matrix(w, compsum)))
    rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat)
    concomp = ConnectedComponents(rowLabels, colLabels, maxLabel)
    
    ##Find cluster indexes
    if verbose
        maxsize = maximum(concomp.rowcounts[2:end] .* concomp.colcounts[2:end])
        println("Maximum Cluster Size: $maxsize")
    end
    
    #Determine error levels for complete assignment
    margin = minimum_margin(w)
    ε0 = 0.5 * margin
    
    ##Loop over clusters
    r2c = zeros(Int64, compsum.nrow)
    c2r = zeros(Int64, compsum.ncol)
    rowCosts = zeros(T, compsum.nrow)
    colCosts = zeros(T, compsum.ncol)
    
    for kk in 1:maxLabel

        clusterRows, clusterCols = get_component(kk, concomp)
        rct, cct = get_dimensions(kk, concomp)
        
        #Set precision for cluster
        εfinal = margin / min(rct, cct)
        
        if verbose
            println("Cluster: $kk of $maxLabel")
        end

        if rct == 1 && cct == 1
            r2c[clusterRows[1]] = clusterCols[1]
            c2r[clusterCols[1]] = clusterRows[1]
            rowCosts[clusterRows[1]] = weightMat[clusterRows[1], clusterCols[1]] - margin
            colCosts[clusterCols[1]] = margin
        else
        
            #Run auction algorithm
            r2c[clusterRows], c2r[clusterCols], rowCosts[clusterRows], colCosts[clusterCols], λ = scaling_forward_backward(full(weightMat[clusterRows, clusterCols]), ε0, εfinal, εscale, check = true)

            #Map from subset rows / columns to actual
            for ii in clusterRows
                if !iszero(r2c[ii])
                    r2c[ii] = clusterCols[r2c[ii]]
                end
            end
            for jj in clusterCols
                if !iszero(c2r[jj])
                    c2r[jj] = clusterRows[c2r[jj]]
                end
            end
        end
    end
    
    mrows = Int[]
    for (ii, jj) in enumerate(IndexLinear(), r2c)
        if jj != 0 && (weightMat[ii, jj] > 0.0)
            push!(mrows, ii)
        end
    end
    mcols = r2c[mrows]
    return mrows, mcols, r2c, c2r, rowCosts, colCosts, w, margin
end

function max_C_auction_cluster(pM::Array{T, 1},
                               pU::Array{T, 1},
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               r2c::Array{<:Integer, 1},
                               c2r::Array{<:Integer, 1},
                               rowCosts::Array{T, 1},
                               colCosts::Array{T, 1},
                               prevw::Array{T, 1},
                               prevmargin::AbstractFloat,
                               penalty::AbstractFloat = 0.0,
                               εscale::AbstractFloat = 0.2;
                               verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    if maximum(w) <= zero(T)
        return Int64[], Int64[], r2c, c2r, rowCosts, colCosts, w, 0.0
    end
    weightMat = dropzeros(sparse(penalized_weights_matrix(w, compsum)))
    rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat)
    concomp = ConnectedComponents(rowLabels, colLabels, maxLabel)

    if verbose
        maxsize = maximum(concomp.rowcounts[2:end] .* concomp.colcounts[2:end])
        println("Maximum Cluster Size: $maxsize")
    end
    
    #Determine error levels for complete assignment
    mininc, maxinc = extrema(w - prevw)
    maxmove = maxinc - mininc
    ε0 = εscale * (prevmargin + maxmove)
    margin = minimum_margin(w)
    
    ##Loop over clusters
    for kk in 1:maxLabel

        if verbose
            println("Cluster: $kk of $maxLabel")
        end
        
        #Define cluster quantities        
        clusterRows, clusterCols = get_component(kk, concomp)
        rct, cct = get_dimensions(kk, concomp)
        
        if rct == 1 && cct == 1
            r2c[clusterRows[1]] = clusterCols[1]
            c2r[clusterCols[1]] = clusterRows[1]
            rowCosts[clusterRows[1]] = weightMat[clusterRows[1], clusterCols[1]] - margin
            colCosts[clusterCols[1]] = margin
        else
            
            #Set precision for cluster
            if maxinc > 0.0
                if rct <= cct
                    rowCosts[clusterRows] += maxinc
                else
                    colCosts += maxinc
                end
            end

            if rct <= cct
                for jj in clusterCols
                    if colCosts[jj] < 0.0
                        colCosts[jj] = 0.0
                    end
                end
            else
                for ii in clusterRows
                    if rowCosts[ii] < 0.0
                        rowCosts[ii] = 0.0
                    end
                end
            end

            for ii in clusterRows
                if !in(r2c[ii], clusterCols)
                    r2c[ii] = 0
                end
            end

            for jj in clusterCols
                if in(c2r[jj], clusterRows)
                    c2r[jj] = 0
                end
            end

            if rct < cct
                nz = .!iszero.(c2r[clusterCols])
                if count(nz) > 0
                    λ = minimum(colCosts[clusterCols][nz])
                else
                    λ = 0.0
                end
            elseif cct < rct
                nz = .!iszero.(r2c[clusterRows])
                if count(nz) > 0
                    λ = minimum(rowCosts[clusterRows][nz])
                else
                    λ = 0.0
                end
            else
                λ = 0.0
            end
            
            εfinal = margin / min(rct, cct)
            λ = max(λ, 0.0)
            if verbose
                println("Cluster: $kk of $maxLabel")
            end

            r2c[clusterRows], c2r[clusterCols], rowCosts[clusterRows], colCosts[clusterCols], λ = scaling_forward_backward(full(weightMat[clusterRows, clusterCols]), ε0, εfinal, εscale, check = true)

            #Map from subset rows / columns to actual
            for ii in clusterRows
                if !iszero(r2c[ii])
                    r2c[ii] = clusterCols[r2c[ii]]
                end
            end
            for jj in clusterCols
                if !iszero(c2r[jj])
                    c2r[jj] = clusterRows[c2r[jj]]
                end
            end
            
        end
    end
    
    mrows = Int[]
    for (ii, jj) in enumerate(IndexLinear(), r2c)
        if jj != 0 && (weightMat[ii, jj] > 0.0)
            push!(mrows, ii)
        end
    end
    mcols = r2c[mrows]
    return mrows, mcols, r2c, c2r, rowCosts, colCosts, w, margin
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
function max_C_sparseauction_cluster(pM::Array{T, 1},
                                     pU::Array{T, 1},
                                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                     penalty::AbstractFloat = 0.0,
                                     εscale::AbstractFloat = 0.2;
                                     verbose::Bool = false) where T <: AbstractFloat
    
    ##Compute weights
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = weights_matrix(w, compsum)
    
    ##Run clustering algorithm to split LSAP
    #aboveThreshold = indicator_weights_matrix(pM, pU, compsum, penalty)
    if typeof(compsum) <: ComparisonSummary
        rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat, penalty)
    elseif typeof(compsum) <: SparseComparisonSummary
        rowLabels, colLabels, maxLabel = bipartite_cluster_sparseblock(weightMat, penalty)
    else
        error("Unexpected type for compsum")
    end

    ##Find cluster indexes
    rowCounts = counts(rowLabels) #off by one because zero is included as a label
    colCounts = counts(colLabels) #off by one because zero is included as a label
    rowperm = sortperm(rowLabels)
    colperm = sortperm(colLabels)
    
    #Determine error levels for complete assignment
    margin = minimum_margin(w)
    ε0 = 0.5 * margin
    
    ##Loop over clusters
    r2c = zeros(Int64, compsum.nrow)
    c2r = zeros(Int64, compsum.ncol)
    rowCosts = zeros(T, compsum.nrow)
    colCosts = zeros(T, compsum.ncol)
    rowstart = rowCounts[1] + 1
    colstart = colCounts[1] + 1
    for kk in 1:maxLabel

        #Define cluster quantities
        rct = rowCounts[kk + 1]
        cct = colCounts[kk + 1]
        rrng = range(rowstart, rct)
        crng = range(colstart, cct)
        clusterRows = rowperm[rrng]
        clusterCols = colperm[crng]
        rowstart += rct
        colstart += cct
        
        #Set precision for cluster
        εfinal = margin / min(rct, cct)
        
        if verbose
            println("Cluster: $kk of $maxLabel")
        end

        if rct == 1 && cct == 1
            r2c[clusterRows[1]] = clusterCols[1]
            c2r[clusterCols[1]] = clusterRows[1]
            rowCosts[clusterRows[1]] = weightMat[clusterRows[1], clusterCols[1]] - margin
            colCosts[clusterCols[1]] = margin
        else
        
            #Run auction algorithm
            rW = sparse(weightMat[clusterRows, clusterCols])
            rW = add_dummy_entries(rW)
            r2c[clusterRows], c2r[clusterCols], rowCosts[clusterRows], colCosts[clusterCols], λ = asymmetric_scaling_forward_backward(rW, ε0, εfinal, εscale, check = true)

            #Map from subset rows / columns to actual
            for ii in clusterRows
                if !iszero(r2c[ii])
                    r2c[ii] = clusterCols[r2c[ii]]
                end
            end
            for jj in clusterCols
                if !iszero(c2r[jj])
                    c2r[jj] = clusterRows[c2r[jj]]
                end
            end
        end
    end
    
    mrows = Int[]
    for (ii, jj) in enumerate(IndexLinear(), r2c)
        if jj != 0 && (weightMat[ii, jj] > 0.0)
            push!(mrows, ii)
        end
    end
    mcols = r2c[mrows]
    return mrows, mcols, r2c, c2r, rowCosts, colCosts, w, margin
end
