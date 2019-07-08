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

#can also be used inside cluster based solver
function max_C_hungarian(weightMat::SparseMatrixCSC{T}, maxw::T) where T <: AbstractFloat
    cost = reward2cost(Matrix(weightMat), maxw)
    astate = hungarian_assignment(cost)
    for (ii, jj) in pairs(astate.r2c)
        if !iszero(jj) && iszero(weightMat[ii, jj])
            astate.r2c[ii] = 0
            astate.c2r[jj] = 0
            astate.nassigned -= 1
        end
    end
    return astate
end

#wrapper for when clustering not used
function max_C_hungarian(wpenalized::Array{<:AbstractFloat, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary})
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    return max_C_hungarian(weightMat, maximum(wpenalized))
end

#recycles prices, can also be used inside cluster based solver
function max_C_auction!(astate::AssignmentState, weightMat::SparseMatrixCSC{T}, lambda::T, epsi0::T, epsitol::T; epsiscale::T = T(0.2)) where T <: AbstractFloat
    astate, lambda = auction_assignment_padasfr1(weightMat, astate = astate, lambda0 = lambda, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale, dfltReward = zero(T), dfltTwo = -T(Inf))
    adjust_inf!(astate, lambda)
    remove_padded!(astate, size(weightMat, 2))
    return astate, lambda
end

#wrapper for when clustering not used but prices are recycled
function max_C_auction!(astate::AssignmentState, lambda::T, epsi0::T, tol::T, wpenalized::Array{T, 1},
                        compsum::Union{ComparisonSummary, SparseComparisonSummary}; epsiscale::T = T(0.2)) where T <: AbstractFloat
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    for jj in 1:size(weightMat, 2)
        if length(nzrange(weightMat, jj)) == 0
            astate.colPrices[jj] = zero(T)
        end
    end
    epsitol = tol / compsum.nrow
    return max_C_auction!(astate, weightMat, lambda, epsi0, epsitol, epsiscale = epsiscale)
end

#wrapper for when clustering is used but prices are not recycled
function max_C_auction(weightMat::SparseMatrixCSC{T}, epsi0::T, epsitol; epsiscale::T = T(0.2)) where T <: AbstractFloat
    astate = AssignmentState(weightMat, maximize = true, assign = true, pad = true)
    return max_C_auction!(astate, weightMat, zero(T), epsi0, epsitol, epsiscale = epsiscale)[1]
end

#wrapper for when clustering not used and prices are not recycled
function max_C_auction(wpenalized::Array{T, 1}, tol::T, compsum::Union{ComparisonSummary, SparseComparisonSummary}; epsiscale::T = T(0.2)) where T <: AbstractFloat
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    epsi0 = maximum(weightMat) * epsiscale
    epsitol = tol / compsum.nrow
    return max_C_auction(weightMat, epsi0, epsitol, epsiscale = epsiscale)
end

function max_C_cluster_setup(wpenalized::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <:AbstractFloat
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat, zero(T))
    cc = ConnectedComponents(rowLabels, colLabels, maxLabel)

    maxw = maximum(wpenalized)
    component2rows = label2dict(cc.rowLabels)
    component2cols = label2dict(cc.colLabels)
    return weightMat, maxw, maxLabel, component2rows, component2cols
end

#hungarian algorithm using clustering
function max_C_cluster_hungarian(wpenalized::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where T <:AbstractFloat

    weightMat, maxw, maxLabel, component2rows, component2cols = max_C_cluster_setup(wpenalized, compsum)
    
    astate = AssignmentState(zeros(T, compsum.nrow), zeros(T, compsum.nrow + compsum.ncol))
    for kk in 1:maxLabel
        
        #get component assignment
        rows = component2rows[kk]
        cols = component2cols[kk]

        if length(rows) == 1
            if length(cols) == 1
                astate.r2c[rows[1]] = cols[1]
                astate.c2r[cols[1]] = rows[1]
                astate.nassigned += 1
            else
                reward, cidx = findmax(weightMat[rows[1], cols])
                astate.r2c[rows[1]] = cols[cidx]
                astate.c2r[cols[cidx]] = rows[1]
                astate.nassigned += 1
            end
        else
            if length(cols) == 1
                reward, ridx = findmax(weightMat[rows, cols[1]])
                astate.r2c[rows[ridx]] = cols[1]
                astate.c2r[cols[1]] = rows[ridx]
                astate.nassigned += 1
            else
                
                astatecomp = max_C_hungarian(weightMat[rows, cols], maxw)
                
                #add component assignments to astate
                astate.r2c[rows] = [iszero(cidx) ? zero(cidx) : cols[cidx] for cidx in astatecomp.r2c]
                astate.c2r[cols] = [iszero(ridx) ? zero(ridx) : rows[ridx] for ridx in astatecomp.c2r]
                astate.nassigned += astatecomp.nassigned
            end
        end
    end
    
    return astate
end

#auction algorithm using clustering without recycled prices
function max_C_cluster_auction(wpenalized::Array{T, 1}, tol::T, compsum::Union{ComparisonSummary, SparseComparisonSummary}; epsiscale::T = T(0.2)) where T <:AbstractFloat

    weightMat, maxw, maxLabel, component2rows, component2cols = max_C_cluster_setup(wpenalized, compsum)
    epsi0 = maxw * epsiscale
    
    astate = AssignmentState(zeros(T, compsum.nrow), zeros(T, compsum.nrow + compsum.ncol))
    for kk in 1:maxLabel
        
        #get component assignment
        rows = component2rows[kk]
        cols = component2cols[kk]
        padcols = [cols; rows .+ compsum.ncol]
        
        if length(rows) == 1 && length(cols) == 1
            astate.r2c[rows[1]] = cols[1]
            astate.c2r[cols[1]] = rows[1]
            astate.rowPrices[rows[1]] = weightMat[rows[1], cols[1]] #assuming padded cost of zero and lambda of 0, starting prices of 0
            astate.nassigned += 1
        else
            epsitol = tol / length(rows)
            astatecomp = max_C_auction(weightMat[rows, cols], max(epsi0, epsitol), epsitol, epsiscale = epsiscale)
            
            #add component assignments to astate
            astate.r2c[rows] = [iszero(cidx) ? zero(cidx) : cols[cidx] for cidx in astatecomp.r2c]
            astate.c2r[cols] = [iszero(ridx) ? zero(ridx) : rows[ridx] for ridx in astatecomp.c2r[1:length(cols)]]
            astate.rowPrices[rows] = astatecomp.rowPrices
            astate.colPrices[padcols] = astatecomp.colPrices
            astate.nassigned += astatecomp.nassigned
        end
    end
    
    return astate
end

#auction algorithm using clustering and recycled prices - supplied astate should be cleared only price information is used
function max_C_cluster_auction!(astate::AssignmentState, wpenalized::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                lambda::T, epsi0::T, tol::T; epsiscale::T = T(0.2)) where T <:AbstractFloat

    weightMat, maxw, maxLabel, component2rows, component2cols = max_C_cluster_setup(wpenalized, compsum)
    epsi0 = maxw * epsiscale
    
    lambdaout = T(Inf)
    for kk in 1:maxLabel
        
        #get component assignment
        rows = component2rows[kk]
        cols = component2cols[kk]
        padcols = [cols; rows .+ compsum.ncol]

        if length(rows) == 1 && length(cols) == 1 #only look at this case here because of managing prices for padded columns
            astate.r2c[rows[1]] = cols[1]
            astate.c2r[cols[1]] = rows[1]
            astate.colPrices[padcols[1]] = lambda
            astate.rowPrices[rows[1]] = weightMat[rows[1], cols[1]] - lambda
            if astate.colPrices[padcols[2]] > max(astate.rowPrices[rows[1]], lambda) #assuming padded cost of zero
                astate.colPrices[padcols[1]] = min(-astate.rowPrices[rows[1]], lambda) 
            end
            astate.nassigned += 1
        else
            epsitol = tol / length(rows)
            astatecomp = AssignmentState(astate.rowPrices[rows], astate.colPrices[padcols])
            astatecomp, lambdacomp = max_C_auction!(astatecomp, weightMat[rows, cols], lambda, max(epsi0, epsitol), epsitol, epsiscale = epsiscale)
            
            #add component assignments to astate
            astate.r2c[rows] = [iszero(cidx) ? zero(cidx) : cols[cidx] for cidx in astatecomp.r2c]
            astate.c2r[cols] = [iszero(ridx) ? zero(ridx) : rows[ridx] for ridx in astatecomp.c2r[1:length(cols)]]
            astate.rowPrices[rows] = astatecomp.rowPrices
            astate.colPrices[padcols] = astatecomp.colPrices
            astate.nassigned += astatecomp.nassigned
            
            if lambdacomp < lambdaout
                lambdaout = lambdacomp
            end
        end
    end
    
    return astate, lambda
end

"""
    max_C(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns
"""
function max_C(pM::Array{T, 1},
               pU::Array{T, 1},
               compsum::Union{ComparisonSummary, SparseComparisonSummary},
               penalty::AbstractFloat = 0.0;
               epsiscale::T = T(0.2),
               minmargin::T = zero(T), digt::Integer = 5) where T <: AbstractFloat

    wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    margin = minimum_margin(wpenalized, minmargin, digt)
    epsi0 = maximum(wpenalized) * epsiscale
    epsitol = margin / min(compsum.nrow, compsum.ncol)

    #Solve assignment problem
    astate = auction_assignment_padasfr2(weightMat, epsi0 = epsi0, epsitol = epsitol)[1]
    row2col = [col > compsum.ncol ? 0 : col for col in astate.r2c]
    return row2col
end

function max_C_cluster(pM::Array{T, 1},
                       pU::Array{T, 1},
                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                       penalty::AbstractFloat = 0.0;
                       epsiscale::T = T(0.2),
                       minmargin::T = zero(T), digt::Integer = 5,
                       verbose::Bool = false) where T <: AbstractFloat
    
    ##Run clustering algorithm to split LSAP
    wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = penalized_weights_matrix(wpenalized, compsum)
    rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat, 0.0)
    margin = minimum_margin(wpenalized, minmargin, digt)
    epsi0 = maximum(wpenalized) * epsiscale
    
    ##Mark clusters for rows
    block2rows = label2dict(rowLabels)
    block2cols = label2dict(colLabels)
    
    if verbose
        maxsize = maximum([length(block2rows[ii]) * length(block2cols[ii]) for ii in 1:maxLabel])
        println("Maximum Cluster Size: $maxsize")
    end
    
    ##Compute cost matrix
    row2col = zeros(Int64, compsum.nrow)
    for kk in 1:maxLabel

        if verbose
            println("Cluster: $kk of $maxLabel")
        end

        nrow = length(block2rows[kk])
        ncol = length(block2cols[kk])
        
        #Matrix should technically already be full here, assuming blocks
        if nrow == 1 && ncol == 1
            row2col[block2rows[kk][1]] = block2cols[kk][1]
        else
            epsitol = margin / min(nrow, ncol)
            
            #Solve assignment problem
            astate = auction_assignment_padasfr2(weightMat[block2rows[kk], block2cols[kk]], epsi0 = epsi0, epsitol = epsitol)[1]
            for ridx in 1:astate.nrow
                if !iszero(astate.r2c[ridx]) && astate.r2c[ridx] < ncol
                    row2col[block2rows[kk][ridx]] = block2cols[kk][astate.r2c[ridx]]
                end
            end
        end
    end
    
    return row2col
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
                       minmargin::T = zero(T), digt::Integer = 5) where T <: AbstractFloat
    #Compute weights
    #wpenalized = penalized_weights_matrix(pM, pU, compsum, penalty)
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    weightMat = weights_matrix(w, compsum)
    #lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1) 
    #Determine error levels for complete assignment
    margin = minimum_margin(w, minmargin, digt)
    if epsi0 < zero(T)
        epsi0 = maximum(w) * epsiscale
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
