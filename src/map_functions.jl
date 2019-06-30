function penalized_likelihood_hungarian(pM0::Array{G, 1}, pU0::Array{G, 1},
                                        compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                        priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                        priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                        penalty::AbstractFloat = G(0.0);
                                        tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                        cluster::Bool = true, verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    if cluster
        astate = max_C_cluster_hungarian(wpenalized, compsum)
    else
        astate = max_C_hungarian(wpenalized, compsum)
    end
    
    
    prevM = copy(pM0)
    prevU = copy(pU0)
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        if cluster
            astate = max_C_cluster_hungarian(wpenalized, compsum)
        else
            astate = max_C_hungarian(wpenalized, compsum)
        end
        
        
        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

function penalized_likelihood_cluster_hungarian(pM0::Array{G, 1}, pU0::Array{G, 1},
                                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                                priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                                penalty::AbstractFloat = G(0.0);
                                                tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                                verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    astate = max_C_cluster_hungarian(wpenalized, compsum)

    prevM = copy(pM0)
    prevU = copy(pU0)
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        astate = max_C_cluster_hungarian(wpenalized, compsum)

        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

function penalized_likelihood_auction(pM0::Array{G, 1}, pU0::Array{G, 1},
                                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                      priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                      priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                      penalty::AbstractFloat = G(0.0);
                                      epsiscale::T = T(0.2), minmargin::T = zero(T), digt::Integer = 5,
                                      tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                      cluster::Bool = true, update::Bool = true, verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    atol = minimum_margin(wpenalized, minmargin, digt)
    if cluster
        astate = max_C_cluster_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
    else
        astate = max_C_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
    end

    if update
        lambda = min_assigned_colprice(astate)
        prevW = wpenalized
    end
    
    prevM = copy(pM0)
    prevU = copy(pU0)
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        atol = minimum_margin(wpenalized, minmargin, digt)
        if update
            if cluster
                epsi0 = max(maximum(wpenalized - prevW), zero(T)) + atol
                clear_assignment!(astate)
                astate, lambda = max_C_cluster_auction!(astate, wpenalized, compsum, lambda, epsi0, atol, epsiscale = epsiscale)
            else
                epsi0 = max(maximum(wpenalized - prevW), zero(T)) + (atol / compsum.nrow)
                clear_assignment!(astate)
                
                astate, lambda = max_C_auction!(astate, lambda, epsi0, atol, wpenalized, compsum, epsiscale = epsiscale)
            end
        else
            if cluster
                astate = max_C_cluster_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
            else
                astate = max_C_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
            end
        end

        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        if update
            prevW = wpenalized
        end
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

function penalized_likelihood_cluster_auction(pM0::Array{G, 1}, pU0::Array{G, 1},
                                              compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                              priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                              priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                              penalty::AbstractFloat = G(0.0);
                                              epsiscale::T = T(0.2), minmargin::T = zero(T), digt::Integer = 5,
                                              tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                              verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    atol = minimum_margin(wpenalized, minmargin, digt)
    astate = max_C_cluster_auction(wpenalized, atol, compsum, epsiscale = epsiscale)

    prevM = copy(pM0)
    prevU = copy(pU0)
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        atol = minimum_margin(wpenalized, minmargin, digt)
        astate = max_C_cluster_auction(wpenalized, atol, compsum, epsiscale = epsiscale)

        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

function penalized_likelihood_auction_update(pM0::Array{G, 1}, pU0::Array{G, 1},
                                             compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                             priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                             priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                             penalty::AbstractFloat = G(0.0);
                                             epsiscale::T = T(0.2), minmargin::T = zero(T), digt::Integer = 5,
                                             tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                             verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    atol = minimum_margin(wpenalized, minmargin, digt)
    astate = max_C_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
    lambda = min_assigned_colprice(astate)
    
    prevM = pM0
    prevU = pU0
    prevW = wpenalized
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        atol = minimum_margin(wpenalized, minmargin, digt)
        epsi0 = max(maximum(wpenalized - prevW), zero(T)) + (atol / compsum.nrow)
        clear_assignment!(astate)
        astate, lambda = max_C_auction!(astate, lambda, epsi0, atol, wpenalized, compsum, epsiscale = epsiscale) #issue here

        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        prevW = wpenalized
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

function penalized_likelihood_cluster_auction_update(pM0::Array{G, 1}, pU0::Array{G, 1},
                                                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                     priorM::Array{T, 1} = ones(Int, length(compsum.counts)),
                                                     priorU::Array{T, 1} = ones(Int, length(compsum.counts)),
                                                     penalty::AbstractFloat = G(0.0);
                                                     epsiscale::T = T(0.2), minmargin::T = zero(T), digt::Integer = 5,
                                                     tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                                     verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty)
    atol = minimum_margin(wpenalized, minmargin, digt)
    astate = max_C_cluster_auction(wpenalized, atol, compsum, epsiscale = epsiscale)
    lambda = min_assigned_colprice(astate)
    
    prevM = pM0
    prevU = pU0
    prevW = wpenalized
    iter = 0
    while iter < maxIter

        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)
        if all(abs.(pM - prevM) .<= tol) && all(abs.(pU - prevU) .<= tol)
            return astate, pM, pU, iter
        end

        wpenalized = penalized_weights_vector(pM, pU, compsum, penalty)
        atol = minimum_margin(wpenalized, minmargin, digt)
        epsi0 = max(maximum(wpenalized - prevW), zero(T)) + atol
        clear_assignment!(astate)
        astate, lambda = max_C_cluster_auction!(astate, wpenalized, compsum, lambda, epsi0, atol, epsiscale = epsiscale)

        iter += 1
        if verbose
            println("Iteration: $iter")
            println("pM: $prevM")
            nmatch = astate.nassigned
            println("Matches: $nmatch")
        end
        
        prevM = pM
        prevU = pU
        prevW = wpenalized
        
    end
    @warn "Maximum number of iterations reached"
    return astate, prevM, prevU, iter
end

"""
    map_solver(pM0, pU0, comparisonSummary, [priorM], [priorU], penalty; maxIter) -> matchRows, matchColumns, pM, pU, iterations
"""
function map_solver(pM0::Array{G, 1},
                    pU0::Array{G, 1},
                    compsum::Union{ComparisonSummary, SparseComparisonSummary},
                    priorM::Array{T, 1} = ones(T, length(compsum.counts)),
                    priorU::Array{T, 1} = ones(T, length(compsum.counts)),
                    penalty::AbstractFloat = 0.0;
                    maxIter::Integer = 100,
                    verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    row2col = max_C(pM0, pU0, compsum, penalty)
    matchcounts, obscounts = counts_matches(row2col, compsum)
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
        pM, pU = max_MU(row2col, compsum, pseudoM, pseudoU)
        newrow2col = max_C(pM, pU, compsum, penalty)
        newmatchcounts, newobscounts = counts_matches(newrow2col, compsum)
        if obscounts == newobscounts && matchcounts == newmatchcounts
            return row2col, pM, pU, iter
        end
        row2col = newrow2col
        matchcounts = newmatchcounts
        obscounts = newobscounts
    end
    @warn "Maximum number of iterations reached"
    return row2col, pM, pU, iter
end

"""

"""
function map_solver_cluster(pM0::Array{G, 1},
                            pU0::Array{G, 1},
                            compsum::Union{ComparisonSummary, SparseComparisonSummary},
                            priorM::Array{T, 1},
                            priorU::Array{T, 1},
                            penalty::AbstractFloat = 0.0;
                            maxIter::Integer = 100,
                            verbose::Bool = false) where {G <: AbstractFloat, T <: Real}

    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    currmrows, currmcols = max_C_cluster(pM0, pU0, compsum, penalty, verbose = verbose)
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
    @warn "Maximum number of iterations reached"
    return currmrows, currmcols, pM, pU, iter
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
function map_solver_auction(pM0::Array{G, 1},
                            pU0::Array{G, 1},
                            compsum::Union{ComparisonSummary, SparseComparisonSummary},
                            priorM::Array{T, 1} = ones(T, length(compsum.counts)),
                            priorU::Array{T, 1} = ones(T, length(compsum.counts)),
                            penalty::AbstractFloat = 0.0,
                            εscale::T = 0.2;
                            maxIter::Integer = 100,
                            verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    mrows, mcols, rowCosts, colCosts, prevw, prevε, λ = max_C_auction(pM0, pU0, compsum, penalty, εscale)
    pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    if pM == pM0 && pU == pU0
        return mrows, mcols, pM, pU, 1
    end

    #track match counts for convergence
    prevcounts = counts_matches(mrows, mcols, compsum)[1]
    
    iter = 0
    while iter < maxIter
        iter += 1
        if verbose
            println("Iteration: $iter, Matches: $(length(mrows))")
        end

        #solve new problem
        mrows, mcols, rowCosts, colCosts, prevw, prevε, λ = max_C_auction(pM, pU, compsum, rowCosts, colCosts, prevw, prevε, λ, penalty, εscale)

        #check convergence
        matchcounts = counts_matches(mrows, mcols, compsum)[1]
        if prevcounts == matchcounts
            return mrows, mcols, pM, pU, iter
        end
        prevcounts = matchcounts

        #update parameters
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
        
    end
    @warn "Maximum number of iterations reached"
    return mrows, mcols, pM, pU, iter
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
function map_solver_cluster_auction(pM0::Array{G, 1},
                                    pU0::Array{G, 1},
                                    compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                    priorM::Array{T, 1} = ones(T, length(compsum.counts)),
                                    priorU::Array{T, 1} = ones(T, length(compsum.counts)),
                                    penalty::AbstractFloat = 0.0,
                                    εscale::T = 0.2;
                                    maxIter::Integer = 100,
                                    verbose::Bool = false) where {G <: AbstractFloat, T <: Real}
    ##Modes are found using pseudo counts of 1 - αᵢ
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_cluster_auction(pM0, pU0, compsum, penalty, εscale, verbose = verbose)
    pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    if pM == pM0 && pU == pU0
        return mrows, mcols, pM, pU, 1
    end

    #track match counts for convergence
    prevcounts = counts_matches(mrows, mcols, compsum)[1]
    
    iter = 0
    while iter < maxIter
        iter += 1
        if verbose
            println("Iteration: $iter, Matches: $(length(mrows))")
            println("prevcounts: $prevcounts")
        end
        
        #solve new problem
        mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_cluster_auction(pM, pU, compsum, r2c, c2r, rowCosts, colCosts, prevw, prevmargin, penalty, εscale, verbose = verbose)

        #check convergence
        matchcounts = counts_matches(mrows, mcols, compsum)[1]
        if prevcounts == matchcounts
            return mrows, mcols, pM, pU, iter
        end
        prevcounts = matchcounts

        #update parameters
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
        
    end
    @warn "Maximum number of iterations reached"
    return mrows, mcols, pM, pU, iter
end
