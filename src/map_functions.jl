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
function map_solver_auction_cluster(pM0::Array{G, 1},
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

    mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_auction_cluster(pM0, pU0, compsum, penalty, εscale, verbose = verbose)
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
        mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_auction_cluster(pM, pU, compsum, r2c, c2r, rowCosts, colCosts, prevw, prevmargin, penalty, εscale, verbose = verbose)

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
