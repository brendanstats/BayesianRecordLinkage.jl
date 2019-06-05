"""

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

"""
function pair2idx(row::G, col::G, nrow::Integer) where G <: Integer
    return (col - one(G)) * nrow + row
end

"""

"""
function sample_proposal_full(rows::Array{G, 1}, cols::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}
    idx = sample(Weights(exp.(logweights .- logsumweights)))
    ridx, cidx = idx2pair(idx, length(rows))
    return rows[ridx], cols[cidx], logweights[idx]
end

"""

"""
function sample_proposal_sparse(rows::Array{G, 1}, cols::Array{G, 1}, logweights::Array{T, 1}, logsumweights::T = logsumexp(logweights)) where {G <: Integer, T <: AbstractFloat}
    idx = sample(Weights(exp.(logweights .- logsumweights)))
    return rows[idx], cols[idx], logweights[idx]
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
get_loglik(row::G, col::G, compsum::ComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf) where {G <: Integer, T <: AbstractFloat} = loglikRatios[compsum.obsidx[row, col]]

function get_loglik(row::G, col::G, compsum::SparseComparisonSummary, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    if iszero(compsum.obsidx[row, col])
        return loglikMissing
    else
        return loglikRatios[compsum.obsidx[row, col]]
    end
end

"""

"""
function loglik_add(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""

"""
function loglik_remove(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    loglikratio = -get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""

"""
function loglik_rowswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing) - get_loglik(C.col2row[col], col, compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""

"""
function loglik_colswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    loglikratio = get_loglik(row, col, compsum, loglikRatios) - get_loglik(row, C.row2col[row], compsum, loglikRatios, loglikMissing)
    return loglikratio
end

"""

"""
function loglik_doubleswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf)  where {G <: Integer, T <: AbstractFloat}
    rowalt = C.col2row[col]
    colalt = C.row2col[row]
    loglikratio = get_loglik(row, col, compsum, loglikRatios, loglikMissing)
    loglikratio += get_loglik(rowalt, colalt, compsum, loglikRatios, loglikMissing)
    loglikratio -= get_loglik(row, colalt, compsum, loglikRatios, loglikMissing)
    loglikratio -= get_loglik(rowalt, col, compsum, loglikRatios, loglikMissing)
    return loglikratio
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
get_counts(row::G, col::G, compsum::ComparisonSummary, obsidxCounts::Array{T, 2}) where {G <: Integer, T <: Integer} = obsidxCounts[:, compsum.obsidx[row, col]]


function get_counts(row::G, col::G, compsum::SparseComparisonSummary, obsidxCounts::Array{T, 2}) where {G <: Integer, T <: Integer}
    if iszero(compsum.obsidx[row, col])
        return zeros(T, size(obsidxCounts, 1))
    else
        return obsidxCounts[:, compsum.obsidx[row, col]]
    end
end

"""

"""
function counts_add(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{T, 2})  where {G <: Integer, T <: Integer}
    countsdelta = get_counts(row, col, compsum, obsidxCounts)
    return countsdelta
end

function counts_remove(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{T, 2})  where {G <: Integer, T <: Integer}
    countsdelta = -get_counts(row, col, compsum, obsidxCounts)
    return countsdelta
end

function counts_rowswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{T, 2})  where {G <: Integer, T <: Integer}
    countsdelta = get_counts(row, col, compsum, obsidxCounts) - get_counts(C.col2row[col], col, compsum, obsidxCounts)
    return countsdelta
end

function counts_colswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{T, 2})  where {G <: Integer, T <: Integer}
    countsdelta = get_counts(row, col, compsum, obsidxCounts) - get_counts(row, C.row2col[row], compsum, obsidxCounts)
    return countsdelta
end

function counts_doubleswitch(row::G, col::G, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, obsidxCounts::Array{T, 2})  where {G <: Integer, T <: Integer} 
    rowalt = C.col2row[col]
    colalt = C.row2col[row]
    countsdelta = get_counts(row, col, compsum, obsidxCounts)
    countsdelta += get_counts(rowalt, colalt, compsum, obsidxCounts)
    countsdelta -= get_counts(row, colalt, compsum, obsidxCounts)
    countsdelta -= get_counts(rowalt, col, compsum, obsidxCounts)
    return countsdelta
end

#uses a margin, eventually expand to more general form
function logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    return logpCRatio(one(G), C)
end

function logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Function) where G <: Integer
    return logpCRatio(-one(G), C)
end

function logpCRatios_add(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    if C.nlink == length(logpCRatio)
        return -T(Inf)
    else
        return logpCRatio[C.nlink + one(G)]
    end
end

function logpCRatios_remove(C::LinkMatrix{G}, logpCRatio::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    if iszero(C.nlink)
        return -T(Inf)
    else
        return -logpCRatio[C.nlink]
    end
end

"""
    locbal_kernel_move!(row, column, LinkMatrix) -> MovedLinkMatrix

Performes a move as described in Zanella (2017) appendix C
"""
function randomwalk1_move!(row::T, col::T, C::LinkMatrix{T}) where T <: Integer
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
    locbal_kernel_move!(row, column, LinkMatrix) -> MovedLinkMatrix

Performes a move as described in Zanella (2017) appendix C
"""
function randomwalk1_loglikpCratio(row::Integer, col::Integer, C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary}, logpCRatioAdd::T, logpCRatioRemove::T, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf) where T <: AbstractFloat
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
    locbal_kernel_move!(row, column, LinkMatrix) -> MovedLinkMatrix

Performes a move as described in Zanella (2017) appendix C
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

function randomwalk1_log_movecount(nrow::Integer, ncol::Integer, nlink::Integer)
    return log(nrow * ncol - div(nlink * (nlink - 1), 2))
end

function randomwalk1_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                             C::LinkMatrix{G},
                             compsum::Union{ComparisonSummary, SparseComparisonSummary},
                             obsidxCounts::Array{A, 2},
                             loglikRatios::Array{T, 1},
                             logpCRatio::Union{Function, Array{T, 1}},
                             log_balance_function::Function = lidentity_balance,
                             loglikMissing::T = -Inf) where {G <: Integer, T <: AbstractFloat, A <: Integer}
    
    row = sample(rows)
    col = sample(cols)

    #Resample if missing recordpair sampled
    if iszero(compsum.obsidx[row, col])
        return randomwalk1_update!(rows, cols, C, compsum, obsidxCounts, loglikRatios, logpCRatio, log_balance_function, loglikMissing)
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
                return randomwalk1_update!(rows, cols, C, compsum, obsidxCounts, loglikRatios, logpCRatio, log_balance_function, loglikMissing)
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

"""
function randomwalk1_log_move_weights(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                          C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          logpCRatioAdd::T, logpCRatioRemove::T,
                          loglikRatios::Array{T, 1},
                          log_balance_function::Function = lidentity_balance,
                          loglikMissing::T = -Inf) where T <: AbstractFloat
    
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

function randomwalk1_log_move_weights_sparse(cols::Array{<:Integer, 1}, nzobs::Integer,
                                 C::LinkMatrix, compsum::SparseComparisonSummary,
                                 logpCRatioAdd::T, logpCRatioRemove::T,
                                 loglikRatios::Array{T, 1},
                                 log_balance_function::Function = lidentity_balance,
                                 loglikMissing::T = -Inf) where T <: AbstractFloat
    
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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function randomwalk1_locally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                loglikRatios::Array{T, 1},
                                obsidxCounts::Array{G, 2},
                                logpCRatio::Union{Function, Array{T, 1}}, 
                                log_balance_function::Function = lidentity_balance,
                                loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}

    ##Move weights
    logpCRatioAdd = logpCRatios_add(C, logpCRatio)
    logpCRatioRemove = logpCRatios_remove(C, logpCRatio)  
    lmoveweights, lsummw = randomwalk1_log_move_weights(rows, cols, C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, log_balance_function, loglikMissing)
    
    #randomwalk1_loglikpCratio(rows[ii], cols[jj], C, compsum, logpCRatioAdd, logpCRatioRemove, loglikRatios, loglikMissing)
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

    #rowcheck, colcheck = randomwalk1_inverse(invrow, invcol, C)
    #if rowcheck != moverow || colcheck != movecol
    #    @error "check inverse move function"
    #    println("($moverow, $movecol)")
    #    println("($invrow, $invcol)")
    #    println("($rowcheck, $colcheck)")
    #    println("$(C.row2col)")
    #    println("$doubleswitch")
    #end
    
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

"""
function randomwalk1_globally_balanced_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                 C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 loglikRatios::Array{T, 1},
                                 obsidxCounts::Array{G, 2},
                                 logpCRatio::Union{Function, Array{T, 1}},
                                 loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lidentity_balance, loglikMissing)
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
function randomwalk1_locally_balanced_sqrt_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                 C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 loglikRatios::Array{T, 1},
                                 obsidxCounts::Array{G, 2},
                                 logpCRatio::Union{Function, Array{T, 1}},
                                 loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lsqrt_balance, loglikMissing)
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
function randomwalk1_locally_balanced_barker_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                                                     C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                     loglikRatios::Array{T, 1},
                                                     obsidxCounts::Array{G, 2},
                                                     logpCRatio::Union{Function, Array{T, 1}},
                                                     loglikMissing = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    return randomwalk1_locally_balanced_update!(rows, cols, C, compsum, loglikRatios, obsidxCounts, logpCRatio, lbarker_balance, loglikMissing)
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
function randomwalk2_move!(row::T, col::T, C::LinkMatrix{T}) where T <: Integer
    
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

function randomwalk2_loglikpCratio(row::T, col::T, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, logpCRatioAdd::T, logpCRatioRemove::T, loglikRatios::Array{T, 1}, loglikMissing::T = -Inf) where T <: Integer
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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function randomwalk2_countsdelta(row::T, col::T,
                                 C::LinkMatrix{T},
                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 obsidxCounts::Array{G, 2}) where {T <: Integer, G <: Integer}
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return get_counts(row, col, compsum, obsidxCounts)
        else ##single switch move I
            warn("incorrect draw for randomwalk 2")
            return countsdelta
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return get_counts(row, col, compsum, obsidxCounts) - get_counts(row, C.row2col[row], compsum, obsidxCounts)
        elseif C.col2row[col] == row ##delete move
            return -get_counts(row, col, compsum, obsidxCounts)
        else ##double switch move
            warn("incorrect draw for randomwalk 2 (double switch)")
            return zeros(G, size(obsidxCounts, 1))
        end
    end
end

function randomwalk2_update!(rows::Array{<:Integer, 1}, cols::Array{<:Integer, 1},
                             C::LinkMatrix,
                             compsum::Union{ComparisonSummary, SparseComparisonSummary},
                             obsidxCounts::Array{A, 2},
                             loglikRatios::Array{T, 1},
                             logpCRatio::Union{Function, Array{T, 1}},
                             p::AbstractFloat = 0.5,
                             log_balance_function::Function = lidentity_balance,
                             loglikMissing::T = -Inf) where {G <: Integer, T <: AbstractFloat, A <: Integer}

    row = sample(rows)
    opencols = cols[findall(iszero, C.col2row[cols])]
    
    if iszero(C.row2col[row]) #add move
        if length(opencols) == 0 #this should only happen if ncol < nrow in which case a different kernel should be used
            return randomwalk2_update!(rows, cols, C, compsum, obsidxCounts, loklikMargin, logpCRatio, p, log_balance_function, loglikMissing)
        end
        col = sample(opencols)
        loglik = loglik_add(row, col, C, compsum, loglikRatios, loglikMissing) + logpCRatios_add(C, logpCRatio)
        if length(opencols) == 1
            #lmoveratio = zero(T)
        else
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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function singleton_gibbs!(row::Integer, col::Integer, C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikRatios::Array{T, 1}, obsidxCounts::Array{G, 2},
                          logpCRatio::Function,
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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function singlerow_gibbs(row::Integer, cols::Array{<:Integer}, C::LinkMatrix,
                         compsum::Union{ComparisonSummary, SparseComparisonSummary},
                         loglikRatios::Array{<:AbstractFloat, 1}, obsidxCounts::Array{<:Integer, 2},
                         logpCRatio::Function)
    if iszero(C.row2col[row])
        lpriorratio0 = logpCRatio(0, C)
        lpriorratio1 = logpCRatio(1, C)
    else
        lpriorratio0 = logpCRatio(-1, C)
        lpriorratio1 = logpCRatio(0, C)
    end
    loglikes = Array{Float64}(undef, length(cols) + 1)
    loglikes[1] = lpriorratio0
    for (jj, col) in enumerate(cols)
        loglikes[jj + 1] = loglikRatios[compsum.obsidx[row, col]] + lpriorratio1
    end
    idx = searchsortedfirst(rand(), cumsum(softmax!(loglikes)))
    if idx == 1
        if iszero(C.row2col[row])
            return zeros(eltype(obsidxCounts), size(obsidxCounts, 1)), false
        else
            remove_link!(row, C.row2col[row], C)
        end
    else
        #if idx == 1
        #elseif #equals chosen index
        #else
        #end
    end
end

"""
Compute the conditional dirchlet distribution for updating M and U parameters via a gibbs step
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

function gibbs_MU_draw(matchcounts::Array{<:Integer, 1},
                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                       obsidxCounts::Array{<:Integer, 2} = counts_delta(compsum),
                       priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                       priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikRatios = obsidxCounts' * logDiff
    return pM, pU, loglikRatios
end
