get_loglik{G <: Integer, T <: AbstractFloat}(row::G, col::G, compsum::ComparisonSummary, loglikMargin::Array{T, 1}, loglikMissing::T = -10.0) = loglikMargin[compsum.obsidx[row, col]]

function get_loglik{G <: Integer, T <: AbstractFloat}(row::G, col::G, compsum::SparseComparisonSummary, loglikMargin::Array{T, 1}, loglikMissing::T = -10.0)
    if iszero(compsum.obsidx[row, col])
        return loglikMissing
    else
        return loglikMargin[compsum.obsidx[row, col]]
    end
end

get_counts{G <: Integer, T <: Integer}(row::G, col::G, compsum::ComparisonSummary, countDeltas::Array{T, 2}) = countDeltas[:, compsum.obsidx[row, col]]


function get_counts{G <: Integer, T <: Integer}(row::G, col::G, compsum::SparseComparisonSummary, countDeltas::Array{T, 2})
    if iszero(compsum.obsidx[row, col])
        return zeros(T, size(countDeltas, 1))
    else
        return countDeltas[:, compsum.obsidx[row, col]]
    end
end



"""
    locbal_kernel_move!(row, column, LinkMatrix) -> MovedLinkMatrix

Performes a move as described in Zanella (2017) appendix C
"""
function randomwalk1_move!{T <: Integer}(row::T, col::T, C::LinkMatrix{T})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            add_link!(row, col, C)
            return C
        else ##single switch move I
            C.row2col[C.col2row[col]] = zero(T)
            dropzeros!(C.row2col)
            C.col2row[col] = row
            C.row2col[row] = col
            return C
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            C.col2row[C.row2col[row]] = zero(T)
            dropzeros!(C.col2row)
            C.col2row[col] = row
            C.row2col[row] = col
            return C
        elseif C.col2row[col] == row ##delete move
            remove_link!(row, col, C)
            return C
        else ##double switch move
            #switch_link!(row, colalt, rowalt, col), test...)
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            C.col2row[colalt] = rowalt
            C.row2col[rowalt] = colalt
            C.col2row[col] = row
            C.row2col[row] = col
            return C
        end
    end
end

randomwalk1_move!{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}) = randomwalk1_move!(idx.I[1], idx.I[2], C)

randomwalk1_move{T <: Integer}(row::T, col::T, C::LinkMatrix{T}) = randomwalk1_move!(row, col, deepcopy(C))
randomwalk1_move{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}) = randomwalk1_move!(idx.I[1], idx.I[2], deepcopy(C))

function randomwalk1_movetype{T <: Integer}(row::T, col::T, C::LinkMatrix)
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            return 1
        else ##single switch move I
            return 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            return 3
        elseif C.col2row[col] == row ##delete move
            return 4
        else ##double switch move
            return 5
        end
    end
end

"""
    locbal_kernel_move!(row, column, LinkMatrix) -> MovedLinkMatrix

Performes a move as described in Zanella (2017) appendix C
"""
function randomwalk1_countdelta{T <: Integer}(row::T, col::T,
                                            C::LinkMatrix{T},
                                            compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                            countDeltas::Array{<:Integer, 2})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            countdelta = get_counts(row, col, compsum, countDeltas)
            return countdelta, 1
        else ##single switch move I
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(C.col2row[col], col, compsum, countDeltas)
            return countdelta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(row, C.row2col[row], compsum, countDeltas)
            return countdelta, 3
        elseif C.col2row[col] == row ##delete move
            countdelta = -get_counts(row, col, compsum, countDeltas)
            return countdelta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            countdelta = get_counts(row, col, compsum, countDeltas)
            countdelta += get_counts(rowalt, colalt, compsum, countDeltas)
            countdelta -= get_counts(row, colalt, compsum, countDeltas)
            countdelta -= get_counts(rowalt, col, compsum, countDeltas)
            return countdelta, 5
        end
    end
end

randomwalk1_countdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, countDeltas::Array{<:Integer, 2}) =
    randomwalk1_countdelta(idx.I[1], idx.I[2], C, compsum, countDeltas)

function randomwalk1_loglikdelta{T <: Integer}(row::T, col::T,
                                               C::LinkMatrix{T},
                                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                               loglikMargin::Array{<:AbstractFloat, 1})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            return loglikdelta, 1
        else ##single switch move I
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(C.col2row[col], col, compsum, loglikMargin)
            return loglikdelta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(row, C.row2col[row], compsum, loglikMargin)
            return loglikdelta, 3
        elseif C.col2row[col] == row ##delete move
            loglikdelta = -get_loglik(row, col, compsum, loglikMargin)
            return loglikdelta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            loglikdelta += get_loglik(rowalt, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(row, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(rowalt, col, compsum, loglikMargin)
            return loglikdelta, 5
        end
    end
end

randomwalk1_loglikdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikMargin::Array{<:AbstractFloat, 1}) =
    randomwalk1_loglikdelta(idx.I[1], idx.I[2], C, compsum, loglikMargin)

function randomwalk1_countobsdelta{T <: Integer}(row::T, col::T,
                                                 C::LinkMatrix{T},
                                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                 countDeltas::Array{<:Integer, 2},
                                                 obsDeltas::Array{<:Integer, 2})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            countdelta = get_counts(row, col, compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas)
            return countdelta, obsdelta, 1
        else ##single switch move I
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(C.col2row[col], col, compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas) - get_counts(C.col2row[col], col, compsum, obsDeltas)
            return countdelta, obsdelta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(row, C.row2col[row], compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas) - get_counts(row, C.row2col[row], compsum, obsDeltas)
            return countdelta, obsdelta, 3
        elseif C.col2row[col] == row ##delete move
            countdelta = -get_counts(row, col, compsum, countDeltas)
            obsdelta = -get_counts(row, col, compsum, obsDeltas)
            return countdelta, obsdelta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]

            countdelta = get_counts(row, col, compsum, countDeltas)
            countdelta += get_counts(rowalt, colalt, compsum, countDeltas)
            countdelta -= get_counts(row, colalt, compsum, countDeltas)
            countdelta -= get_counts(rowalt, col, compsum, countDeltas)
            
            obsdelta = get_counts(row, col, compsum, obsDeltas)
            obsdelta += get_counts(rowalt, colalt, compsum, obsDeltas)
            obsdelta -= get_counts(row, colalt, compsum, obsDeltas)
            obsdelta -= get_counts(rowalt, col, compsum, obsDeltas)
            
            return countdelta, obsdelta, 5
        end
    end
end

randomwalk1_countobsdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, countDeltas::Array{<:Integer, 2}, obsDeltas::Array{<:Integer, 2}) =
    randomwalk1_countobsdelta(idx.I[1], idx.I[2], C, compsum, countDeltas, obsDeltas)

function randomwalk1_loglikcountdelta{T <: Integer}(row::T, col::T,
                                                       C::LinkMatrix{T},
                                                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                       loglikMargin::Array{<:AbstractFloat, 1},
                                                       countDeltas::Array{<:Integer, 2})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas)
            return loglikdelta, countdelta, 1
        else ##single switch move I
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(C.col2row[col], col, compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(C.col2row[col], col, compsum, countDeltas)
            return loglikdelta, countdelta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(row, C.row2col[row], compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(row, C.row2col[row], compsum, countDeltas)
            return loglikdelta, countdelta, 3
        elseif C.col2row[col] == row ##delete move
            loglikdelta = -get_loglik(row, col, compsum, loglikMargin)
            countdelta = -get_counts(row, col, compsum, countDeltas)
            return loglikdelta, countdelta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            loglikdelta += get_loglik(rowalt, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(row, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(rowalt, col, compsum, loglikMargin)
            
            countdelta = get_counts(row, col, compsum, countDeltas)
            countdelta += get_counts(rowalt, colalt, compsum, countDeltas)
            countdelta -= get_counts(row, colalt, compsum, countDeltas)
            countdelta -= get_counts(rowalt, col, compsum, countDeltas)
            
            return loglikdelta, countdelta, 5
        end
    end
end

randomwalk1_loglikcountdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikMargin::Array{<:AbstractFloat, 1}, countDeltas::Array{<:Integer, 2}) =
    randomwalk1_loglikcountdelta(idx.I[1], idx.I[2], C, compsum, loglikMargin, countDeltas)

function randomwalk1_loglikcountobsdelta{T <: Integer}(row::T, col::T,
                                                       C::LinkMatrix{T},
                                                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                       loglikMargin::Array{<:AbstractFloat, 1},
                                                       countDeltas::Array{<:Integer, 2},
                                                       obsDeltas::Array{<:Integer, 2})
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas)
            return loglikdelta, countdelta, obsdelta, 1
        else ##single switch move I
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(C.col2row[col], col, compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(C.col2row[col], col, compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas) - get_counts(C.col2row[col], col, compsum, obsDeltas)
            return loglikdelta, countdelta, obsdelta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            loglikdelta = get_loglik(row, col, compsum, loglikMargin) - get_loglik(row, C.row2col[row], compsum, loglikMargin)
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(row, C.row2col[row], compsum, countDeltas)
            obsdelta = get_counts(row, col, compsum, obsDeltas) - get_counts(row, C.row2col[row], compsum, obsDeltas)
            return loglikdelta, countdelta, obsdelta, 3
        elseif C.col2row[col] == row ##delete move
            loglikdelta = -get_loglik(row, col, compsum, loglikMargin)
            countdelta = -get_counts(row, col, compsum, countDeltas)
            obsdelta = -get_counts(row, col, compsum, obsDeltas)
            return loglikdelta, countdelta, obsdelta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            
            loglikdelta = get_loglik(row, col, compsum, loglikMargin)
            loglikdelta += get_loglik(rowalt, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(row, colalt, compsum, loglikMargin)
            loglikdelta -= get_loglik(rowalt, col, compsum, loglikMargin)
            
            countdelta = get_counts(row, col, compsum, countDeltas)
            countdelta += get_counts(rowalt, colalt, compsum, countDeltas)
            countdelta -= get_counts(row, colalt, compsum, countDeltas)
            countdelta -= get_counts(rowalt, col, compsum, countDeltas)
            
            obsdelta = get_counts(row, col, compsum, obsDeltas)
            obsdelta += get_counts(rowalt, colalt, compsum, obsDeltas)
            obsdelta -= get_counts(row, colalt, compsum, obsDeltas)
            obsdelta -= get_counts(rowalt, col, compsum, obsDeltas)
            
            return loglikdelta, countdelta, obsdelta, 5
        end
    end
end

randomwalk1_loglikcountobsdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikMargin::Array{<:AbstractFloat, 1}, countDeltas::Array{<:Integer, 2}, obsDeltas::Array{<:Integer, 2}) =
    randomwalk1_loglikcountobsdelta(idx.I[1], idx.I[2], C, compsum, loglikMargin, countDeltas, obsDeltas)

function randomwalk1_movecount(nrow::Integer, ncol::Integer, nlink::Integer)
    return nrow * ncol - div(nlink * (nlink - 1), 2)
end

randomwalk1_movecount(crng::CartesianRange{CartesianIndex{2}}, nlink::Integer) = randomwalk1_movecount(size(crng)[1], size(crng)[2], nlink)

function randomwalk1_reversemove(moverow::Integer, movecol::Integer, movetype::Integer, C::LinkMatrix)
    #movetype = 1 => add
    #movetype = 2 => switch link row
    #movetype = 3 => switch link column
    #movetype = 4 => delete
    #movetype = 5 => double switch
    if movetype == 1 || movetype == 4
        return moverow, movecol
    elseif movetype == 2
        return C.col2row[movecol], movecol
    elseif movetype == 3
        return moverow, C.row2col[moverow]
    elseif movetype == 5
        return moverow, C.row2col[moverow] #correction from original coding
    else
        error("unexpected move type")
        return nothing
    end
end

function randomwalk1_movetype2ratio(movetype::Integer, C::LinkMatrix)
    #ratio = P(move back) / P(move) = #moves / #moves back
    if movetype == 1 #check ratio P(new -> old) / P(old -> new)
        return randomwalk1_movecount(C.nrow, C.ncol, C.nlink) / randomwalk1_movecount(C.nrow, C.ncol, C.nlink + 1)
        #ratio = (C.nrow * C.ncol - div(C.nlink * (C.nlink - 1), 2)) / (C.nrow * C.ncol - div(C.nlink * (C.nlink + 1), 2))
    elseif movetype == 4 #check ratio P(new -> old) / P(old -> new)
        return randomwalk1_movecount(C.nrow, C.ncol, C.nlink) / randomwalk1_movecount(C.nrow, C.ncol, C.nlink - 1)
        #ratio = (C.nrow * C.ncol - div(C.nlink * (C.nlink - 1), 2)) / (C.nrow * C.ncol - div((C.nlink - 1) * (C.nlink - 2), 2))
    else
        return 1.0
    end
end

function randomwalk1_draw(C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          countDeltas::Array{<:Integer, 2})
    
    ii = Int(ceil(rand() * C.nrow))
    jj = Int(ceil(rand() * C.ncol))
    countdelta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)

    #only accept double switches with p = 0.5 to make sampling uniform over moves
    if movetype == 5
        if rand() < 0.5
            return randomwalk1_draw(C, compsum, countDeltas)
        end
    end
    moveratio = randomwalk1_movetype2ratio(movetype, C)
    return randomwalk1_move(ii, jj, C), countdelta, moveratio
end

function randomwalk1_draw(rng::CartesianRange{CartesianIndex{2}},
                          C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          countDeltas::Array{<:Integer, 2})
    ii = Int(ceil(rand() * size(rng)[1])) + rng.start.I[1] - 1
    jj = Int(ceil(rand() * size(rng)[2])) + rng.start.I[2] - 1
    countdelta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)

    #only accept double switches with p = 0.5 to make sampling uniform over moves
    if movetype == 5
        if rand() < 0.5
            return randomwalk1_draw(rng, C, compsum, countDeltas)
        end
    end

    moveratio = randomwalk1_movetype2ratio(movetype, C)
    return randomwalk1_move(ii, jj, C), countdelta, moveratio
end

#efficent compute ratio of likelihoods
function move_weights(C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2}, logDiff::Array{<:AbstractFloat, 1},
                      balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        countdelta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(countdelta, logDiff)
        else
            moveweights[idx] = balance_function(countdelta, logDiff)
        end
    end
    return moveweights
end

function move_weights(C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2}, logDiff::Array{<:AbstractFloat, 1},
                      logpdfC::Function, ratioPrior::Bool,
                      balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        propC = randomwalk1_move(ii, jj, C)
        countdelta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(countdelta, logDiff, propC, C, logpdfC, ratioPrior)
        else
            moveweights[idx] = balance_function(countdelta, logDiff, propC, C, logpdfC, ratioPrior)
        end
    end
    return moveweights
end

function move_weights(C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      loglikMargin::Array{<:AbstractFloat, 1},
                      logpdfC::Function, balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        loglikdelta, movetype = randomwalk1_loglikdelta(ii, jj, C, compsum, loglikMargin)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 1
            moveweights[idx] = balance_function(loglikdelta, 1, C, logpdfC)
        elseif movetype == 4
            moveweights[idx] = balance_function(loglikdelta, -1, C, logpdfC)
        elseif movetype == 5
            moveweights[idx] = 0.5 * balance_function(loglikdelta, 0, C, logpdfC)
        else
            moveweights[idx] = balance_function(loglikdelta, 0, C, logpdfC)
        end
    end
    return moveweights
end

function log_move_weights(C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      loglikMargin::Array{<:AbstractFloat, 1},
                      logpdfC::Function, log_balance_function::Function = lidentity_balance)
    lmoveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        loglikdelta, movetype = randomwalk1_loglikdelta(ii, jj, C, compsum, loglikMargin)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 1
            lmoveweights[idx] = log_balance_function(loglikdelta, 1, C, logpdfC)
        elseif movetype == 4
            lmoveweights[idx] = log_balance_function(loglikdelta, -1, C, logpdfC)
        elseif movetype == 5
            lmoveweights[idx] = loghalf + log_balance_function(loglikdelta, 0, C, logpdfC)
        else
            lmoveweights[idx] = log_balance_function(loglikdelta, 0, C, logpdfC)
        end
    end
    return lmoveweights
end

function move_weights(rng::CartesianRange{CartesianIndex{2}},
                      C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2}, logDiff::Array{<:AbstractFloat, 1},
                      balance_function::Function = identity_balance)
    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    moveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        countdelta, movetype = randomwalk1_countdelta(startrow + ii,
                                                   startcol + jj,
                                                   C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(countdelta, logDiff)
        else
            moveweights[idx] = balance_function(countdelta, logDiff)
        end
    end
    return moveweights
end

function move_weights(rng::CartesianRange{CartesianIndex{2}},
                      C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2}, logDiff::Array{<:AbstractFloat, 1},
                      logpdfC::Function, ratioPrior::Bool,
                      balance_function::Function = identity_balance)
    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    moveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        propC = randomwalk1_move(ii, jj, C)
        countdelta, movetype = randomwalk1_countdelta(startrow + ii,
                                                   startcol + jj,
                                                   C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(countdelta, logDiff, propC, C, logpdfC, ratioPrior)
        else
            moveweights[idx] = balance_function(countdelta, logDiff, propC, C, logpdfC, ratioPrior)
        end
    end
    return moveweights
end

function move_weights(rng::CartesianRange{CartesianIndex{2}},
                      C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      loglikMargin::Array{<:AbstractFloat, 1},
                      logpdfC::Function, balance_function::Function = identity_balance)

    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    moveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        loglikdelta, movetype = randomwalk1_loglikdelta(startrow + ii, startcol + jj, C, compsum, loglikMargin)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 1
            moveweights[idx] = balance_function(loglikdelta, 1, C, logpdfC)
        elseif movetype == 4
            moveweights[idx] = balance_function(loglikdelta, -1, C, logpdfC)
        elseif movetype == 5
            moveweights[idx] = 0.5 * log_balance_function(loglikdelta, 0, C, logpdfC)
        else
            moveweights[idx] = balance_function(loglikdelta, 0, C, logpdfC)
        end
    end
    return moveweights
end

function log_move_weights(rng::CartesianRange{CartesianIndex{2}},
                          C::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikMargin::Array{<:AbstractFloat, 1},
                          logpdfC::Function, log_balance_function::Function = lidentity_balance)
    
    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    lmoveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        loglikdelta, movetype = randomwalk1_loglikdelta(startrow + ii, startcol + jj, C, compsum, loglikMargin)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 1
            lmoveweights[idx] = log_balance_function(loglikdelta, 1, C, logpdfC)
        elseif movetype == 4
            lmoveweights[idx] = log_balance_function(loglikdelta, -1, C, logpdfC)
        elseif movetype == 5
            lmoveweights[idx] = loghalf + log_balance_function(loglikdelta, 0, C, logpdfC)
        else
            lmoveweights[idx] = log_balance_function(loglikdelta, 0, C, logpdfC)
        end
    end
    return lmoveweights
end

function locally_balanced_draw(C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               countDeltas::Array{<:Integer, 2},
                               logDiff::Array{<:AbstractFloat, 1},
                               balance_function::Function)
    ##Move weights
    moveweights = move_weights(C, compsum, countDeltas, logDiff, balance_function)
    
    ##Sample move
    idx = sample(Weights(vec(moveweights)))
    moverow, movecol = ind2sub((compsum.nrow, compsum.ncol), idx)
    countdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(propC, compsum, countDeltas, logDiff, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, countdelta, moveratio
end

function locally_balanced_draw(C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               countDeltas::Array{<:Integer, 2},
                               logDiff::Array{<:AbstractFloat, 1},
                               logpdfC::Function, ratioPrior::Bool,
                               balance_function::Function)
    ##Move weights
    moveweights = move_weights(C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    
    ##Sample move
    idx = sample(Weights(vec(moveweights)))
    moverow, movecol = ind2sub((compsum.nrow, compsum.ncol), idx)
    countdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(propC, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, countdelta, moveratio
end

function locally_balanced_draw(C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               loglikMargin::Array{<:AbstractFloat, 1},
                               countDeltas::Array{<:Integer, 2},
                               logpdfC::Function, 
                               log_balance_function::Function)
    ##Move weights
    lmoveweights = log_move_weights(C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsummw = logsumexp(lmoveweights)
    
    ##Sample move
    idx = sample(Weights(exp.(lmoveweights .- lsummw)))
    moverow, movecol = ind2sub((compsum.nrow, compsum.ncol), idx)
    loglikdelta, countdelta, movetype = randomwalk1_loglikcountdelta(moverow, movecol, C, compsum, loglikMargin, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)
    
    ##Proposal weights and reverse move
    lpropweights = log_move_weights(propC, compsum, loglikMargin, logpdfC, log_balance_function)
    lsumpw = logsumexp(lpropweights)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
        
    ##Compute ratio
    moveratio = exp(lpropweights[revidx] - lmoveweights[idx] + lsummw - lsumpw)

    #also return change in likelihood
    return propC, loglikdelta, countdelta,  moveratio
end

function locally_balanced_draw!(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                loglikMargin::Array{<:AbstractFloat, 1},
                                countDeltas::Array{<:Integer, 2},
                                logpdfC::Function, 
                                log_balance_function::Function)
    ##Move weights
    lmoveweights = log_move_weights(C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsummw = logsumexp(lmoveweights)
    
    ##Sample move
    idx = sample(Weights(exp.(lmoveweights .- lsummw)))
    moverow, movecol = ind2sub((compsum.nrow, compsum.ncol), idx)
    loglikdelta, countdelta, movetype = randomwalk1_loglikcountdelta(moverow, movecol, C, compsum, loglikMargin, countDeltas)

    #Compute change in prior
    if movetype == 1
        lpriorratio = logpdfC(1, C)
        #lmoveweights[idx] = log_balance_function(loglikdelta, 1, C, logpdfC)
    elseif movetype == 4
        lpriorratio = logpdfC(-1, C)
    else
        lpriorratio = logpdfC(0, C)
    end

    #Assume move to compute reverse weights
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
    randomwalk1_move!(moverow, movecol, C)
    
    ##Proposal weights and reverse move
    lpropweights = log_move_weights(C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsumpw = logsumexp(lpropweights)
        
    ##Compute ratio
    lmoveratio = lpropweights[revidx] - lmoveweights[idx] + lsummw - lsumpw

    #also return change in likelihood
    if rand() < exp(loglikdelta + lpriorratio + lmoveratio)
        #return C, countdelta, true
        return countdelta, true
    else
        randomwalk1_move!(revrow, revcol, C)
        return countdelta, false
    end
end

function locally_balanced_draw(rng::CartesianRange{CartesianIndex{2}},
                               C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               countDeltas::Array{<:Integer, 2},
                               logDiff::Array{<:AbstractFloat, 1},
                               balance_function::Function)
    ##Move weights
    moveweights = move_weights(rng, C, compsum, countDeltas, logDiff, balance_function)
    
    ##Sample move
    idx = sample(Weights(vec(moveweights)))
    moverow, movecol = ind2sub(size(rng), idx)
    moverow += rng.start.I[1] - 1
    movecol += rng.start.I[2] - 1
    countdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(rng, propC, compsum, countDeltas, logDiff, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revrow -= rng.start.I[1] - 1
    revcol -= rng.start.I[2] - 1
    revidx = sub2ind(size(rng), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, countdelta, moveratio
end

function locally_balanced_draw(rng::CartesianRange{CartesianIndex{2}},
                               C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               loglikMargin::Array{<:AbstractFloat, 1},
                               countDeltas::Array{<:Integer, 2},
                               logpdfC::Function, 
                               log_balance_function::Function)
    ##Move weights
    lmoveweights = log_move_weights(C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsummw = logsumexp(lmoveweights)
    
    ##Sample move
    idx = sample(Weights(exp.(moveweights .- lsummw)))
    moverow, movecol = ind2sub(size(rng), idx)
    moverow += rng.start.I[1] - 1
    movecol += rng.start.I[2] - 1
    loglikdelta, countdelta, movetype = randomwalk1_loglikcountdelta(moverow, movecol, C, compsum, loglikMargin, countDeltas)
    
    propC = randomwalk1_move(moverow, movecol, C)
    
    ##Proposal weights and reverse move
    lpropweights = log_move_weights(rng, propC, compsum, loglikMargin, logpdfC, log_balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revrow -= rng.start.I[1] - 1
    revcol -= rng.start.I[2] - 1
    revidx = sub2ind(size(rng), revrow, revcol)
        
    ##Compute ratio
    moveratio = exp(lpropweights[revidx] - lmoveweights[idx] + lsummw - lsumpw)

    #also return change in likelihood
    return propC, loglikdelta, countdelta,  moveratio
end

function locally_balanced_draw(rng::CartesianRange{CartesianIndex{2}},
                               C::LinkMatrix,
                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                               countDeltas::Array{<:Integer, 2},
                               logDiff::Array{<:AbstractFloat, 1},
                               logpdfC::Function, ratioPrior::Bool,
                               balance_function::Function)
    ##Move weights
    moveweights = move_weights(rng, C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    
    ##Sample move
    idx = sample(Weights(vec(moveweights)))
    moverow, movecol = ind2sub(size(rng), idx)
    moverow += rng.start.I[1] - 1
    movecol += rng.start.I[2] - 1
    countdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(rng, propC, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revrow -= rng.start.I[1] - 1
    revcol -= rng.start.I[2] - 1
    revidx = sub2ind(size(rng), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, countdelta, moveratio
end

function locally_balanced_draw!(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                loglikMargin::Array{<:AbstractFloat, 1},
                                countDeltas::Array{<:Integer, 2},
                                logpdfC::Function, 
                                log_balance_function::Function)
    ##Move weights
    lmoveweights = log_move_weights(rng, C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsummw = logsumexp(lmoveweights)
    
    ##Sample move
    idx = sample(Weights(exp.(lmoveweights .- lsummw)))
    moverow, movecol = ind2sub(size(rng), idx)
    moverow += rng.start.I[1] - 1
    movecol += rng.start.I[2] - 1
    loglikdelta, countdelta, movetype = randomwalk1_loglikcountdelta(moverow, movecol, C, compsum, loglikMargin, countDeltas)

    #Compute change in prior
    if movetype == 1
        lpriorratio = logpdfC(1, C)
        #lmoveweights[idx] = log_balance_function(loglikdelta, 1, C, logpdfC)
    elseif movetype == 4
        lpriorratio = logpdfC(-1, C)
    else
        lpriorratio = logpdfC(0, C)
    end

    #Assume move to compute reverse weights
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind(size(rng), revrow - rng.start.I[1] + 1, revcol - rng.start.I[2] + 1)
    randomwalk1_move!(moverow, movecol, C)
    
    ##Proposal weights and reverse move
    lpropweights = log_move_weights(rng, C, compsum, loglikMargin, logpdfC, log_balance_function)
    lsumpw = logsumexp(lpropweights)

    ##Compute ratio
    lmoveratio = (lpropweights[revidx] - lmoveweights[idx] + lsummw - lsumpw)
    
    #also return change in likelihood
    if rand() < exp(loglikdelta + lpriorratio + lmoveratio)
        #return C, countdelta, true
        return countdelta, true
    else
        randomwalk1_move!(revrow, revcol, C)
        #return C, countdelta, false
        return countdelta, false
    end
end

function globally_balanced_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, identity_balance)
end

function globally_balanced_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, identity_balance)
end

function globally_balanced_draw!(C::LinkMatrix,
                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 loglikMargin::Array{<:AbstractFloat, 1},
                                 countDeltas::Array{<:Integer, 2},
                                 logpdfC::Function)
    return locally_balanced_draw!(C, compsum, loglikMargin, countDeltas, logpdfC, lidentity_balance)
end

function globally_balanced_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, identity_balance)
end

function globally_balanced_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, identity_balance)
end

function globally_balanced_draw!(rng::CartesianRange{CartesianIndex{2}},
                                 C::LinkMatrix,
                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 loglikMargin::Array{<:AbstractFloat, 1},
                                 countDeltas::Array{<:Integer, 2},
                                 logpdfC::Function)
    return locally_balanced_draw!(rng, C, compsum, loglikMargin, countDeltas, logpdfC, lidentity_balance)
end

function locally_balanced_sqrt_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, sqrt_balance)
end

function locally_balanced_sqrt_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, sqrt_balance)
end

function locally_balanced_sqrt_draw!(C::LinkMatrix,
                                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                     loglikMargin::Array{<:AbstractFloat, 1},
                                     countDeltas::Array{<:Integer, 2},
                                     logpdfC::Function)
    return locally_balanced_draw!(C, compsum, loglikMargin, countDeltas, logpdfC, lsqrt_balance)
end

function locally_balanced_sqrt_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, sqrt_balance)
end

function locally_balanced_sqrt_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, sqrt_balance)
end

function locally_balanced_sqrt_draw!(rng::CartesianRange{CartesianIndex{2}},
                                     C::LinkMatrix,
                                     compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                     loglikMargin::Array{<:AbstractFloat, 1},
                                     countDeltas::Array{<:Integer, 2},
                                     logpdfC::Function)
    return locally_balanced_draw!(rng, C, compsum, loglikMargin, countDeltas, logpdfC, lsqrt_balance)
end

function locally_balanced_barker_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, barker_balance)
end

function locally_balanced_barker_draw(C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, barker_balance)
end

function locally_balanced_barker_draw!(C::LinkMatrix,
                                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                       loglikMargin::Array{<:AbstractFloat, 1},
                                       countDeltas::Array{<:Integer, 2},
                                       logpdfC::Function)
    return locally_balanced_draw!(C, compsum, loglikMargin, countDeltas, logpdfC, lbarker_balance)
end

function locally_balanced_barker_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1})
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, barker_balance)
end

function locally_balanced_barker_draw(rng::CartesianRange{CartesianIndex{2}},
                                C::LinkMatrix,
                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                countDeltas::Array{<:Integer, 2},
                                logDiff::Array{<:AbstractFloat, 1},
                                logpdfC::Function, ratioPrior::Bool)
    return locally_balanced_draw(rng, C, compsum, countDeltas, logDiff, logpdfC, ratioPrior, barker_balance)
end

function locally_balanced_barker_draw!(rng::CartesianRange{CartesianIndex{2}},
                                       C::LinkMatrix,
                                       compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                       loglikMargin::Array{<:AbstractFloat, 1},
                                       countDeltas::Array{<:Integer, 2},
                                       logpdfC::Function)
    return locally_balanced_draw!(rng, C, compsum, loglikMargin, countDeltas, logpdfC, lbarker_balance)
end

function randomwalk2_move!{T <: Integer}(row::T, col::T, C::LinkMatrix{T})
    
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            add_link!(row, col, C)
            return C
        else
            warn("incorrect draw for randomwalk 2")
            return C
        end
    else
        if iszero(C.col2row[col]) ##switch move 
            C.col2row[C.row2col[row]] = zero(T)
            C.col2row[col] = row
            C.row2col[row] = col
            dropzeros!(C.col2row)
            return C
        elseif C.col2row[col] == row ##delete move
            remove_link!(row, col, C)
            #C.row2col[row] = zero(T)
            #C.col2row[col] = zero(T)
            #C.nlink -= 1
            return C
        else ##double switch move
            warn("incorrect draw for randomwalk 2 (double switch)")
            return C
        end
    end
end

randomwalk2_move{T <: Integer}(row::T, col::T, C::LinkMatrix{T}) = randomwalk2_move!(row, col, deepcopy(C))

function randomwalk2_countdelta{T <: Integer}(row::T, col::T,
                                                C::LinkMatrix{T},
                                                compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                countDeltas::Array{<:Integer, 2})
    countdelta = zeros(eltype(countDeltas), size(countDeltas, 1))
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            countdelta = get_counts(row, col, compsum, countDeltas)
            return countdelta, 1
        else ##single switch move I
            warn("incorrect draw for randomwalk 2")
            return countdelta, 0
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            countdelta = get_counts(row, col, compsum, countDeltas) - get_counts(row, C.row2col[row], compsum, countDeltas)
            return countdelta, 3
        elseif C.col2row[col] == row ##delete move
            countdelta = -get_counts(row, col, compsum, countDeltas)
            return countdelta, 2
        else ##double switch move
            warn("incorrect draw for randomwalk 2 (double switch)")
            return countdelta, 0
        end
    end
end

function randomwalk2_draw(C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          countDeltas::Array{<:Integer, 2},
                          p::AbstractFloat = 0.5)
    
    ii = Int(ceil(rand() * C.nrow))
    if iszero(C.row2col[ii])
        opencols = find(map(jj -> iszero(C.col2row[jj]), 1:C.ncol))
        jj = sample(opencols)
    elseif rand() < p
        jj = C.row2col[ii]
    else
        opencols = find(map(jj -> iszero(C.col2row[jj]), 1:C.ncol))
        if length(opencols) == 0
            jj = C.row2col[ii]
        else
            jj = sample(push!(opencols))
        end
    end
    
    countdelta, movetype = randomwalk2_countdelta(ii, jj, C, compsum, countDeltas)
    
    ratio = 1.0
    if movetype == 1 #check ratio P(new -> old) / P(old -> new)
        ratio = p * (C.ncol - C.nlink)
    elseif movetype == 2 #check ratio P(new -> old) / P(old -> new)
        ratio = (p * (C.ncol - C.nlink + 1))^-1
    end
    #ratio = P(move back) / P(move) = #moves / #moves back
    return randomwalk1_move(ii, jj, C), countdelta, ratio
end


function singleton_gibbs(rng::CartesianRange{CartesianIndex{2}}, C::LinkMatrix,
                         compsum::Union{ComparisonSummary, SparseComparisonSummary},
                         loglikMargin::Array{<:AbstractFloat, 1}, countDeltas::Array{<:Integer, 2},
                         logpdfC::Function)
    countdelta = countDeltas[:, compsum.obsidx[rng.start]]
    loglikdelta = loglikMargin[compsum.obsidx[rng.start]]
    if iszero(C.row2col[rng.start.I[1]])

        lpriorratio = logpdfC(1, C)
        lp1 = loglikdelta + lpriorratio
        
        if rand() <= logistic(lp1)
            return add_link(rng.start.I..., C), countdelta, true
        else
            return deepcopy(C), zeros(countdelta), false
        end
    else

        lpriorratio = -logpdfC(-1, C)
        lp1 = loglikdelta + lpriorratio
        
        if rand() < logistic(lp1)
            return deepcopy(C), zeros(countdelta), false
        else
            propC = remove_link(rng.start.I..., C)
            return propC, -countdelta, true
        end
    end
end

#if blockRows[kk] == 1
#    if blockCols[kk] = 1
#    else
#    end
#elseif blockCols[kk] = 1
#elseif blockRows[kk] == 2 && blockCols[kk] == 2
#end
#if blockRows[kk] == 1 && blockCols[kk] == 1
#    if rand() < logistic(dot(countDeltas[:, compsum.obsidx[blockRows[kk].start]], logDiff))
#       if !has_link(blockRows[kk].start..., C)
#           add_link!(blockRows[kk].start..., C)
#       end
#   else
#       if has_link(blockRows[kk].start..., C)
#           remove_link!(blockRows[kk].start..., C)
#       end
#   end
#end


function singleton_gibbs!(rng::CartesianRange{CartesianIndex{2}}, C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          loglikMargin::Array{<:AbstractFloat, 1}, countDeltas::Array{<:Integer, 2},
                          logpdfC::Function)
    countdelta = countDeltas[:, compsum.obsidx[rng.start]]
    loglikdelta = loglikMargin[compsum.obsidx[rng.start]]
    if iszero(C.row2col[rng.start[1]])
        
        lpriorratio = logpdfC(1, C)
        lp1 = loglikdelta + lpriorratio
        
        if rand() <= logistic(lp1)
            add_link!(rng.start.I..., C)
            return countdelta, true
        else
            return countdelta, false
        end
    else

        lpriorratio = -logpdfC(-1, C)
        lp1 = loglikdelta + lpriorratio
        
        if rand() < logistic(lp1)
            return countdelta, false
        else
            remove_link!(rng.start.I..., C)
            return -countdelta, true
        end
    end
end
