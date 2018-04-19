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
            delta = countDeltas[:, compsum.obsidx[row, col]]
            return delta, 1
        else ##single switch move I
            delta = countDeltas[:, compsum.obsidx[row, col]] - countDeltas[:, compsum.obsidx[C.col2row[col], col]]
            return delta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            delta = countDeltas[:, compsum.obsidx[row, col]] - countDeltas[:, compsum.obsidx[row, C.row2col[row]]]
            return delta, 3
        elseif C.col2row[col] == row ##delete move
            delta = -countDeltas[:, compsum.obsidx[row, col]]
            return delta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            delta = countDeltas[:, compsum.obsidx[row, col]]
            delta += countDeltas[:, compsum.obsidx[rowalt, colalt]]
            delta -= countDeltas[:, compsum.obsidx[row, colalt]]
            delta -= countDeltas[:, compsum.obsidx[rowalt, col]]
            return delta, 5
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
            delta = loglikMargin[compsum.obsidx[row, col]]
            return delta, 1
        else ##single switch move I
            delta = loglikMargin[compsum.obsidx[row, col]] - loglikMargin[compsum.obsidx[C.col2row[col], col]]
            return delta, 2
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            delta = loglikMargin[compsum.obsidx[row, col]] - loglikMargin[compsum.obsidx[row, C.row2col[row]]]
            return delta, 3
        elseif C.col2row[col] == row ##delete move
            delta = -loglikMargin[compsum.obsidx[row, col]]
            return delta, 4
        else ##double switch move
            rowalt = C.col2row[col]
            colalt = C.row2col[row]
            delta = loglikMargin[compsum.obsidx[row, col]]
            delta += loglikMargin[compsum.obsidx[rowalt, colalt]]
            delta -= loglikMargin[compsum.obsidx[row, colalt]]
            delta -= loglikMargin[compsum.obsidx[rowalt, col]]
            return delta, 5
        end
    end
end

randomwalk1_loglikdelta{T <: Integer}(idx::CartesianIndex{2}, C::LinkMatrix{T}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, loglikMargin::Array{<:AbstractFloat, 1}) =
    randomwalk1_loglikdelta(idx.I[1], idx.I[2], C, compsum, loglikMargin)

                                                                          
function randomwalk1_movecount(nrow::Integer, ncol::Integer, nlink::Integer)
    return nrow * ncol - div(nlink * (nlink - 1), 2)
end

randomwalk1_movecount(crng::CartesianRange{CartesianIndex{2}}, nlink::Integer) = randomwalk1_movecount(size(crng)[1], size(crng)[2], nlink)

function randomwalk1_reversemove(moverow::Integer, movecol::Integer,
                                 movetype::Integer, C::LinkMatrix)
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
        #revrow = C.col2row[movecol]
        #revcol = C.row2col[moverow]
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
    delta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)

    #only accept double switches with p = 0.5 to make sampling uniform over moves
    if movetype == 5
        if rand() < 0.5
            return randomwalk1_draw(C, compsum, countDeltas)
        end
    end
    moveratio = randomwalk1_movetype2ratio(movetype, C)
    return randomwalk1_move(ii, jj, C), delta, moveratio
end

function randomwalk1_draw(rng::CartesianRange{CartesianIndex{2}},
                          C::LinkMatrix,
                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                          countDeltas::Array{<:Integer, 2})
    ii = Int(ceil(rand() * size(rng)[1])) + rng.start.I[1] - 1
    jj = Int(ceil(rand() * size(rng)[2])) + rng.start.I[2] - 1
    delta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)

    #only accept double switches with p = 0.5 to make sampling uniform over moves
    if movetype == 5
        if rand() < 0.5
            return randomwalk1_draw(C, compsum, countDeltas)
        end
    end

    moveratio = randomwalk1_movetype2ratio(movetype, C)
    return randomwalk1_move(ii, jj, C), delta, moveratio
end

#balancing functions
function identity_balance(loglikMargin::Array{<:AbstractFloat, 1})
    return exp(sum(loglikMargin))
end

function identity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1})
    return exp(dot(delta, logDiff))
end

function identity_balance(loglikMargin::Array{<:AbstractFloat, 1},
                          propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return exp(sum(loglikMargin) + logpdfC(propC, C))
    else
        return exp(sum(loglikMargin) + logpdfC(propC) - logpdf(C))
    end
end

function identity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1},
                          propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return exp(dot(delta, logDiff) + logpdfC(propC, C))
    else
        return exp(dot(delta, logDiff) + logpdfC(propC) - logpdf(C))
    end
end

function sqrt_balance(loglikMargin::Array{<:AbstractFloat, 1})
    return exp(0.5 * sum(loglikMargin))
end


function sqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1})
    return exp(0.5 * dot(delta, logDiff))
end

function sqrt_balance(loglikMargin::Array{<:AbstractFloat, 1},
                      propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return exp(0.5 * (sum(loglikMargin) + logpdfC(propC, C)))
    else
        return exp(0.5 * (sum(loglikMargin) + logpdfC(propC) - logpdf(C)))
    end
end


function sqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1},
                      propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return exp(0.5 * (dot(delta, logDiff) + logpdfC(propC, C)))
    else
        return exp(0.5 * (dot(delta, logDiff) + logpdfC(propC) - logpdf(C)))
    end
end

function barker_balance(loglikMargin::Array{<:AbstractFloat, 1})
    return logistic(sum(loglikMargin))
end


function barker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1})
    return logistic(dot(delta, logDiff))
end
#log1pexp or similar, softmax = exp(xi) / sum(exp.(x))
#log(logistic(x)) = -log1pexp(-x)
function barker_balance(loglikMargin::Array{<:AbstractFloat, 1},
                        propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return logistic(sum(loglikMargin) + logpdfC(propC, C))
    else
        return logistic(sum(loglikMargin) + logpdfC(propC) - logpdf(C))
    end
end


function barker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1},
                        propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    if ratioPrior
        return logistic(dot(delta, logDiff) + logpdfC(propC, C))
    else
        return logistic(dot(delta, logDiff) + logpdfC(propC) - logpdf(C))
    end
end

#efficent compute ratio of likelihoods
function move_weights(C::LinkMatrix,
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      loglikMargin::Array{<:AbstractFloat, 1},
                      balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)

        ##down weight double switches since two of them arrive at the same result
        if (!iszerow(C.row2col[ii])) && (C.row2col[ii] != jj)
            moveweights[idx] = 0.5 * balance_function(loglikMargin)
        else
            moveweights[idx] = balance_function(loglikMargin)
        end
    end
    return moveweights
end

function move_weights(C::LinkMatrix,
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2},
                      logDiff::Array{<:AbstractFloat, 1},
                      balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        delta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(delta, logDiff)
        else
            moveweights[idx] = balance_function(delta, logDiff)
        end
    end
    return moveweights
end

function move_weights(C::LinkMatrix,
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2},
                      logDiff::Array{<:AbstractFloat, 1},
                      logpdfC::Function, ratioPrior::Bool,
                      balance_function::Function = identity_balance)
    moveweights = Array{Float64}(C.nrow * C.ncol)
    for jj in 1:C.ncol, ii in 1:C.nrow
        idx = sub2ind((C.nrow, C.ncol), ii, jj)
        propC = randomwalk1_move(ii, jj, C)
        delta, movetype = randomwalk1_countdelta(ii, jj, C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(delta, logDiff, propC, C, logpdfC, ratioPrior)
        else
            moveweights[idx] = balance_function(delta, logDiff, propC, C, logpdfC, ratioPrior)
        end
    end
    return moveweights
end

function move_weights(rng::CartesianRange{CartesianIndex{2}},
                      C::LinkMatrix,
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2},
                      logDiff::Array{<:AbstractFloat, 1},
                      balance_function::Function = identity_balance)
    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    moveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        delta, movetype = randomwalk1_countdelta(startrow + ii,
                                                   startcol + jj,
                                                   C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(delta, logDiff)
        else
            moveweights[idx] = balance_function(delta, logDiff)
        end
    end
    return moveweights
end

function move_weights(rng::CartesianRange{CartesianIndex{2}},
                      C::LinkMatrix,
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      countDeltas::Array{<:Integer, 2},
                      logDiff::Array{<:AbstractFloat, 1},
                      logpdfC::Function, ratioPrior::Bool,
                      balance_function::Function = identity_balance)
    startrow = rng.start.I[1] - 1
    startcol = rng.start.I[2] - 1
    moveweights = Array{Float64}(length(rng))
    for jj in 1:size(rng)[2], ii in 1:size(rng)[1]
        idx = sub2ind(size(rng), ii, jj)
        propC = randomwalk1_move(ii, jj, C)
        delta, movetype = randomwalk1_countdelta(startrow + ii,
                                                   startcol + jj,
                                                   C, compsum, countDeltas)
        
        ##down weight double switches since two of them arrive at the same result
        if movetype == 5
            moveweights[idx] = 0.5 * balance_function(delta, logDiff, propC, C, logpdfC, ratioPrior)
        else
            moveweights[idx] = balance_function(delta, logDiff, propC, C, logpdfC, ratioPrior)
        end
    end
    return moveweights
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
    matchdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(propC, compsum, countDeltas, logDiff, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, matchdelta, moveratio
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
    matchdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(propC, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revidx = sub2ind((compsum.nrow, compsum.ncol), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, matchdelta, moveratio
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
    matchdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(rng, propC, compsum, countDeltas, logDiff, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revrow -= rng.start.I[1] - 1
    revcol -= rng.start.I[2] - 1
    revidx = sub2ind(size(rng), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, matchdelta, moveratio
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
    matchdelta, movetype = randomwalk1_countdelta(moverow, movecol, C, compsum, countDeltas)
    propC = randomwalk1_move(moverow, movecol, C)

    ##Proposal weights and reverse move
    propweights = move_weights(rng, propC, compsum, countDeltas, logDiff, logpdfC, ratioPrior, balance_function)
    revrow, revcol = randomwalk1_reversemove(moverow, movecol, movetype, C)
    revrow -= rng.start.I[1] - 1
    revcol -= rng.start.I[2] - 1
    revidx = sub2ind(size(rng), revrow, revcol)
        
    ##Compute ratio
    moveratio = (propweights[revidx] / moveweights[idx]) * (sum(moveweights) / sum(propweights))
    
    return propC, matchdelta, moveratio
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
    delta = zeros(eltype(countDeltas), size(countDeltas, 1))
    if iszero(C.row2col[row])
        if iszero(C.col2row[col]) ##add move
            delta = countDeltas[:, compsum.obsidx[row, col]]
            return delta, 1
        else ##single switch move I
            warn("incorrect draw for randomwalk 2")
            return delta, 0
        end
    else
        if iszero(C.col2row[col]) ##single switch move II
            delta = countDeltas[:, compsum.obsidx[row, col]] - countDeltas[:, compsum.obsidx[row, C.row2col[row]]]
            return delta, 3
        elseif C.col2row[col] == row ##delete move
            delta = -countDeltas[:, compsum.obsidx[row, col]]
            return delta, 2
        else ##double switch move
            warn("incorrect draw for randomwalk 2 (double switch)")
            return delta, 0
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
    
    delta, movetype = randomwalk2_countdelta(ii, jj, C, compsum, countDeltas)
    
    ratio = 1.0
    if movetype == 1 #check ratio P(new -> old) / P(old -> new)
        ratio = p * (C.ncol - C.nlink)
    elseif movetype == 2 #check ratio P(new -> old) / P(old -> new)
        ratio = (p * (C.ncol - C.nlink + 1))^-1
    end
    #ratio = P(move back) / P(move) = #moves / #moves back
    return randomwalk1_move(ii, jj, C), delta, ratio
end
