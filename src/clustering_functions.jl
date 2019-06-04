"""
bipartite_cluster(linkArray) -> rowLabels, columnLabels, maxLabel

Finds the connected components taking a binary edge matrix for a bipartite graph
and labeles all components.  Nodes that are connected to no other nodes are left
with a label of 0.
"""
function bipartite_cluster(linkArray::A) where A <: AbstractArray{Bool, 2}
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue{Int64}()
    colQ = Queue{Int64}()
    
    maxLabel = 0
    nextCol = 1
    while nextCol != nothing && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in 1:n
            if rowUnassigned[ii] && linkArray[ii, nextCol]
                enqueue!(rowQ, ii)
                #rowLabels[ii] = maxLabel
                rowUnassigned[ii] = false
            end
        end

        ##Only assign column if row link found (otherwise column links to nothing)
        if length(rowQ) > 0
            maxLabel += 1
            colLabels[nextCol] = maxLabel
        end

        ##Continue adding to cluster until Queues are empty
        while length(rowQ) > 0 || length(colQ) > 0

            ##Dequeue row and seach row for links to unassigned columns
            if length(rowQ) > 0
                row = dequeue!(rowQ)
                rowLabels[row] = maxLabel
                for jj in 1:m
                    if colUnassigned[jj] && linkArray[row, jj]
                        enqueue!(colQ, jj)
                        colUnassigned[jj] = false
                    end
                end
            end

            ##Dequeue column and seach column for links to unassigned rowss
            if length(colQ) > 0
                col = dequeue!(colQ)
                colLabels[col] = maxLabel
                for ii in 1:n
                    if rowUnassigned[ii] && linkArray[ii, col]
                        enqueue!(rowQ, ii)
                        rowUnassigned[ii] = false
                    end
                end
            end
        end
        
        ##Queue's are empty so move to next column
        nextCol = findnext(colUnassigned, nextCol + 1)
    end
    return rowLabels, colLabels, maxLabel
end

"""
bipartite_cluster(weightArray, [threshold]) -> rowLabels, columnLabels, maxLabel

Finds the connected components taking a matrix of edge weights and a threshold
for a bipartite graph and labeles all components.  All edge weights above the
threshold are used to connect components.  Nodes that are connected to no other
nodes are left with a label of 0.  Threshold value defaults to 0.
"""
function bipartite_cluster(weightArray::Array{<:AbstractFloat, 2}, threshold::AbstractFloat = 0.0)
    n, m = size(weightArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue{Int64}()
    colQ = Queue{Int64}()
    
    maxLabel = 0
    nextCol = 1
    while nextCol != nothing && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in 1:n
            if rowUnassigned[ii] && weightArray[ii, nextCol] > threshold
                enqueue!(rowQ, ii)
                #rowLabels[ii] = maxLabel
                rowUnassigned[ii] = false
            end
        end

        ##Only assign column if row link found (otherwise column links to nothing)
        if length(rowQ) > 0
            maxLabel += 1
            colLabels[nextCol] = maxLabel
        end

        ##Continue adding to cluster until Queues are empty
        while length(rowQ) > 0 || length(colQ) > 0

            ##Dequeue row and seach row for links to unassigned columns
            if length(rowQ) > 0
                row = dequeue!(rowQ)
                rowLabels[row] = maxLabel
                for jj in 1:m
                    if colUnassigned[jj] && weightArray[row, jj] > threshold
                        enqueue!(colQ, jj)
                        colUnassigned[jj] = false
                    end
                end
            end

            ##Dequeue column and seach column for links to unassigned rowss
            if length(colQ) > 0
                col = dequeue!(colQ)
                colLabels[col] = maxLabel
                for ii in 1:n
                    if rowUnassigned[ii] && weightArray[ii, col] > threshold
                        enqueue!(rowQ, ii)
                        rowUnassigned[ii] = false
                    end
                end
            end
        end
        
        ##Queue's are empty so move to next column
        nextCol = findnext(colUnassigned, nextCol + 1)
    end
    return rowLabels, colLabels, maxLabel
end


function bipartite_cluster(linkArray::A) where A <: SparseMatrixCSC{Bool}
    transposeArray = permutedims(linkArray, (2,1))
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue{Int64}()
    colQ = Queue{Int64}()

    rows = rowvals(linkArray)
    cols = rowvals(transposeArray)
    
    maxLabel = 0
    nextCol = 1
    while nextCol != nothing && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in nzrange(linkArray, nextCol)#1:n
            if rowUnassigned[rows[ii]] #&& linkArray[rows[ii], nextCol]
                enqueue!(rowQ, rows[ii])
                rowUnassigned[rows[ii]] = false
            end
        end

        ##Only assign column if row link found (otherwise column links to nothing)
        if length(rowQ) > 0
            maxLabel += 1
            colLabels[nextCol] = maxLabel
        end

        ##Continue adding to cluster until Queues are empty
        while length(rowQ) > 0 || length(colQ) > 0

            ##Dequeue row and seach row for links to unassigned columns
            if length(rowQ) > 0
                row = dequeue!(rowQ)
                rowLabels[row] = maxLabel
                for jj in nzrange(transposeArray, row)
                    if colUnassigned[cols[jj]] #&& linkArray[row, cols[jj]]
                        enqueue!(colQ, cols[jj])
                        colUnassigned[cols[jj]] = false
                    end
                end
            end

            ##Dequeue column and seach column for links to unassigned rowss
            if length(colQ) > 0
                col = dequeue!(colQ)
                colLabels[col] = maxLabel
                for ii in nzrange(linkArray, col)#1:n
                    if rowUnassigned[rows[ii]] #&& linkArray[rows[ii], nextCol]
                        enqueue!(rowQ, rows[ii])
                        rowUnassigned[rows[ii]] = false
                    end
                end
            end
        end
        
        ##Queue's are empty so move to next column
        nextCol = findnext(colUnassigned, nextCol + 1)
    end
    return rowLabels, colLabels, maxLabel
end

function bipartite_cluster(linkArray::SparseMatrixCSC{T}, threshold::T) where T <: AbstractFloat
    transposeArray = permutedims(linkArray, (2,1))
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue{Int64}()
    colQ = Queue{Int64}()

    rows = rowvals(linkArray)
    cols = rowvals(transposeArray)
    
    maxLabel = 0
    nextCol = 1
    while nextCol != nothing && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in nzrange(linkArray, nextCol)#1:n
            if rowUnassigned[rows[ii]] && linkArray[rows[ii], nextCol] > threshold
                enqueue!(rowQ, rows[ii])
                rowUnassigned[rows[ii]] = false
            end
        end

        ##Only assign column if row link found (otherwise column links to nothing)
        if length(rowQ) > 0
            maxLabel += 1
            colLabels[nextCol] = maxLabel
        end

        ##Continue adding to cluster until Queues are empty
        while length(rowQ) > 0 || length(colQ) > 0

            ##Dequeue row and seach row for links to unassigned columns
            if length(rowQ) > 0
                row = dequeue!(rowQ)
                rowLabels[row] = maxLabel
                for jj in nzrange(transposeArray, row)
                    if colUnassigned[cols[jj]] && linkArray[row, cols[jj]] > threshold
                        enqueue!(colQ, cols[jj])
                        colUnassigned[cols[jj]] = false
                    end
                end
            end

            ##Dequeue column and seach column for links to unassigned rowss
            if length(colQ) > 0
                col = dequeue!(colQ)
                colLabels[col] = maxLabel
                for ii in nzrange(linkArray, col)#1:n
                    if rowUnassigned[rows[ii]] && linkArray[rows[ii], col] > threshold
                        enqueue!(rowQ, rows[ii])
                        rowUnassigned[rows[ii]] = false
                    end
                end
            end
        end
        
        ##Queue's are empty so move to next column
        nextCol = findnext(colUnassigned, nextCol + 1)
    end
    return rowLabels, colLabels, maxLabel
end

#iterative_bipartite_cluster(sparse([4.0 3. 3. 2. 2.; 3. 4. 3. 2. 2.; 3. 3. 4. 2. 2.; 2. 2. 2. 3. 1.; 2. 2. 2. 1. 3.]), 4, 1.1, 0.5)
#iterative_bipartite_cluster(sparse([4.0 3. 3. 1. 1.; 3. 4. 3. 1. 1.; 3. 3. 4. 1. 1.; 1. 1. 1. 3. 2.; 1. 1. 1. 2. 3.]), 4, 1.1, 0.5)

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
function iterative_bipartite_cluster(linkArray::SparseMatrixCSC{T}, maxsize::Integer, threshold0::T, incr::T, rows::Array{Int64, 1} = collect(1:size(linkArray, 1)), cols::Array{Int64, 1} = collect(1:size(linkArray, 2))) where T <: AbstractFloat

    #run with initial clustering
    rowLabels0, colLabels0, maxLabel0 = bipartite_cluster(linkArray[rows, cols], threshold0)
    clusterThresholds0 = fill(threshold0, maxLabel0)

    #define dictionary for mapping
    rowIndicies = Dict{Int64, Array{Int64, 1}}()
    colIndicies = Dict{Int64, Array{Int64, 1}}()
    for kk in 1:maxLabel0
        rowIndicies[kk] = Array{Int64}(undef, 0)
        colIndicies[kk] = Array{Int64}(undef, 0)
    end
    
    #get mapping from components to indicies
    for (row, lab) in enumerate(IndexLinear(), rowLabels0)
        if !iszero(lab)
            push!(rowIndicies[lab], row)
        end
    end

    
    for (col, lab) in enumerate(IndexLinear(), colLabels0)
        if !iszero(lab)
            push!(colIndicies[lab], col)
        end
    end
    
    for kk in 1:maxLabel0
        if (length(rowIndicies[kk]) * length(colIndicies[kk])) > maxsize

            #find row and column indcies
            clusterRows = rowIndicies[kk]
            clusterCols = colIndicies[kk]
            
            rowLabels, colLabels, maxLabel, clusterThresholds = iterative_bipartite_cluster(linkArray, maxsize, threshold0 + incr, incr, rows[clusterRows], cols[clusterCols])
            #possible results - (1) no clusters, (2) only one smaller cluster, (3) >1 cluster

            #update row cluster labels
            for (idx, row) in enumerate(IndexLinear(), clusterRows)
                if rowLabels[idx] == 0
                    rowLabels0[row] = 0
                elseif rowLabels[idx] == 1
                    rowLabels0[row] = kk #this should leave the value unchanged
                else #> 1 case
                    rowLabels0[row] = maxLabel0 + rowLabels[idx] - 1
                end    
            end

            #update column cluster labels
            for (idx, col) in enumerate(IndexLinear(), clusterCols)
                if colLabels[idx] == 0
                    colLabels0[col] = 0
                elseif colLabels[idx] == 1
                    colLabels0[col] = kk #this should leave the value unchanged
                else #> 1 case
                    colLabels0[col] = maxLabel0 + colLabels[idx] - 1
                end
            end

            #update cluster thresholds and cluster number
            if maxLabel == 0
                clusterThresholds0[kk] = 0
            elseif maxLabel == 1
                clusterThresholds0[kk] = clusterThresholds[1]
            else #>1 case
                clusterThresholds0[kk] = clusterThresholds[1]
                append!(clusterThresholds0, clusterThresholds[2:end])
            end
            
            maxLabel0 += maxLabel - 1
        end
    end
    return rowLabels0, colLabels0, maxLabel0, clusterThresholds0
end

#iterative_bipartite_cluster2(sparse([4. 3. 3. 2. 2.; 3. 4. 3. 2. 2.; 3. 3. 4. 2. 2.; 2. 2. 2. 3. 1.; 2. 2. 2. 1. 3.]), 4, 1.1, 0.5)
#iterative_bipartite_cluster2(sparse([4. 3. 3. 1. 1.; 3. 4. 3. 1. 1.; 3. 3. 4. 1. 1.; 1. 1. 1. 3. 2.; 1. 1. 1. 2. 3.]), 4, 1.1, 0.5)
#iterative_bipartite_cluster2(sparse([4. 4. 4. 1. ; 4. 4. 4. 1. ; 4. 4. 4. 1.; 1. 1. 1. 2.]), 4, 1.1, 0.5)

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
function iterative_bipartite_cluster2(linkArray::SparseMatrixCSC{T}, maxsize::Integer, threshold0::T, incr::T) where T <: AbstractFloat

    #run with initial clustering
    rowLabels0, colLabels0, maxLabel0 = bipartite_cluster(linkArray, threshold0)
    clusterThresholds0 = fill(threshold0, maxLabel0)

    #define dictionary for mapping
    rowIndicies = Dict{Int64, Array{Int64, 1}}()
    colIndicies = Dict{Int64, Array{Int64, 1}}()
    for kk in 1:maxLabel0
        rowIndicies[kk] = Array{Int64}(undef, 0)
        colIndicies[kk] = Array{Int64}(undef, 0)
    end
    
    #get mapping from components to indicies
    for (row, lab) in enumerate(IndexLinear(), rowLabels0)
        if !iszero(lab)
            push!(rowIndicies[lab], row)
        end
    end

    for (col, lab) in enumerate(IndexLinear(), colLabels0)
        if !iszero(lab)
            push!(colIndicies[lab], col)
        end
    end

    kk = 1
    while kk <= maxLabel0
        if (length(rowIndicies[kk]) * length(colIndicies[kk])) > maxsize
            
            #find row and column indcies
            clusterRows = rowIndicies[kk]
            clusterCols = colIndicies[kk]
            rowLabels, colLabels, maxLabel = bipartite_cluster(linkArray[clusterRows, clusterCols], clusterThresholds0[kk] + incr)
            #possible results - (1) no clusters, (2) only one smaller cluster, (3) >1 cluster

            rowIndicies[kk] = Array{Int64}(undef, 0)
            colIndicies[kk] = Array{Int64}(undef, 0)

            if maxLabel > 1
                for clust in (maxLabel0 + 1):(maxLabel0 + maxLabel - 1)
                    rowIndicies[clust] = Array{Int64}(undef, 0)
                    colIndicies[clust] = Array{Int64}(undef, 0)
                end
            end
            
            #update row cluster labels
            for (row, lab) in zip(clusterRows, rowLabels)
                if lab == 0
                    rowLabels0[row] = 0
                elseif lab == 1
                    rowLabels0[row] = kk #this should leave the value unchanged
                    push!(rowIndicies[kk], row)
                else #> 1 case
                    rowLabels0[row] = maxLabel0 + lab - 1
                    push!(rowIndicies[maxLabel0 + lab - 1], row)
                end    
            end

            #update column cluster labels
            for (col, lab) in zip(clusterCols, colLabels)
                if lab == 0
                    colLabels0[col] = 0
                elseif lab == 1
                    colLabels0[col] = kk #this should leave the value unchanged
                    push!(colIndicies[kk], col)
                else #> 1 case
                    colLabels0[col] = maxLabel0 + lab - 1
                    push!(colIndicies[maxLabel0 + lab - 1], col)
                end
            end
            
            #update cluster thresholds and cluster number
            if maxLabel == 0
                @warn "component with $(length(clusterRows)) rows and $(length(clusterCols)) columns could not be subdivided, consider using a smaller increment."

                #reset cluster indicies so that there are no gaps in the numbering
                for clust in kk:(maxLabel0 - 1)
                    rowIndicies[clust] = rowIndicies[clust + 1]
                    colIndicies[clust] = colIndicies[clust + 1]
                end
                delete!(rowIndicies, maxLabel0)
                delete!(colIndicies, maxLabel0)
                
                clusterThresholds0 = clusterThresholds0[1:maxLabel0 .!= kk]
            elseif maxLabel == 1
                clusterThresholds0[kk] += incr
            else #>1 case
                clusterThresholds0[kk] += incr
                append!(clusterThresholds0, fill(clusterThresholds0[kk], maxLabel - 1))
            end

            maxLabel0 += maxLabel - 1
        else
            kk += 1
        end
    end
    return rowLabels0, colLabels0, maxLabel0, clusterThresholds0
end
