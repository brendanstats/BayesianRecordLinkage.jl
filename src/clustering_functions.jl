"""
    bipartite_cluster(linkArray) -> rowLabels, columnLabels

Finds the connected components taking a binary edge matrix for a bipartite graph
and labeles all components.  Nodes that are connected to no other nodes are left
with a label of 0.
"""
function bipartite_cluster{A <: AbstractArray{Bool, 2}}(linkArray::A)
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue(Int64)
    colQ = Queue(Int64)
    
    maxLabel = 0
    nextCol = 1
    while nextCol > 0 && nextCol <= m
        
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
    bipartite_cluster(weightArray, [threshold]) -> rowLabels, columnLabels

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
    rowQ = Queue(Int64)
    colQ = Queue(Int64)
    
    maxLabel = 0
    nextCol = 1
    while nextCol > 0 && nextCol <= m
        
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


function bipartite_cluster{A <: SparseMatrixCSC}(linkArray::A)
    transposeArray = linkArray'
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue(Int64)
    colQ = Queue(Int64)

    rows = rowvals(linkArray)
    cols = rowvals(transposeArray)
    
    maxLabel = 0
    nextCol = 1
    while nextCol > 0 && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in nzrange(linkArray, nextCol)#1:n
            #if rowUnassigned[ii] && linkArray[ii, nextCol]
            #    enqueue!(rowQ, ii)
            #    rowUnassigned[ii] = false
            #end
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
                #for jj in 1:m
                #    if colUnassigned[jj] && linkArray[row, jj]
                #        enqueue!(colQ, jj)
                #        colUnassigned[jj] = false
                #    end
                #end
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
                #for ii in 1:n
                #    if rowUnassigned[ii] && linkArray[ii, col]
                #        enqueue!(rowQ, ii)
                #        rowUnassigned[ii] = false
                #    end
                #end
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

function sparseblock_idxlims{A <: SparseMatrixCSC}(linkArray::A)
    n, m = size(linkArray)
    col2rowstart = fill(n + 1, m)
    col2rowend = fill(0, m)
    row2colstart = fill(m + 1, n)
    row2colend = fill(0, n)
    rows = rowvals(linkArray)
    for col = 1:m
        for ii in nzrange(linkArray, col)
            row = rows[ii]
            if row < col2rowstart[col]
                col2rowstart[col] = row
            end
            if row > col2rowend[col]
                col2rowend[col] = row
            end
            if col < row2colstart[row]
                row2colstart[row] = col
            end
            if col > row2colend[row]
                row2colend[row] = col
            end
        end
    end
    return row2colstart, row2colend, col2rowstart, col2rowend
end

function bipartite_cluster_sparseblock{G <: Integer, A <: SparseMatrixCSC{Bool}}(linkArray::A, row2colstart::Array{G, 1}, row2colend::Array{G, 1}, col2rowstart::Array{G, 1}, col2rowend::Array{G, 1})
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue(Int64)
    colQ = Queue(Int64)
    
    maxLabel = 0
    nextCol = 1
    while nextCol > 0 && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in col2rowstart[nextCol]:col2rowend[nextCol]
            if rowUnassigned[ii] && linkArray[ii, nextCol]
                enqueue!(rowQ, ii)
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
                for jj in row2colstart[row]:row2colend[row]
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
                for ii in col2rowstart[col]:col2rowend[col]
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

function bipartite_cluster_sparseblock{A <: SparseMatrixCSC{Bool}}(linkArray::A)
    return bipartite_cluster_sparseblock(linkArray, sparseblock_idxlims(linkArray)...)
end

function bipartite_cluster_sparseblock{G <: Integer, T <: AbstractFloat, A <: SparseMatrixCSC{T}}(linkArray::A, threshold::T, row2colstart::Array{G, 1}, row2colend::Array{G, 1}, col2rowstart::Array{G, 1}, col2rowend::Array{G, 1})
    n, m = size(linkArray)
    rowLabels = zeros(Int64, n)
    colLabels = zeros(Int64, m)
    rowUnassigned = trues(n)
    colUnassigned = trues(m)
    rowQ = Queue(Int64)
    colQ = Queue(Int64)
    
    maxLabel = 0
    nextCol = 1
    while nextCol > 0 && nextCol <= m
        
        ##Mark column
        colUnassigned[nextCol] = false
        
        ##Seach column for links
        for ii in col2rowstart[nextCol]:col2rowend[nextCol]
            if rowUnassigned[ii] && linkArray[ii, nextCol] > threshold
                enqueue!(rowQ, ii)
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
                for jj in row2colstart[row]:row2colend[row]
                    if colUnassigned[jj] && linkArray[row, jj]  > threshold
                        enqueue!(colQ, jj)
                        colUnassigned[jj] = false
                    end
                end
            end

            ##Dequeue column and seach column for links to unassigned rowss
            if length(colQ) > 0
                col = dequeue!(colQ)
                colLabels[col] = maxLabel
                for ii in col2rowstart[col]:col2rowend[col]
                    if rowUnassigned[ii] && linkArray[ii, col] > threshold
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

function bipartite_cluster_sparseblock{T <: AbstractFloat, A <: SparseMatrixCSC{T}}(linkArray::A, threshold::T)
    return bipartite_cluster_sparseblock(linkArray, threshold, sparseblock_idxlims(linkArray)...)
end
