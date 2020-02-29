"""
    mutable struct LinkMatrix{G <: Integer}

Stores the state of links between two datasets.

# Fields

* `row2col::Array{G, 1}`: Mapping from row to column.  row2ccol[ii] = jj if row ii is linked to column jj.  If row ii is unlinked then row2col[ii] == 0.
* `col2row::Array{G, 1}`: Mapping from column to row.  col2row[jj] = ii if column jj is linked to row ii.  If column jj is unlinked then col2row[jj] == 0.
* `nlink::G`: Number of record pairs counted as links.
* `nrow::G`: Number of rows equal, nrow == length(row2col).
* `ncol::G`: Number of columns, ncol == length(col2row)

# Constructors

    LinkMatrix(nrow::G, ncol::G) where G <: Integer
    LinkMatrix(row2col::Array{G, 1}, col2row::Array{G, 1}, nlink::G) where G <: Integer
    LinkMatrix(row2col::Array{G, 1}, col2row::Array{G, 1}) where G <: Integer
    LinkMatrix(row2col::Array{G}, ncol::G) where G <: Integer
    LinkMatrix(nrow::G, ncol::G, mrows::Array{G, 1}, mcols::Array{G, 1})

# Arguments

* `nrow::G`: Number of rows equal, nrow == length(row2col).
* `ncol::G`: Number of columns, ncol == length(col2row)
* `row2col::Array{G, 1}`: Mapping from row to column.  row2ccol[ii] = jj if row ii is linked to column jj.  If row ii is unlinked then row2col[ii] == 0.
* `col2row::Array{G, 1}`: Mapping from column to row.  col2row[jj] = ii if column jj is linked to row ii.  If column jj is unlinked then col2row[jj] == 0.
* `nlink::G`: Number of record pairs counted as links.
* `mrows::G`: List of rows that are linked should be paired with a `mcols` variable containing the linked column.
* `mcowl::G`: List of columns that are linked should be paired with a `mrows` variable containing the linked rows.
"""
mutable struct LinkMatrix{G <: Integer}
    row2col::Array{G, 1}
    col2row::Array{G, 1}
    nlink::G
    nrow::G
    ncol::G
end

LinkMatrix(nrow::G, ncol::G) where G <: Integer = LinkMatrix(zeros(G, nrow), zeros(G, ncol), zero(G), nrow, ncol)
LinkMatrix(row2col::Array{G, 1}, col2row::Array{G, 1}, nlink::G) where G <: Integer = LinkMatrix(row2col, col2row, nlink, G(length(row2col)), G(length(col2row)))
LinkMatrix(row2col::Array{G, 1}, col2row::Array{G, 1}) where G <: Integer = LinkMatrix(row2col, col2row, count(!iszero, x))
function LinkMatrix(row2col::Array{G}, ncol::G) where G <: Integer
    col2row = zeros(G, ncol)
    nlink = zero(G)
    for ii in one(G):G(length(row2col))
        if !iszero(row2col[ii])
            col2row[row2col[ii]] = ii
            nlink += one(G)
        end
    end
    LinkMatrix(row2col, col2row, nlink, G(length(row2col)), ncol)
end

function LinkMatrix(nrow::G, ncol::G, mrows::Array{G, 1}, mcols::Array{G, 1}) where G <: Integer
    row2col = zeros(G, nrow)
    col2row = zeros(G, ncol)
    for (ii, jj) in zip(mrows, mcols)
        row2col[ii] = jj
        col2row[jj] = ii
    end
    return LinkMatrix(row2col, col2row, length(mrows), nrow, ncol)
end

"""
    ==(C1::LinkMatrix, C2::LinkMatrix)

Return true if C1 and C2 match
"""
function ==(C1::LinkMatrix, C2::LinkMatrix)
    if C1.nrow != C2.nrow
        return false
    elseif C1.ncol != C2.ncol
        return false
    elseif C1.nlink != C2.nlink
        return false
    else
        for (row1, row2) in zip(C1.row2col, C2.row2col)
            if row1 != row2
                return false
            end
        end
    
        for (col1, col2) in zip(C1.col2row, C2.col2row)
            if col1 != col2
                return false
            end
        end
    end
    return true
end

"""
    has_link(row::G, col::G, C::LinkMatrix) where G <: Integer

Return `true` if row and column are linked and `false` otherwise.

See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
function has_link(row::G, col::G, C::LinkMatrix) where G <: Integer
    if izero(row) || iszero(col)
        return false
    elseif (C.row2col[row] != col)
        return false
    elseif (C.col2row[col] != row)
        @warn "row2col and col2row are inconsistent"
    else
        return true
    end
end

"""
    add_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Update `C` to link `row` and `col` checking that neither is already linked.

See also: [`add_link`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
function add_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(row)
        @warn "Row is zero, no addition made"
    elseif iszero(col)
        @warn "Col is zero, no addition made"
    elseif !iszero(C.row2col[row])
        @warn "row non-empty no addition made"
    elseif !iszero(C.col2row[col])
        @warn "col non-empty no addition made"
    else
        C.row2col[row] = col
        C.col2row[col] = row
        C.nlink += one(G)
    end
    return C
end

"""
    add_link(row::G, col::G, C::LinkMatrix) where G <: Integer

Update a copy of `C` to link `row` and `col` checking that neither is already linked.

See also: [`add_link!`](@ref), [`remove_link`](@ref), [`rowswitch_link`](@ref), [`colswitch_link`](@ref), [`doubleswitch_link`](@ref), [`LinkMatirx`](@ref)
"""
add_link(row::G, col::G, C::LinkMatrix) where G <: Integer = add_link!(row, col, deepcopy(C))

"""
    remove_link!(row::G, col::G, C::LinkMatrix) where G <: Integer

Remove `row` and `col` link checking that link exists.

See also: [`add_link!`](@ref), [`remove_link`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
function remove_link!(row::G, col::G, C::LinkMatrix) where G <: Integer
    if iszero(row)
        @warn "row is zero, no removal"
    elseif iszero(col)
        @warn "col is zero, no removal"
    elseif C.row2col[row] != col
        @warn "row column pair not linked, no removal"
    elseif C.col2row[col] != row
        @warn "row2col and col2row inconsistent debugging needed"
    else
        C.row2col[row] = zero(G)
        C.col2row[col] = zero(G)
        C.nlink -= one(G)
    end
    return C
end

"""
    remove_link(row::G, col::G, C::LinkMatrix) where G <: Integer

Copy `C` and remove `row` and `col` link checking that link exists.

See also: [`add_link`](@ref), [`remove_link!`](@ref), [`rowswitch_link`](@ref), [`colswitch_link`](@ref), [`doubleswitch_link`](@ref), [`LinkMatirx`](@ref)
"""
remove_link(row::G, col::G, C::LinkMatrix) where G <: Integer = remove_link!(row, col, deepcopy(C))

"""
    rowswitch_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Check that `col` is currently linked and change link from `C.col2row[col]`, `col`. Only performed if `row` currently unassigned.


See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
function rowswitch_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(row)
        @warn "Row is zero, no switch made"
    elseif iszero(col)
        @warn "Col is zero, no switch made"
    elseif iszero(C.col2row[col])
        @warn "Col not linked, no switch made"
    elseif !iszero(C.row2col[row])
        @warn "Row already assigned, no switch made"
    else
        C.row2col[C.col2row[col]] = zero(G)
        C.col2row[col] = row
        C.row2col[row] = col
    end
    return C
end

"""
    rowswitch_link(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Copy `C`, check that `col` is currently linked and change link from `C.col2row[col]`, `col`. Only performed if `row` currently unassigned.

See also: [`add_link`](@ref), [`remove_link`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link`](@ref), [`doubleswitch_link`](@ref), [`LinkMatirx`](@ref)
"""
rowswitch_link(row::G, col::G, C::LinkMatrix{G}) where G <: Integer = rowswitch_link!(row, col, deepcopy(C))

"""
    colswitch_link!(row::G, newcol::G, C::LinkMatrix{G}) where G <: Integer

Check that `row` is currently linked and change link from `row`, `C.row2col[row]`. Only performed if `col` currently unassigned.

See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
function colswitch_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(row)
        @warn "Row is zero, no switch made"
    elseif iszero(col)
        @warn "Col is zero, no switch made"
    elseif iszero(C.row2col[row])
        @warn "Row not linked, no switch made"
    elseif !iszero(C.col2row[col])
        @warn "Col already assigned, no switch made"
    else
        C.col2row[C.row2col[row]] = zero(G)
        C.row2col[row] = col
        C.col2row[col] = row
    end
    return C
end

"""
    colswitch_link(row::G, col::G, C::LinkMatrix{G}) where G <: Integer

Copy `C` and checks that `row` is currently linked and change link from `row`, `C.row2col[row]`. Only performed if `col` currently unassigned.

See also: [`add_link`](@ref), [`remove_link`](@ref), [`rowswitch_link`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link`](@ref), [`LinkMatirx`](@ref)
"""
colswitch_link(row::G, col::G, C::LinkMatrix{G}) where G <: Integer = colswitch_link!(row, col, deepcopy(C))

"""
   doubleswitch_link!(newrow::G, newcol::G, C::LinkMatrix) where G <: Integer

Performed where `newrow` currnetly linked to `col` and `row` currently linked to `newcol`.  Switches links to `newrow`, `newcol` and `row`, `col`

See also: [`add_link!`](@ref), [`remove_link!`](@ref), [`rowswitch_link!`](@ref), [`colswitch_link!`](@ref), [`doubleswitch_link`](@ref), [`LinkMatirx`](@ref)
"""
function doubleswitch_link!(newrow::G, newcol::G, C::LinkMatrix) where G <: Integer
    if iszero(newrow)
        @warn "Newrow is zero, no switch made"
    elseif iszero(newcol)
        @warn "Newcol is zero, no switch made"
    elseif iszero(C.row2col[newrow])
        @warn "Newrow not linked, no switch made"
    elseif iszero(C.col2row[newcol])
        @warn "New col not linked, no switch made"
    elseif C.row2col[newrow] == newcol
        @warn "Newrow and newcol already linked, no switch made"
    else
        col = C.row2col[newrow]
        row = C.col2row[newcol]
        C.row2col[row] = col
        C.col2row[col] = row
        C.row2col[newrow] = newcol
        C.col2row[newcol] = newrow
    end
    return C
end

"""
   doubleswitch_link(newrow::G, newcol::G, C::LinkMatrix) where G <: Integer

Copy `C` and then where `newrow` currnetly linked to `col` and `row` currently linked to `newcol`.  Switches links to `newrow`, `newcol` and `row`, `col`

See also: [`add_link`](@ref), [`remove_link`](@ref), [`rowswitch_link`](@ref), [`colswitch_link`](@ref), [`doubleswitch_link!`](@ref), [`LinkMatirx`](@ref)
"""
doubleswitch_link(newrow::G, newcol::G, C::LinkMatrix) where G <: Integer = doubleswitch_link!(newrow, newcol, deepcopy(C))

"""
    tuple2links(rows::Array{G, 1}, cols::Array{G, 1}, nrow::G) where G <: Integer

Transform a tuple of link pairs into row2col format of a `LinkMatrix` object.

See also: [`links2tuple`](@ref)
"""
function tuple2links(rows::Array{G, 1}, cols::Array{G, 1}, nrow::G) where G <: Integer
    row2col = zeros(G, nrow)
    for (ii, jj) in zip(rows, cols)
        row2col[ii] = jj
    end
    return row2col
end

"""
    links2tuple(row2col::Array{G, 1}) where G <: Integer
    links2tuple(C::LinkMatrix{G}) where G <: Integer

Transform a LinkMatrix or a row2col mapping into a tuple of link pairs.

See also: [`tuple2links`](@ref)
"""
function links2tuple(row2col::Array{G, 1}) where G <: Integer
    rows = G[]
    cols = G[]
    for (row, col) in enumerate(row2col)
        if !iszero(col)
            push!(rows, row)
            push!(cols, col)
        end
    end
    return rows, cols
end

function links2tuple(C::LinkMatrix{G}) where G <: Integer
    rows = Array{G}(undef, C.nlink)
    cols = Array{G}(undef, C.nlink)
    
    row = 1
    ii = 1
    
    while ii <= C.nlink
        if !iszero(C.row2col[ii])
            rows[ii] = row
            cols[ii] = C.row2col[ii]
            ii += 1
        end
        row += 1
    end
    return rows, cols
end

"""
    row2col_removed(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    row2col_removed(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_removed(newC.row2col, oldC.row2col)

Return a tuple of vectors (rowsremoved, colsremoved) containing the row, col pairs of links present in the old set of links but not in the new set of links.

See also: [`row2col_added`](@ref), [`row2col_difference`](@ref)
"""
function row2col_removed(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    if length(newrow2col) != length(oldrow2col)
        @error "lengths must match"
    end
    rowsremoved = G[]
    colsremoved = G[]
    for row in 1:length(newrow2col)
        if newrow2col[row] != oldrow2col[row]
            if !iszero(oldrow2col[row])
                push!(rowsremoved, row)
                push!(colsremoved, oldrow2col[row])
            end
        end
    end
    return rowsremoved, colsremoved
end

row2col_removed(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_removed(newC.row2col, oldC.row2col)

"""
    row2col_removed(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    row2col_removed(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_removed(newC.row2col, oldC.row2col)

Return a tuple of vectors (rowsadded, colsadded) containing the row, col pairs of links present in the new set of links but not in the old set of links.

See also: [`row2col_removed`](@ref), [`row2col_difference`](@ref)
"""
function row2col_added(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    if length(newrow2col) != length(oldrow2col)
        @error "lengths must match"
    end
    rowsadded = G[]
    colsadded = G[]
    for row in 1:length(newrow2col)
        if newrow2col[row] != oldrow2col[row]
            if !iszero(newrow2col[row])
                push!(rowsadded, row)
                push!(colsadded, newrow2col[row])
            end
        end
    end
    return rowsadded, colsadded
end

row2col_added(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_added(newC.row2col, oldC.row2col)


"""
    row2col_difference(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    row2col_difference(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_removed(newC.row2col, oldC.row2col)

Return a tuple of vectors (rowsremoved, colsremoved, rowsadded, colsadded) containing the row, col pairs of links present in only one of the two sets. Equivlant to calling `row2col_removed` and `row2col_added` but slightly more efficient as only a single pass through each array is made.

See also: [`row2col_removed`](@ref), [`row2col_added`](@ref)
"""
function row2col_difference(newrow2col::Array{G, 1}, oldrow2col::Array{G, 1}) where G <: Integer
    if length(newrow2col) != length(oldrow2col)
        @error "lengths must match"
    end
    
    rowsremoved = G[]
    colsremoved = G[]
    rowsadded = G[]
    colsadded = G[]
    
    for row in 1:length(newrow2col)
        if newrow2col[row] != oldrow2col[row]
            if !iszero(oldrow2col[row])
                push!(rowsremoved, row)
                push!(colsremoved, oldrow2col[row])
            end

            if !iszero(newrow2col[row])
                push!(rowsadded, row)
                push!(colsadded, newrow2col[row])
            end
        end
    end

    return rowsremoved, colsremoved, rowsadded, colsadded
    
end

row2col_difference(newC::LinkMatrix{G, 1}, oldC::LinkMatrix{G, 1}) where G <: Integer = row2col_difference(newC.row2col, oldC.row2col)

"""
    matched_comparisons(row2col::Array{G, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer
    matched_comparisons(C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer

Return a boolean vector with values of true for each comparison vector is at least one record pair with that comparison is matched
"""
function matched_comparisons(row2col::Array{G, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer
    matchobs = falses(length(compsum.obsvecct))
    for row in 1:length(row2col)
        if !iszero(row2col[row]) && row2col[row] <= compsum.ncol
            if !matchobs[compsum.obsidx[row, row2col[row]]]
                matchobs[compsum.obsidx[row, row2col[row]]] = true
            end
        end
    end
    return matchobs
end

matched_comparisons(C::LinkMatrix{G}, compsum::Union{ComparisonSummary, SparseComparisonSummary}) where G <: Integer = matched_comparisons(C.row2col, compsum)
