"""
This types does...

### Constructors

    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
mutable struct LinkMatrix{G <: Integer}
    row2col::Array{G, 1}
    col2row::Array{G, 1}
    #row2col::SparseVector{G, G}
    #col2row::SparseVector{G, G}
    nlink::G
    nrow::G
    ncol::G
    #if length(row2col) != nrow
    #    error("Length of row2col must link nrow")
    #elseif length(col2row) != ncol
    #    error("Length of col2row must link ncol")
    #else
    #    LinkMatrix{G}(row2col, col2row, nlink, nrow, ncol) = new(row2col, col2row, nlink, nrow, ncol)
    #end
end

LinkMatrix(nrow::G, ncol::G) where G <: Integer = LinkMatrix(zeros(G, nrow), zeros(G, ncol), zero(G), nrow, ncol)
LinkMatrix(row2col::Array{G, 1}, col2row::Array{G, 1}, nlink::G) where G <: Integer = LinkMatrix(row2col, col2row, G(length(row2col)), G(length(col2row)))
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

LinkMatrix(row2col::SparseVector{G, G}, col2row::SparseVector{G, G}) where G <: Integer = LinkMatrix(row2col, col2row, G(countnz(row2col)), G(row2col.n), G(col2row.n))
function LinkMatrix(row2col::SparseVector{G, T}, col2row::SparseVector{G, T}) where {G <: Integer, T <: Integer}
    if countnz(row2col) != countnz(col2row)
        error("inconsistent row and column mappings provided")
    end
    if typemax(G) >= typemax(T)
        return LinkMatrix(SparseVector(row2col.n, G.(row2col.nzind), row2col.nzval), SparseVector(col2row.n, G.(col2row.nzind), col2row.nzval))
    else
        return LinkMatrix(SparseVector(row2col.n, row2col.nzind, T.(row2col.nzval)), SparseVector(col2row.n, col2row.nzind, T.(col2row.nzval)))
    end
end

function LinkMatrix(G::DataType, row2col::SparseVector, col2row::SparseVector)
    if countnz(row2col) != countnz(col2row)
        error("inconsistent row and column mappings provided")
    end
    if !(G <: Integer)
        error("G must be an integer")
    end
    return LinkMatrix(SparseVector(row2col.n, G.(row2col.nzind), G.(row2col.nzval)), SparseVector(col2row.n, G.(col2row.nzind), G.(col2row.nzval)))
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
function ==(C1::LinkMatrix, C2::LinkMatrix)
    if C1.nrow != C2.nrow
        return false
    end
    
    if C1.ncol != C2.ncol
        return false
    end
    
    if C1.nlink != C2.nlink
        return false
    end
    
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
    return true
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
function has_link(row::G, col::G, C::LinkMatrix) where G <: Integer
    if izero(row) || iszero(col)
        return false
    elseif (C.row2col[row] != col) || (C.col2row[col] != row)
        return false
    else
        return false
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
function add_link!(row::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if !iszero(C.row2col[row])
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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
add_link(row::G, col::G, C::LinkMatrix) where G <: Integer = add_link!(row, col, deepcopy(C))

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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
remove_link(row::G, col::G, C::LinkMatrix) where G <: Integer = remove_link!(row, col, deepcopy(C))

function rowswitch_link!(newrow::G, col::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(C.col2row[col])
        @warn "Col not linked, no switch made"
    elseif !iszero(C.row2col[newrow])
        @warn "Newrow already assigned, no switch made"
    else
        C.row2col[C.col2row[col]] = zero(G)
        C.col2row[col] = newrow
        C.row2col[newrow] = col
    end
    return C
end

rowswitch_link(newrow::G, col::G, C::LinkMatrix{G}) where G <: Integer = rowswitch_link!(newrow, col, deepcopy(C))

function colswitch_link!(row::G, newcol::G, C::LinkMatrix{G}) where G <: Integer
    if iszero(C.row2col[row])
        @warn "Row not linked, no switch made"
    elseif !iszero(C.col2row[newcol])
        @warn "Newcol already assigned, no switch made"
        C.col2row[C.row2col[row]] = zero(G)
        C.row2col[row] = newcol
        C.col2row[newcol] = row
    end
    return C
end

colswitch_link(row::G, newcol::G, C::LinkMatrix{G}) where G <: Integer = colswitch_link!(row, newcol, deepcopy(C))

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
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
doubleswitch_link(row1::G, col1::G, row2::G, col2::G, C::LinkMatrix) where G <: Integer = doubleswitch_link!(row1, col1, row2, col2, deepcopy(C))
