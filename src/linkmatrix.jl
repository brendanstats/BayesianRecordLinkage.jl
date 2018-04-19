mutable struct LinkMatrix{G <: Integer}
    #row2col::Array{G, 1}
    #col2row::Array{G, 1}
    row2col::SparseVector{G, G}
    col2row::SparseVector{G, G}
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

LinkMatrix{G <: Integer}(row2col::SparseVector{G, G}, col2row::SparseVector{G, G}) = LinkMatrix(row2col, col2row, G(countnz(row2col)), G(row2col.n), G(col2row.n))
function LinkMatrix{G <: Integer, T <: Integer}(row2col::SparseVector{G, T}, col2row::SparseVector{G, T})
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

LinkMatrix{G <: Integer}(row2col::Array{G, 1}, col2row::Array{G, 1}) = LinkMatrix(sparse(row2col), sparse(col2row))
LinkMatrix{G <: Integer}(row2col::Array{G, 1}, col2row::Array{G, 1}, nlink::G) = LinkMatrix(row2col, col2row, G(length(row2col)), G(length(col2row)))
LinkMatrix{G <: Integer}(nrow::G, ncol::G) = LinkMatrix(spzeros(G, nrow), spzeros(G, ncol), zero(G), nrow, ncol)

function LinkMatrix{G <: Integer}(nrow::G, ncol::G, mrows::Array{G, 1}, mcols::Array{G, 1})
    row2col = zeros(G, nrow)
    col2row = zeros(G, ncol)
    for (ii, jj) in zip(mrows, mcols)
        row2col[ii] = jj
        col2row[jj] = ii
    end
    return LinkMatrix(row2col, col2row, length(mrows), nrow, ncol)
end

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

function has_link{G <: Integer}(row::G, col::G, C::LinkMatrix)
    if (C.row2col[row] == col) && (C.col2row[col] == row)
        return true
    else
        return false
    end
end

function add_link!{G <: Integer}(row::G, col::G, C::LinkMatrix{G})
    if (!iszero(C.row2col[row])) || (!iszero(C.col2row[col]))
        warn("row or column already contains a link, no addition made")
    else
        C.row2col[row] = col
        C.col2row[col] = row
        C.nlink += one(G)
    end
    return C
end

add_link{G <: Integer}(row::G, col::G, C::LinkMatrix) = add_link!(row, col, deepcopy(C))

function remove_link!{G <: Integer}(row::G, col::G, C::LinkMatrix)
    if iszero(C.row2col[row]) || iszero(C.col2row[col])
        warn("row column pair not linked, no removal")
    else
        C.row2col[row] = zero(G)
        dropzeros!(C.row2col)
        C.col2row[col] = zero(G)
        dropzeros!(C.col2row)
        C.nlink -= one(G)
    end
    return C
end

remove_link{G <: Integer}(row::G, col::G, C::LinkMatrix) = remove_link!(row, col, deepcopy(C))

function switch_link!{G <: Integer}(row1::G, col1::G, row2::G, col2::G, C::LinkMatrix)
    if (C.row2col[row1] != col1) || (C.row2col[row2] != col2)
        warn("provided pairs not current links, no switch")
    end
    C.row2col[row1] = col2
    C.row2col[row2] = col1
    C.col2row[col1] = row2
    C.col2row[col2] = row1
    return C
end

switch_link{G <: Integer}(row1::G, col1::G, row2::G, col2::G, C::LinkMatrix) = switch_link!(row1, col1, row2, col2, deepcopy(C))
