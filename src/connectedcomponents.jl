"""
    ConnectedComponents{G <: Integer}

Summarize results of bipartite graph clustering algorithm where rows correspond to one set of nodes and columns the other.

# Fields

* `rowLabels::Array{G, 1}`: Component assignment of row, zero indicates no edges from row.
* `colLabels::Array{G, 1}`: Component assignment of col, zero indicates no edges from column.
* `rowperm::Array{G, 1}`: Permutation which will reorder rows so that labels are ascending.
* `colperm::Array{G, 1}`: Permutation which will reorder columns so that labels are ascending.
* `rowcounts::Array{G, 1}`: Number of rows contained in each component, because counts for
unassigned (label == 0) rows are included rowcounts[kk + 1] contains the count for component kk.
* `colcounts::Array{G, 1}`: Number of columns contained in each component, because counts for
unassigned (label == 0) columns are included colcounts[kk + 1] contains the count for component kk.
* `cumrows::Array{G, 1}`: Cumulative sum of `rowcounts`.
* `cumcols::Array{G, 1}`: Cumulative sum of `colcounts`.
* `nrow::G`: Number of rows (nodes from first group).
* `ncol::G`: Number of columns (nodes from second group).
* `ncomponents::G`: Number of components.

# Constructors

    ConnectedComponents(rowLabels::Array{G, 1}, colLabels::Array{G, 1}, ncomponents::G = maximum(rowLabels), nrow::G = length(rowLabels), ncol::G = length(colLabels)) where G <: Integer

See also: [`bipartite_cluster`](@ref), [`iterative_bipartite_cluster`](@ref), [`iterative_bipartite_cluster2`](@ref)
"""
struct ConnectedComponents{G <: Integer}
    rowLabels::Array{G, 1}
    colLabels::Array{G, 1}
    rowperm::Array{G, 1}
    colperm::Array{G, 1}
    
    rowcounts::Array{G, 1}
    colcounts::Array{G, 1}
    cumrows::Array{G, 1}
    cumcols::Array{G, 1}
    
    nrow::G
    ncol::G
    ncomponents::G
end

function ConnectedComponents(rowLabels::Array{G, 1}, colLabels::Array{G, 1}, ncomponents::G = maximum(rowLabels), nrow::G = length(rowLabels), ncol::G = length(colLabels)) where G <: Integer

    rowperm = sortperm(rowLabels)
    colperm = sortperm(colLabels)    
    
    #off by one because zero is included as a label
    rowcounts = counts(rowLabels, 0:ncomponents)
    colcounts = counts(colLabels, 0:ncomponents)
    cumrows = cumsum(rowcounts)
    cumcols = cumsum(colcounts)

    return ConnectedComponents(rowLabels, colLabels, rowperm, colperm, rowcounts, colcounts, cumrows, cumcols, nrow, ncol, ncomponents)
end

"""
   get_component(ii::Integer, cc::ConnectedComponents)

Return array of rows and columns assigned to component `ii`.
"""
function get_component(ii::Integer, cc::ConnectedComponents)
    if ii < zero(ii)
        @warn "all labels are positive"
        rows = Int[]
        cols = Int[]
    elseif ii > cc.ncomponents
        warn("Label greater than number of connected components")
        rows = Int[]
        cols = Int[]
    elseif ii == zero(ii)
        rows = cc.rowLabels[cc.rowperm[1:cc.cumrows[1]]]
        cols = cc.colLabels[cc.colperm[1:cc.cumcols[1]]]
    else #+ 1 in range length to account for 0 labels
        rows = cc.rowperm[range(cc.cumrows[ii] + 1, cc.rowcounts[ii + 1])]
        cols = cc.colperm[range(cc.cumcols[ii] + 1, cc.colcounts[ii + 1])]
    end
    return rows, cols
end

"""
    get_ranges(cc::ConnectedComponents)

If `cc.rowLabels` and `c.colLabels` are sorted return array of `CartesianIndicies{1}` giving combined row and column range for each component.
"""
function get_ranges(cc::ConnectedComponents)
    if issorted(cc.rowLabels) && issorted(cc.colLabels)
        out = Array{CartesianIndices{2}, 1}(undef, cc.ncomponents)
        for ii in 1:cc.ncomponents
            rowrange = range(cc.cumrows[ii] + 1,cc.rowcounts[ii + 1])
            colrange = range(cc.cumcols[ii] + 1,cc.colcounts[ii + 1])
            out[ii] = CartesianIndices((rowrange, colrange))
        end
    else
        error("input must be sorted by row and column cluster to summarize with ranges")
    end
    return out
end

#off by one because first component is labeled zero
"""
    get_dimensions(ii::Integer, cc::ConnectedComponents)

Return number of rows and columns in component as a tuple.
"""
function get_dimensions(ii::Integer, cc::ConnectedComponents)
    return cc.rowcounts[ii + 1], cc.colcounts[ii + 1]
end

"""
    get_mids(x::Array{<:Real, 1})

Sort unique entries in array and then return midpoints between each entry.
"""
function get_mids(x::Array{<:Real, 1})
    sx = sort(unique(x))
    return 0.5 .* (sx[1:end-1] + sx[2:end])
end

"""
    count_pairs(cc::ConnectedComponents)

Sum product of rows and columns across components, equivalent to number of edges if each componenet is complete.
"""
function count_pairs(cc::ConnectedComponents)
    return dot(cc.rowcounts[2:end], cc.colcounts[2:end])
end

"""
    maxcomponent_pairs(cc::ConnectedComponents)

Return the maximum component size as measured by the product of the number of rows in the component and the columns in the component.
"""
function maxcomponent_pairs(cc::ConnectedComponents)

    maxpairs = 0#cc.rowcounts[1] .* cc.colcounts[1]
    for ii in 1:cc.ncomponents
        if *(get_dimensions(ii, cc)...) > maxpairs
            maxpairs = *(get_dimensions(ii, cc)...)
        end
    end
    return maxpairs
end

"""
   maxdimension(cc::ConnectedComponents)

Return the largest number of rows or columns observed in a single component.
"""
function maxdimension(cc::ConnectedComponents)
    if cc.ncomponents == 0
        return 0
    else
        return max(maximum(cc.rowcounts[2:end]), maximum(cc.colcounts[2:end]))
    end
end

"""
    count_singleton(cc::ConnectedComponents)

Count the number of components containing only a single entry (one row and one column).
"""
function count_singleton(cc::ConnectedComponents)
    nsingleton = 0
    for ii in 1:cc.ncomponents
        if cc.rowcounts[ii + 1] == 1 && cc.colcounts[ii + 1] == 1
            nsingleton += 1
        end
    end
    return nsingleton
end

"""
    summarize_components(cc::ConnectedComponents)

Summarize `cc` returning results in a vector.

Returned vector contains the results of `count_pairs`, `maxcomponent_pairs`, `maxdimension`,
the number of components in `cc` and the results of `count_singleton`.

See also[`count_pairs`](@ref), [`maxcomponent_pairs`](@ref), [`maxdimension`](@ref), [`count_singleton`](@ref)
"""
function summarize_components(cc::ConnectedComponents)
    return [count_pairs(cc), maxcomponent_pairs(cc), maxdimension(cc), cc.ncomponents, count_singleton(cc)]
end
