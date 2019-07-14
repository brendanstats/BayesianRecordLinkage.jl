function h5write_ComparisonSummary(filename::String,
                                   compsum::ComparisonSummary,
                                   groupname::String = "/",
                                   mode::String = "w",
                                   complevel::Integer = 7)
    h5open(filename, mode) do writef
        writef[groupname * "/" * "obsidx", "chunk", (size(compsum.obsidx)), "shuffle", (), "compress", complevel] = compsum.obsidx
        writef[groupname * "/" * "obsvecs"] = compsum.obsvecs
        writef[groupname * "/" * "obsvecct"] = compsum.obsvecct
        writef[groupname * "/" * "counts"] = compsum.counts
        writef[groupname * "/" * "obsct"] = compsum.obsct
        writef[groupname * "/" * "misct"] = compsum.misct
        writef[groupname * "/" * "nlevels"] = compsum.nlevels
        writef[groupname * "/" * "cmap"] = compsum.cmap
        writef[groupname * "/" * "levelmap"] = compsum.levelmap
        writef[groupname * "/" * "cadj"] = compsum.cadj
        writef[groupname * "/" * "nrow"] = compsum.nrow
        writef[groupname * "/" * "ncol"] = compsum.ncol
        writef[groupname * "/" * "npairs"] = compsum.npairs
        writef[groupname * "/" * "ncomp"] = compsum.ncomp
    end
    nothing
end

function h5read_ComparisonSummary(filename::String;
                                  groupname::String = "/")
    return ComparisonSummary(
        h5read(filename, groupname * "/" * "obsidx"),
        h5read(filename, groupname * "/" * "obsvecs"),
        h5read(filename, groupname * "/" * "obsvecct"),
        h5read(filename, groupname * "/" * "counts"),
        h5read(filename, groupname * "/" * "obsct"),
        h5read(filename, groupname * "/" * "misct"),
        h5read(filename, groupname * "/" * "nlevels"),
        h5read(filename, groupname * "/" * "cmap"),
        h5read(filename, groupname * "/" * "levelmap"),
        h5read(filename, groupname * "/" * "cadj"),
        h5read(filename, groupname * "/" * "nrow"),
        h5read(filename, groupname * "/" * "ncol"),
        h5read(filename, groupname * "/" * "npairs"),
        h5read(filename, groupname * "/" * "ncomp")
    )
end

function h5write_SparseComparisonSummary(filename::String,
                                         compsum::SparseComparisonSummary;
                                         groupname::String = "/",
                                         mode::String = "w",
                                         maxlen::Integer = 600000000,
                                         complevel::Integer = 7)
    chunklen = min(maxlen, length(compsum.obsidx.rowval))
    h5open(filename, mode) do writef
        writef[groupname * "/" * "m"] = compsum.obsidx.m
        writef[groupname * "/" * "n"] = compsum.obsidx.n
        writef[groupname * "/" * "colptr"] = compsum.obsidx.colptr
        writef[groupname * "/" * "rowval", "chunk", (chunklen), "shuffle", (), "compress", complevel] = compsum.obsidx.rowval
        writef[groupname * "/" * "nzval", "chunk", (chunklen), "shuffle", (), "compress", complevel] = compsum.obsidx.nzval
        writef[groupname * "/" * "obsvecs"] = compsum.obsvecs
        writef[groupname * "/" * "obsvecct"] = compsum.obsvecct
        writef[groupname * "/" * "counts"] = compsum.counts
        writef[groupname * "/" * "obsct"] = compsum.obsct
        writef[groupname * "/" * "misct"] = compsum.misct
        writef[groupname * "/" * "nlevels"] = compsum.nlevels
        writef[groupname * "/" * "cmap"] = compsum.cmap
        writef[groupname * "/" * "levelmap"] = compsum.levelmap
        writef[groupname * "/" * "cadj"] = compsum.cadj
        writef[groupname * "/" * "nrow"] = compsum.nrow
        writef[groupname * "/" * "ncol"] = compsum.ncol
        writef[groupname * "/" * "npairs"] = compsum.npairs
        writef[groupname * "/" * "ncomp"] = compsum.ncomp
    end
    nothing
end

function h5read_SparseComparisonSummary(filename::String;
                                        groupname::String = "/")
    return SparseComparisonSummary(
        SparseMatrixCSC(h5read(filename, groupname * "/" * "m"),
                        h5read(filename, groupname * "/" * "n"),
                        h5read(filename, groupname * "/" * "colptr"),
                        h5read(filename, groupname * "/" * "rowval"),
                        h5read(filename, groupname * "/" * "nzval")
                        ),
        h5read(filename, groupname * "/" * "obsvecs"),
        h5read(filename, groupname * "/" * "obsvecct"),
        h5read(filename, groupname * "/" * "counts"),
        h5read(filename, groupname * "/" * "obsct"),
        h5read(filename, groupname * "/" * "misct"),
        h5read(filename, groupname * "/" * "nlevels"),
        h5read(filename, groupname * "/" * "cmap"),
        h5read(filename, groupname * "/" * "levelmap"),
        h5read(filename, groupname * "/" * "cadj"),
        h5read(filename, groupname * "/" * "nrow"),
        h5read(filename, groupname * "/" * "ncol"),
        h5read(filename, groupname * "/" * "npairs"),
        h5read(filename, groupname * "/" * "ncomp")
    )
end

function h5write_ConnectedComponents(filename::String,
                                     cc::ConnectedComponents;
                                     groupname::String = "/",
                                     mode::String = "w")
        h5open(filename, mode) do writef
            writef[groupname * "/" * "rowLabels"] = cc.rowLabels
            writef[groupname * "/" * "colLabels"] = cc.colLabels
            writef[groupname * "/" * "rowperm"] = cc.rowperm
            writef[groupname * "/" * "colperm"] = cc.colperm
            writef[groupname * "/" * "rowcounts"] = cc.rowcounts
            writef[groupname * "/" * "colcounts"] = cc.colcounts
            writef[groupname * "/" * "cumrows"] = cc.cumrows
            writef[groupname * "/" * "cumcols"] = cc.cumcols
            writef[groupname * "/" * "nrow"] = cc.nrow
            writef[groupname * "/" * "ncol"] = cc.ncol
            writef[groupname * "/" * "ncomponents"] = cc.ncomponents
        end
    nothing
end

function h5read_ConnectedComponents(filename::String; groupname::String = "/")
    return ConnectedComponents(
        h5read(filename, groupname * "/" * "rowLabels"),
        h5read(filename, groupname * "/" * "colLabels"),
        h5read(filename, groupname * "/" * "rowperm"),
        h5read(filename, groupname * "/" * "colperm"),
        h5read(filename, groupname * "/" * "rowcounts"),
        h5read(filename, groupname * "/" * "colcounts"),
        h5read(filename, groupname * "/" * "cumrows"),
        h5read(filename, groupname * "/" * "cumcols"),
        h5read(filename, groupname * "/" * "nrow"),
        h5read(filename, groupname * "/" * "ncol"),
        h5read(filename, groupname * "/" * "ncomponents")
    )
end

function h5write_ParameterChain(filename::String,
                                pchain::ParameterChain;
                                groupname::String = "/",
                                mode::String = "w",
                                maxlen::Integer = 100000000,
                                complevel::Integer = 7)
    chunklen = min(maxlen, size(pchain.C, 1))
    h5open(filename, mode) do writef
        if !exists(writef, groupname)
            g_create(writef, groupname)
        end
        writef[groupname * "/C", "chunk", (chunklen, size(pchain.C, 2)), "shuffle", (), "compress", complevel] = pchain.C
        writef[groupname * "/nlinks"] = pchain.nlinks
        writef[groupname * "/pM"] = pchain.pM
        writef[groupname * "/pU"] = pchain.pU
        writef[groupname * "/nsteps"] = pchain.nsteps
        writef[groupname * "/linktrace"] = pchain.linktrace
    end
    nothing
end

function h5read_ParameterChain(filename::String,
                               groupname::String = "/",
                               mode::String = "r")
    return ParameterChain(h5read(filename, groupname * "/C"),
                          h5read(filename, groupname * "/nlinks"),
                          h5read(filename, groupname * "/pM"),
                          h5read(filename, groupname * "/pU"),
                          h5read(filename, groupname * "/nsteps"),
                          h5read(filename, groupname * "/linktrace"))
end

function h5write_PosthocBlocks(filename::String,
                               phb::PosthocBlocks;
                               groupname::String = "/",
                               mode::String = "w") where T <: AbstractFloat
    h5open(filename, mode) do writef
        for (ky, vl) in phb.block2rows
            writef[groupname * "/" * "rows/$ky"] = vl
        end
        for (ky, vl) in phb.block2cols
            writef[groupname * "/" * "cols/$ky"] = vl
        end
        writef[groupname * "/" * "blocknrows"] = phb.blocknrows
        writef[groupname * "/" * "blockncols"] = phb.blockncols
        writef[groupname * "/" * "blocksingleton"] = phb.blocksingleton
        writef[groupname * "/" * "blocknnz"] = phb.blocknnz
        writef[groupname * "/" * "nrow"] = phb.nrow
        writef[groupname * "/" * "ncol"] = phb.ncol
        writef[groupname * "/" * "nblock"] = phb.nblock
        writef[groupname * "/" * "nnz"] = phb.nnz
    end
    nothing
end

function h5read_PosthocBlocks(filename::String,
                              groupname::String = "/",
                              mode::String = "r") where T <: AbstractFloat
    return h5open(filename, mode) do readf
        #G = eltype(readf[groupname * "/" * "rows"][names(readf[groupname * "/" * "rows"])[1]])
        nblock = read(readf[groupname * "/nblock"])
        G = eltype(nblock)
        block2rows = Dict{G, Array{G, 1}}()
        block2cols = Dict{G, Array{G, 1}}()
        for ky in zero(G):nblock
            if exists(readf, groupname * "/rows/$ky")
                block2rows[ky] = read(readf[groupname * "/rows/$ky"])
                block2cols[ky] = read(readf[groupname * "/cols/$ky"])
            end
        end
        PosthocBlocks(block2rows,
                      block2cols,
                      read(readf[groupname * "/blocknrows"]),
                      read(readf[groupname * "/blockncols"]),
                      read(readf[groupname * "/blocksingleton"]),
                      read(readf[groupname * "/blocknnz"]),
                      read(readf[groupname * "/nrow"]),
                      read(readf[groupname * "/ncol"]),
                      nblock,
                      read(readf[groupname * "/nnz"]))
    end
end

function h5write_penalized_likelihood_estimate(filename::String,
                                               priorM::Array{T, 1},
                                               priorU::Array{T, 1},
                                               pM0::Array{T, 1},
                                               pU0::Array{T, 1},
                                               matches::Dict{Int64,Tuple{Array{Int64,1},Array{Int64,1}}},
                                               pM::Array{T, 2},
                                               pU::Array{T, 2},
                                               penalties::Array{T, 1},
                                               runtime::Real = -1.0;
                                               groupname::String = "/",
                                               mode::String = "w") where T <: AbstractFloat
    nlinks = [length(matches[ii][1]) for ii in 1:length(penalties)]
    h5open(filename, mode) do writef
        writef[groupname * "/" * "priorM"] = priorM
        writef[groupname * "/" * "priorU"] = priorU
        writef[groupname * "/" * "penalties"] = penalties
        writef[groupname * "/" * "nlinks"] = nlinks
        for (ky, vl) in matches
            writef[groupname * "/" * "matches/$ky/mrows"] = vl[1]
            writef[groupname * "/" * "matches/$ky/mcols"] = vl[2]
        end
        writef[groupname * "/" * "pM"] = pM
        writef[groupname * "/" * "pU"] = pU
        writef[groupname * "/" * "pM0"] = pM0
        writef[groupname * "/" * "pU0"] = pU0
        writef[groupname * "/" * "runtime"] = runtime
    end
    nothing
end

function h5write_clustering_diagnostics(filename::String,
                                        ncluster::Array{<:Integer, 1},
                                        maxdim::Array{<:Integer, 1},
                                        npairs::Array{<:Integer, 1},
                                        npairscomp::Array{<:Integer, 1},
                                        nsingle::Array{<:Integer, 1},
                                        ndouble::Array{<:Integer, 1},
                                        penalties::Array{<:AbstractFloat, 1};
                                        groupname::String = "/",
                                        mode::String = "w")
    h5open(filename, mode) do writef
        writef[groupname * "/" * "ncluster"] = ncluster
        writef[groupname * "/" * "maxdim"] = maxdim
        writef[groupname * "/" * "nsingle"] = nsingle
        writef[groupname * "/" * "ndouble"] = ndouble
        writef[groupname * "/" * "npairs"] = npairs
        writef[groupname * "/" * "npairscomp"] = npairscomp
        writef[groupname * "/" * "penalties"] = penalties
    end
    nothing
end
