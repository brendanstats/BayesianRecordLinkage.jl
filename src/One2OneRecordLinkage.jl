#__precompile__()
module One2OneRecordLinkage
using Munkres, RCall, AssignmentSolver
clue = rimport("clue")

using Distributions.Dirichlet
using Levenshtein.levenshtein
using DataStructures: Queue, enqueue!, dequeue!
using StatsBase
using StatsFuns.logistic
import Base.==

export ComparisonSummary, counts_delta, SparseComparisonSummary
export LinkMatrix, add_link, add_link!, remove_link!, remove_link
export E_step, M_step, estimate_EM
export bipartite_cluster, sparseblock_idxlims, bipartite_cluster_sparseblock
export counts_matches,
    max_MU,
    weights_vector,
    weights_matrix,
    maximum_weights_vector,
    maximum_weights_matrix,
    penalized_weights_matrix,
    indicator_weights_matrix,
    compute_costs,
    max_C,
    max_C_offsets,
    max_C_initialized!,
    max_C_cluster,
    lsap_checkoptimal,
    map_solver,
    map_solver_initialize,
    map_solver_cluster,
    map_solver_iter,
    map_solver_iter_cluster,
    map_solver_iter_initialize,
    map_solver_search,
    map_solver_search_cluster,
    map_solver_search_initialize
export randomwalk1_draw,
    identity_balance,
    sqrt_balance,
    barker_balance,
    globally_balanced_draw,
    locally_balanced_sqrt_draw,
    locally_balanced_barker_draw,
    randomwalk2_draw
export exppenalty_logratio, betabipartite_logratio
export dirichlet_draw, mh_gibbs_chain, mh_gibbs_count

include("comparisonsummary.jl")
include("linkmatrix.jl")
include("em_functions.jl")
include("clustering_functions.jl")
include("map_functions.jl")
include("move_functions.jl")
include("prior_functions.jl")
include("mcmc.jl")
end
