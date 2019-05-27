module BayesianRecordLinkage
using AssignmentSolver, HDF5

using Distributions: Dirichlet
using DataStructures: Queue, enqueue!, dequeue!
using StatsBase
using StatsFuns: logit, logistic, log1pexp, loghalf, softmax, logsumexp
using SparseArrays
using SpecialFunctions: lfactorial, lbeta
import Base: ==

export ComparisonSummary,
    SparseComparisonSummary,
    extract_integers,
    extract_integers!,
    stream_comparisonsummary,
    merge_comparisonsummary,
    counts_delta,
    obs_delta
export LinkMatrix,
    add_link, add_link!,
    remove_link!, remove_link,
    switch!_link, switch_link
export E_step, M_step, estimate_EM
export ConnectedComponents,
    get_component,
    label2dict,
    get_component,
    get_ranges,
    get_dimensions,
    get_mids,
    count_pairs,
    maxcomponent_pairs,
    count_singleton,
    maxdimension,
    summarize_components
export bipartite_cluster,
    sparseblock_idxlims,
    bipartite_cluster_sparseblock,
    iterative_bipartite_cluster,
    iterative_bipartite_cluster2
export counts_matches,
    weights_vector,
    penalized_weights_vector,
    weights_matrix,
    maximum_weights_vector,
    maximum_weights_matrix,
    penalized_weights_matrix,
    indicator_weights_matrix,
    compute_costs,
    bayesrule_posterior,
    threshold_sensitivity
export max_MU,
    max_C,
    max_C_offsets,
    max_C_initialized!,
    max_C_cluster,
    max_C_auction,
    max_C_auction_cluster
export map_solver,
    map_solver_initialize,
    map_solver_cluster,
    map_solver_auction,
    map_solver_auction_cluster
export map_solver_iter,
    map_solver_iter_cluster,
    map_solver_iter_initialize,
    map_solver_search,
    map_solver_search_cluster,
    map_solver_search_initialize,
    map_solver_search_auction,
    map_solver_search_auction_cluster
export exppenalty_logprior, betabipartite_logprior,
    exppenalty_logratio, betabipartite_logratio,
    exppenalty_logratiopn, betabipartite_logratiopn
export identity_balance, lidentity_balance, sqrt_balance, lsqrt_balance, barker_balance, lbarker_balance
export randomwalk1_draw,
    globally_balanced_draw, globally_balanced_draw!,
    locally_balanced_sqrt_draw, locally_balanced_sqrt_draw!,
    locally_balanced_barker_draw, locally_balanced_barker_draw!,
    randomwalk2_draw,
    get_loglik,
    get_counts
export dropoutside!, dirichlet_draw, mh_gibbs_chain, mh_gibbs_count, mh_gibbs_chain_inplace, mh_gibbs_count_inplace
export h5write_ComparisonSummary,
    h5read_ComparisonSummary,
    h5write_SparseComparisonSummary,
    h5read_SparseComparisonSummary,
    h5write_ConnectedComponents,
    h5read_ConnectedComponents,
    h5write_penalized_likelihood_estimate,
    h5write_posthoc_blocking

include("comparisonsummary.jl")
include("linkmatrix.jl")
include("em_functions.jl")
include("connectedcomponents.jl")
include("clustering_functions.jl")
include("weight_functions.jl")
include("maximization_functions.jl")
include("map_functions.jl")
include("sequence_map_functions.jl")
include("prior_functions.jl")
include("balancing_functions.jl")
include("move_functions.jl")
include("mcmc.jl")
include("read_write_h5.jl")
end
