module BayesianRecordLinkage
using AssignmentSolver, HDF5

using Distributions: Dirichlet
using DataStructures: Queue, enqueue!, dequeue!, DefaultDict
using StatsBase
using StatsFuns: logit, logistic, log1pexp, loghalf, softmax, softmax!, logsumexp, logaddexp
using SparseArrays
using SpecialFunctions: logfactorial, logbeta
using Dates
import Base: ==

export ComparisonSummary,
    SparseComparisonSummary,
    mapping_variables,
    comparison_variables,
    count_variables,
    merge_comparisonsummary,
    get_obsidxcounts,
    get_obsidxobs
export ConnectedComponents,
    get_component,
    get_ranges,
    get_dimensions,
    get_mids,
    count_pairs,
    maxcomponent_pairs,
    maxdimension,
    count_singleton,
    summarize_components
export LinkMatrix,
    has_link,
    add_link,
    add_link!,
    remove_link!,
    remove_link,
    rowswitch_link!,
    rowswitch_link,
    switchcol_link!,
    switchcol_link,
    doubleswitch!_link,
    doubleswitch_link,
    tuple2links,
    links2tuple,
    row2col_removed,
    row2col_added,
    row2col_difference
export ParameterChain,
    counts2indicies,
    get_linkcounts,
    get_groupidcounts_row,
    get_groupidcounts_column,
    get_groupidcounts_pair,
    get_linkstagecounts,
    get_steplinks,
    get_segmentlinks,
    update_Ctrace_vars!
export PosthocBlocks,
    label2dict
export E_step,
    M_step,
    estimate_EM
export bipartite_cluster,
    iterative_bipartite_cluster,
    iterative_bipartite_cluster2
export minimum_margin,
    counts_matches,
    weights_vector,
    shrink_weights,
    shrink_weights!,
    penalized_weights_vector,
    weights_matrix,
    maximum_weights_vector,
    maximum_weights_matrix,
    penalized_weights_matrix,
    indicator_weights_matrix,
    compute_costs,
    compute_costs_shrunk,
    bayesrule_posterior,
    threshold_sensitivity
export prior_mode,
    max_MU,
    max_C_hungarian,
    max_C_auction,
    max_C_auction!,
    max_C_cluster_setup,
    max_C_cluster_hungarian,
    max_C_cluster_auction,
    max_C_cluster_auction!,
    max_C,
    max_C_cluster,
    max_C_cluster2,
    max_C_auction,
    max_C_auction_cluster
export penalized_likelihood_hungarian,
    penalized_likelihood_cluster_hungarian,
    penalized_likelihood_auction,
    penalized_likelihood_cluster_auction,
    penalized_likelihood_auction_update,
    penalized_likelihood_cluster_auction_update,
    map_solver,
    map_solver_cluster,
    map_solver_auction,
    map_solver_auction_cluster
export incr_penalty,
    penalized_likelihood_search_hungarian,
    penalized_likelihood_search_auction
export exppenalty_logprior,
    betabipartite_logprior,
    exppenalty_logratio,
    betabipartite_logratio,
    exppenalty_logratiopn,
    betabipartite_logratiopn
export lsqrt,
    lsqrt_logx,
    lbarker,
    barker_logx,
    barker,
    lmin1,
    min1_logx,
    min1,
    lmax1,
    max1_logx,
    max1
export idx2pair,
    pair2idx,
    sample_proposal_full,
    sample_proposal_sparse,
    get_loglik,
    loglik_add,
    loglik_remove,
    loglik_rowswitch,
    loglik_colswitch,
    loglik_doubleswitch,
    get_counts,
    counts_add,
    counts_remove,
    counts_rowswitch,
    counts_colswitch,
    counts_doubleswitch,
    logpCRatios_add,
    logpCRatios_remove,
    randomwalk1_move!,
    randomwalk1_inverse,
    randomwalk1_loglikpCratio,
    randomwalk1_countsdelta,
    randomwalk1_log_movecount,
    randomwalk1_update!,
    randomwalk1_log_move_weights,
    randomwalk1_log_move_weights_sparse,
    randomwalk1_locally_balanced_update!,
    randomwalk1_globally_balanced_update!,
    randomwalk1_locally_balanced_sqrt_update!,
    randomwalk1_locally_balanced_barker_update!,
    randomwalk2_move!,
    randomwalk2_inverse,
    randomwalk2_loglikpCratio,
    randomwalk2_countsdelta,
    randomwalk2_update!,
    singleton_gibbs!,
    dirichlet_draw,
    gibbs_MU_draw
export dropoutside!, dropoutside, mh_gibbs_count, mh_gibbs_trace
export h5write_ComparisonSummary,
    h5read_ComparisonSummary,
    h5write_SparseComparisonSummary,
    h5read_SparseComparisonSummary,
    h5write_ConnectedComponents,
    h5read_ConnectedComponents,
    h5write_ParameterChain,
    h5read_ParameterChain,
    h5write_PosthocBlocks,
    h5read_PosthocBlocks,
    h5write_penalized_likelihood_estimate,
    h5write_posthoc_blocking

include("comparisonsummary.jl")
include("connectedcomponents.jl")
include("linkmatrix.jl")
include("parameterchain.jl")
include("posthocblocks.jl")
include("em_functions.jl")
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
