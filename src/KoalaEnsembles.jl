#__precompile__()
module KoalaEnsembles

# new:
export EnsembleRegressor, fit_weights!, weight_regularization_curve

# for use in this module:
import Koala: Regressor, BaseType, SupervisedMachine, type_parameters, params
import Koala: err
import KoalaTrees: TreeRegressor
import StatsBase
import DataFrames: AbstractDataFrame

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, transform, inverse_transform

# development only:
# import ADBUtilities: @dbg, @colon

# constants:
const EPS = eps(Float64)


## Weighted ensembles of predictors

mutable struct WeightedEnsemble{P, Atom <: Regressor{P}} <: BaseType
    model::Atom
    ensemble::Vector{P}
    weights::Vector{Float64}
end

WeightedEnsemble(model::P, ensemble::Vector{P}, weights) where P =
    WeightedEnsemble{P, Atom}(model, ensemble, weights)

function Base.show(stream::IO,
                   object::WeightedEnsemble{P, Atom}) where {P, Atom<:Regressor{P}}
    abbreviated(n) = "..."*string(n)[end-2:end]
    type_string = string("{", Atom, "}")
    print(stream, string(typeof(object).name.name,
                         type_string,
                         "@", abbreviated(hash(object))))
end

function predict(wens::WeightedEnsemble, X)
    ensemble = wens.ensemble
    weights = wens.weights

    !isempty(weights) || error("Empty ensemble cannot make predictions.")
    
    n_regressors = length(ensemble)
    prediction = weights[1]*predict(wens.model, ensemble[1], X, false, false)
    i =2
    while i <= n_regressors
        prediction += weights[i]*predict(wens.model, ensemble[i], X, false, false)
        i += 1
    end
    return prediction
end


## The `EnsembleRegressor` type

mutable struct EnsembleRegressor{P, Atom <: Regressor{P}} <: Regressor{WeightedEnsemble{P, Atom}} 

    atom::Atom
    weight_regularization::Float64
    bagging_fraction::Float64
    n::Int # number of predictors created (added) in call to fit (with add=true)

    function EnsembleRegressor{P, Atom}(atom::Atom, weight_regularization,
                     bagging_fraction, n::Int) where {P, Atom <: Regressor{P}}
        if bagging_fraction > 1 ||  bagging_fraction <= 0
            error("`bagging_fraction` should be in the range (0,1].")
        end
        if weight_regularization > 1 || weight_regularization < 0
            error("`weight_regularization` should be in the range [0,1].")
        end
        return new(atom, weight_regularization, bagging_fraction, n)
    end
end

# constructor to infer type automatically:
EnsembleRegressor(atom::Atom, weight_regularization,
                  bagging_fraction, n::Int) where {P, Atom<:Regressor{P}} =
                      EnsembleRegressor{P, Atom}(atom, weight_regularization, bagging_fraction, n)

# lazy keyword constructor:
EnsembleRegressor(;atom=TreeRegressor(),
                  weight_regularization=1,
                  bagging_fraction=0.8,
                  n::Int=100) =
                      EnsembleRegressor(atom, weight_regularization, bagging_fraction, n)

function Base.show(stream::IO, object::EnsembleRegressor)
    abbreviated(n) = "..."*string(n)[end-2:end]
    type_params = type_parameters(object)
    if isempty(type_params)
        type_string = ""
    else
        type_string = string("{", type_params[2], "}")
    end
    print(stream, string(typeof(object).name.name,
                         type_string,
                         "@", abbreviated(hash(object))))
end

# transformers are inherited from atom:
default_transformer_X(model::EnsembleRegressor) = default_transformer_X(model.atom)
default_transformer_y(model::EnsembleRegressor) = default_transformer_y(model.atom)

function setup(model::EnsembleRegressor{P, Atom},
               Xt, yt, scheme_X, parallel, verbosity) where {P, Atom<:Regressor{P}}
    ensemble = Array{P}(0)
    return Xt, yt, scheme_X, ensemble
end

# Note: Whenceforth I use "X" and "y" instead of "Xt" and "yt"

function fit(model::EnsembleRegressor{P, Atom}, cache, add, parallel,
             verbosity; optimize_weights_only=false) where {P,
             Atom<:Regressor{P}}

    X, y, scheme_X, ensemble_so_far = cache
    n = model.n
    nbr_so_far = length(ensemble_so_far)
    n_patterns = length(y)
    n_train = round(Int, floor(model.bagging_fraction*n_patterns))

    # core ensemble building function:
    function get_ensemble(n, verbosity::Int)

        # initialize random number generator:
        srand((round(Int,time()*1000000)))

        ensemble = Vector{P}(n)
    
        for i in 1:n
            verbosity < 1 || print("\rComputing regressor number: $(i + nbr_so_far)    ")
            train_rows = StatsBase.sample(1:n_patterns, n_train, replace=false)
            atom_cache = setup(model.atom, X, y, scheme_X, false, verbosity - 1)
            atom_predictor, atom_report, atom_cache =
                fit(model.atom, atom_cache, false, false, verbosity - 1)
            ensemble[i] = atom_predictor
        end
        verbosity < 1 || println()

        return ensemble
        
    end

    
    ## Build or supplement ensemble if required
    
    if optimize_weights_only # build not required
        ensemble = ensemble_so_far
    else # build required
        if !parallel || nworkers() == 1 # build in serial
            ensemble = get_ensemble(n, verbosity - 1)
        else # build in parallel
            if verbosity >= 1
                println("Ensemble-building in parallel on $(nworkers()) processors.")
            end
            chunk_size = div(n, nworkers())
            left_over = mod(n, nworkers())
            ensemble =  @parallel (vcat) for i = 1:nworkers()
                if i != nworkers()
                    get_ensemble(chunk_size, 0) # 0 means silent
                else
                    get_ensemble(chunk_size + left_over, 0) 
                end
            end
        end
        # include an existing ensemble if required:
        if add
            ensemble = vcat(ensemble_so_far, ensemble)
        end 
    end

    
    ## Optimize weights

    n = length(ensemble)
    
    if model.weight_regularization == 1
        weights = ones(n)/n
        verbosity < 1 || println("Weighting regressors uniformly.")
    else
        verbosity < 1 || print("\nOptimizing weights...")
        Y = Array{Float64}(n, n_patterns)
        for k in 1:n
            Y[k,:] = predict(model.atom, ensemble[k], X, false, false)
        end

        # If I rescale all predictions by the same amount it makes no
        # difference to the values of the optimal weights:
        ybar = mean(abs.(y))
        Y = Y/ybar
        
        A = Y*Y'
        b = Y*(y/ybar)

        scale = abs(det(A))^(1/n)

        if scale < EPS

            warn("Weight optimization problem ill-conditioned. " *
                 "Using uniform weights.")
            weights = ones(n)/n

        else

            # need regularization, `gamma`, between 0 and infinity:
            if model.weight_regularization == 0 
                gamma = 0
            else
                gamma = exp(atanh(2*model.weight_regularization - 1))
            end
            
            # add regularization and augment linear system for constraint
            # (weights sum to one)
            AA = hcat(A + scale*gamma*eye(n), ones(n))
            AA = vcat(AA, vcat(ones(n), [0.0])')
            bb = b + scale*gamma*ones(n)/n
            bb = vcat(bb, [1.0])
            
            weights = (AA \ bb)[1:n] # drop Lagrange multiplier
            verbosity < 1 || println("\r$n weights optimized.\n")

        end

    end
                
    predictor = WeightedEnsemble(model.atom, ensemble, weights)
    report = Dict{Symbol, Any}()
    report[:normalized_weights] = weights*length(weights)

    cache = (X, y, scheme_X, ensemble)
        
    return predictor, report, cache

end

predict(model::EnsembleRegressor, predictor, Xt, parallel, verbosity) =
    predict(predictor, Xt)

function fit_weights!(mach::SupervisedMachine{WeightedEnsemble{P, Atom},
                                              EnsembleRegressor{P, Atom}};
              verbosity=1, parallel=true) where {P, Atom <: Regressor{P}}

    mach.n_iter != 0 || error("Cannot fit weights to empty ensemble.")

    mach.predictor, report, mach.cache =
        fit(mach.model, mach.cache, false, parallel, verbosity;
            optimize_weights_only=true)
    merge!(mach.report, report)

    return mach
end

function weight_regularization_curve(mach::SupervisedMachine{WeightedEnsemble{P, Atom},
                                                           EnsembleRegressor{P, Atom}},
                                   test_rows;
                                   verbosity=1, parallel=true, range=linspace(0,1,101),
                                   raw=true) where {P, Atom <: Regressor{P}}

    mach.n_iter > 0 || error("No regressors in the ensemble. Run `fit!` first.")
    !raw || warn("Reporting errors for *transformed* target. Use `raw=false` "*
                 " to report true errors.")

    if parallel && nworkers() > 1
        if verbosity >= 1
            println("Optimizing weights in parallel on $(nworkers()) processors.")
        end
        errors = pmap(range) do w
            verbosity < 2 || print("\rweight_regularization=$w       ")
            mach.model.weight_regularization = w
            fit_weights!(mach; parallel=false, verbosity=verbosity - 1)
            err(mach, test_rows, raw=raw)
        end
    else
        errors = Float64[]
        for w in range
            verbosity < 1 || print("\rweight_regularization=$w       ")
            mach.model.weight_regularization = w
            fit_weights!(mach; parallel= parallel, verbosity=verbosity - 1)
            push!(errors, err(mach, test_rows, raw=raw))
        end
        verbosity < 1 || println()
        
        mach.report[:weight_regularization_curve] = (range, errors)
    end
    
    return range, errors
end

end # module
