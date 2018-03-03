# KoalaEnsembles

For building and training ensembles of `Koala` supervised learning models.

````julia
    julia> using Koala
	julia> using KoalaEnsembles
	julia> forest = EnsembleRegressor()
	EnsembleRegressor{TreeRegressor}@...192

	julia> showall(ans)
	EnsembleRegressor{TreeRegressor}@...192

	key                     | value
	------------------------|------------------------
	atom                    |TreeRegressor@...292
	bagging_fraction        |1.0
	n                       |20
	weight_regularization   |0.0
````

The hyperparameter `atom` is the model being ensembled. The default is an extreme tree:

````julia
	julia> showall(forest.atom)
	TreeRegressor@...292
	
	key                     | value
	------------------------|------------------------
	extreme                 |true
	max_features            |0
	max_height              |1000
	min_patterns_split      |2
	penalty                 |0.0
	regularization          |0.0
````

Commonly in ensemble methods, predictions are the means of the
predictions of each regressor in the ensemble. Here predictions are
*weighted* sums and the weights are optimized to minimize the RMS
training error. Since this sometimes leads to "under-regularized"
models the training error is further penalized with a term measuring
the deviation of the weights from uniformity. Set the parameter
`forest.weight_regularization=1` (the default and maximum permitted
value) and weights are completely uniform. Set
`forest.weight_regularization=0` and the training error penalty is
dropped altogether.

````julia 
    julia> const X, y = load_ames(); 
    julia> train, test = split(1:length(y), 0.8); # 80:20 split 
    julia> forest.atom.max_features = 4
    julia> forestM = Machine(forest, X, y, train)
    SupervisedMachine{EnsembleRegressor,}@...968

	julia> fit!(forestM, test)
	Computing regressor number: 20    
	SupervisedMachine{EnsembleRegressor,}@...968

    julia> forestM.report
    Dict{Symbol,Any} with 1 entry:
    :normalized_weights => [0.05, 0.05, 0.05, 0.05,…
````

To refit the weights with a new regularization penalty, but without
changing the ensemble itself, use ``fit_weights``:

````julia
    julia> forest.weight_regularization = 0.5
    julia> fit_weights!(forestM)

    20 weights optimized.                
    SupervisedMachine{EnsembleRegressor,}@...968

    julia> forestM.report
    Dict{Symbol,Any} with 1 entry:
    :normalized_weights => [1.0196, 0.640301, -0.265808,…
````

Tuning the parameter ``forest.weight_regularization`` may be done with
the help of the `weight_regularization_curve` function:

````julia
    julia> weights, errors = weight_regularization_curve(forestM, test; range = linspace(0,1,51));
    julia> opt_weight = weights[indmin(errors)]
````








