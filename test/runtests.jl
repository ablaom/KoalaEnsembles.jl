using Koala
using KoalaEnsembles
using Base.Test

const X, y = load_ames();
const train, test = splitrows(1:length(y), 0.8);
const forest = EnsembleRegressor()
showall(forest)
tree = forest.atom
tree.max_features = 4
forest.n = 50
forestM = Machine(forest, X, y, train)
showall(forestM)
fit!(forestM, train)
forest.weight_regularization = 0.5
fit_weights!(forestM)
err(forestM, test)
fit!(forestM, train, add=true)
err(forestM, test)
u,v = weight_regularization_curve(forestM, test, raw=false, range=linspace(0,1,21))
forest.weight_regularization = u[indmin(v)]
fit_weights!(forestM)
err(forestM, test)
forest.n = 200
fit!(forestM, train)
err(forestM, test)
