using Koala
using KoalaEnsembles
using Base.Test

const X, y = load_ames();
const train, test = splitrows(1:length(y), 0.8);
const model = EnsembleRegressor()
model.atom.max_features = 4
mach = SupervisedMachine(model, X, y, train)
fit!(mach, train)
model.weight_regularization = 0.5
fit_weights!(mach)
err(mach, test)
fit!(mach, train, add=true)
err(mach, test)
u,v = weight_regularization_curve(mach, test, raw=false, range=linspace(0,1,21))
model.weight_regularization = u[indmax(v)]
fit_weights!(mach)
err(mach, test)
