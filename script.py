# Load modules
from classifier import *

# Define model
model1 = ModelDict(groups=3, users=120, features=30)

# Create toy data
data = create_toy_data(model=model1, probability_is_1=0.65)

# Add model to list of schemes
model_schemes_to_run=[]
model_schemes_to_run.append(model1)

# Run model
results = run_models(model_schemes_to_run, attempts_at_each_model=5, max_iterations=1000, data=data)

# Explore results
results.iloc[:,0:3]
results.loc[results['status'] == 'Converged']
results.loglikelihood.argmax()
results.loc[results['status'] == 'Converged'].loglikelihood.argmax()

inex_of_best_model = results.loc[results['status'] == 'Converged'].loglikelihood.argmax()

# Load best model

best_model = load_model(index_of_best_model)
