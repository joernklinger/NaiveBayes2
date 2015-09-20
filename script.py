from classifier import *

# Initialize model
model = dotdict()
model = initialize_model(model, groups=3, users=120, features=30)

# Initialize model2
model2 = dotdict()
model2 = initialize_model(model2, groups=5, users=120, features=30)

# Create toy data (based on model)
data = create_toy_data(model=model, probability_is_1=0.66)

# Add models to models_to_run arrayy
models_to_run=[]
models_to_run.append(model)
models_to_run.append(model2)

# Run models
results_report = run_models(models_to_run, attempts_at_each_model=5, max_iterations=5, data=data)

# Get best model
best_model = ger_best_model(results_report)
