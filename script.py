from classifier import *

# Initialize model
model = dotdict()
model = initialize_model(model, groups=3, users=120, features=30)

# Initialize model2
model2 = dotdict()
model2 = initialize_model(model2, groups=5, users=120, features=30)

# Create toy data (based on model)
data = create_toy_data(model=model, probability_is_1=0.6)

# Add models to models_to_run arrayy
model_schemes_to_run=[]
model_schemes_to_run.append(model)
model_schemes_to_run.append(model2)

# Run models
results = run_models(model_schemes_to_run, attempts_at_each_model=5, max_iterations=5, data=data)

# Get best model
best_models = get_best_models(results)
