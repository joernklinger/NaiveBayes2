from classifier import *

# Initialize model
model = dotdict()
model = initialize_model(model=model)

# Create toy data
data = create_toy_data(model=model, probability_is_1=0.8)

# Run model for some iterations
for i in xrange(50):
    model = iteration(model, data)
