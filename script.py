# See http://nbviewer.ipython.org/gist/joernklinger/e4c4936ad09bc1560546

from classifier import *

# Initialize model
model = dotdict()
model = initialize_model(model=model)

# Create toy data
data = create_toy_data(model=model, dividing_line=0.8)

# Perform step1
model = step1(model, data)


