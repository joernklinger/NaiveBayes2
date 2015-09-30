from classifier import *


# user_ids = np.loadtxt('data/user_ids.txt')
feature_vector = np.loadtxt('data/feature_vector.txt')
# feature_nr_account_id = np.loadtxt('data/feature_nr-account_id.txt')

# Use name data
data = feature_vector

users = data.shape[0]
features = data.shape[1]

# Define model templates
model1 = ModelDict(groups=8, users=users, features=features, sigma=0.000005)
model2 = ModelDict(groups=8, users=users, features=features, sigma=0.000002)
model3 = ModelDict(groups=8, users=users, features=features, sigma=0.000001)
model4 = ModelDict(groups=8, users=users, features=features, sigma=0.0000005)
# model5 = ModelDict(groups=8, users=users, features=features, sigma=0.0000002)
# model6 = ModelDict(groups=8, users=users, features=features, sigma=0.0000001)
# model7 = ModelDict(groups=8, users=users, features=features, sigma=0.00000005)
# model8 = ModelDict(groups=8, users=users, features=features, sigma=0.000000001)
# model9 = ModelDict(groups=8, users=users, features=features, sigma=0.0000000005)
# model10 = ModelDict(groups=8, users=users, features=features, sigma=0.0000000001)


model_schemes_to_run=[]
model_schemes_to_run.append(model1)
model_schemes_to_run.append(model2)
model_schemes_to_run.append(model3)
model_schemes_to_run.append(model4)
model_schemes_to_run.append(model5)
model_schemes_to_run.append(model6)
model_schemes_to_run.append(model7)
model_schemes_to_run.append(model8)
model_schemes_to_run.append(model9)
model_schemes_to_run.append(model10)


# Run model
results = run_models(model_schemes_to_run, attempts_at_each_model=500, max_iterations=1000, data=data)
