# Import modules
import numpy as np
import pdb
from dotmap import dotmap

def initialize_model(model, groups=3, users=12, features=6):
    ''' Initialize the naive bayes model '''
    model.groups = groups
    model.users = users
    model.features = features
    # probabilities that a randomly drawn user is in class 0...n
    model.group_probabilities_log = np.empty(model.groups)
    model.group_probabilities_log.fill(np.log(1.0/model.groups))

    # conditional probabilities for each feature by group
    model.conditional_probabilities_log = np.empty([model.features, model.groups], dtype=float)

    # for one class we initialize them all as log(0.5)
    # for the other class we initialize them as a random close to log(0.5)
    model.conditional_probabilities_log[:, 0].fill(log(1.0/model.groups))

    mu, sigma = 0.5, 0.0001
    pdb.set_trace()
    model.conditional_probabilities_log[:, 1:model.groups] = (np.log(np.random.normal(mu, sigma, [model.features, model.groups-1])))

    # probability_user_in_group_log rows: users, cols: log([p(class0), p(class1)])
    model.probability_user_in_group_log = np.empty([model.users, model.groups], dtype=float)

    # Initialize last_probability_data_given_parameters_log as negative infinity
    model.last_probability_data_given_parameters_log = -np.inf

    return model


def create_toy_data(model, dividing_line):
    ''' Creates toy data set with 2 classes. With nr of users = users and nr of features = features '''
    if np.mod(model.users, model.groups) != 0:
        raise ValueError('users / groups must have remainder 0')
        exit

    toy_data = np.zeros([model.users, model.features], dtype=int)

    for group in xrange(model.groups):
        xrange(group, model.users/model.groups)
        for user in xrange(group*model.users/model.groups, model.users/model.groups*(group+1)):
            n, p = 1, 1-dividing_line
            nr_r = np.random.binomial(n, p, model.features)
            toy_data[user, ] = nr_r

            n, p = 1, dividing_line
            nr_r = np.random.binomial(n, p, model.features/model.groups)
            toy_data[user, model.features/model.groups*group:model.features/model.groups*(group+1)] = nr_r

    return toy_data


def logsumexp(a):
    ''' Log sum exp of an array of numbers '''
    a = np.asarray(a)
    a_max = a.max()
    return a_max + np.log((np.exp(a-a_max)).sum())
