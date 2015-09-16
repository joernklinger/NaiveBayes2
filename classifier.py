# Import modules
import numpy as np
from numpy import log
import pdb
import cPickle as pickle

def logsumexp(a):
    ''' Log sum exp of an array of numbers '''
    a = np.asarray(a)
    a_max = a.max()
    return a_max + np.log((np.exp(a-a_max)).sum())

# Create toy data, data = toy_data to use toy data instead of real data
# I though maybe the real data was too noisy,
# but we face the same problems when using toy data


def init_parameters():
    ''' Quickly initializes core parameters '''
    global groups
    global users
    global features
    global dividing_line
    global group_probabilities_log
    global conditional_probabilities_log
    global probability_user_in_group_log
    global last_probability_data_given_parameters_log

    groups = 3
    users = 12
    features = 6
    dividing_line = 0.8
    group_probabilities_log = None
    conditional_probabilities_log = None
    probability_user_in_group_log = None
    last_probability_data_given_parameters_log = None


def create_toy_data(groups, users, features, dividing_line):
    ''' Creates toy data set with 2 classes. With nr of users = users and nr of features = features '''
    if np.mod(users, groups) != 0:
        raise ValueError('users / groups must have remainder 0')
        exit

    toy_data = np.zeros([users, features], dtype=int)

    for group in xrange(groups):
        xrange(group,users/groups)
        for user in xrange(group*users/groups,users/groups*(group+1)):
            n, p = 1, 1-dividing_line
            nr_r = np.random.binomial(n, p, features)
            toy_data[user, ] = nr_r

            n, p = 1, dividing_line
            nr_r = np.random.binomial(n, p, features/groups)
            toy_data[user, features/groups*group:features/groups*(group+1)] = nr_r

    return toy_data

def initialize_model():
    ''' Initialize the naive bayes model '''
    global group_probabilities_log
    global conditional_probabilities_log
    global probability_user_in_group_log
    global last_probability_data_given_parameters_log
    # probabilities that a randomly drawn user is in class 0...n
    group_probabilities_log = np.empty(groups)
    group_probabilities_log.fill(np.log(1.0/groups))

    # conditional probabilities for each feature by group
    conditional_probabilities_log = np.empty([features, groups], dtype=float)

    # for one class we initialize them all as log(0.5)
    # for the other class we initialize them as a random close to log(0.5)
    conditional_probabilities_log[:, 0].fill(log(1.0/groups))

    mu, sigma = 0.5, 0.0001
    conditional_probabilities_log[:, 1:groups] = (np.log(np.random.normal(mu, sigma, [features, groups-1])))

    # probability_user_in_group_log rows: users, cols: log([p(class0), p(class1)])
    probability_user_in_group_log = np.empty([users, groups], dtype=float)

    # Initialize last_probability_data_given_parameters_log as negative infinity
    last_probability_data_given_parameters_log = -np.inf

