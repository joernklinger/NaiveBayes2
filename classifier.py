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

def create_toy_data(groups, users, features, dividing_line):
    ''' Creates toy data set with 2 classes. With nr of users = users and nr of features = features '''
    toy_data = np.empty([users, features], dtype=int)

    for user in xrange(users/2):
            n, p = 1, dividing_line
            nr_r = np.random.binomial(n, p, features/2)
            toy_data[user, :features/2] = nr_r

            n, p = 1, 1-dividing_line
            nr_r = np.random.binomial(n, p, features/2)
            toy_data[user, features/2:features] = nr_r

    for user in xrange(users/2, users):
        n, p = 1, 1-dividing_line
        nr_r = np.random.binomial(n, p, features/2)
        toy_data[user, :features/2] = nr_r

        n, p = 1, dividing_line
        nr_r = np.random.binomial(n, p, features/2)
        toy_data[user, features/2:features] = nr_r
    return toy_data

data = create_toy_data()
