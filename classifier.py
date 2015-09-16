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









    return toy_data

data = create_toy_data()
