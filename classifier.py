# See http://nbviewer.ipython.org/gist/joernklinger/e4c4936ad09bc1560546

# Import modules
import numpy as np
import pdb
from datetime import datetime
import os

# Dict with .attributes
class dotdict:
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)


def initialize_model(model, groups=3, users=12, features=6):
    ''' Initialize the naive bayes model '''
    model.groups = groups
    model.users = users
    model.features = features

    # Probabilities that a randomly drawn user is in class 0...n
    model.group_probabilities_log = np.empty(model.groups)
    model.group_probabilities_log.fill(np.log(1.0/model.groups))

    # conditional probabilities for each feature by group
    model.conditional_probabilities_log = np.empty([model.features, model.groups], dtype=float)

    # for one class we initialize them all as log(0.5)
    # for the other class we initialize them as a random close to log(0.5)
    model.conditional_probabilities_log[:, 0].fill(np.log(1.0/model.groups))

    mu, sigma = 1.0/model.groups, 0.0001
    model.conditional_probabilities_log[:, 1:model.groups] = (np.log(np.random.normal(mu, sigma, [model.features, model.groups-1])))
    model.conditional_probabilities_log = np.transpose(model.conditional_probabilities_log)

    # probability_user_in_group_log rows: users, cols: log([p(class0), p(class1)])
    model.probability_user_in_group_log = np.empty([model.users, model.groups], dtype=float)

    # Initialize last_probability_data_given_parameters_log as negative infinity
    model.probability_data_given_parameters_log = None
    model.last_probability_data_given_parameters_log = -np.inf

    # Model Status
    model.status = None

    return model


def create_toy_data(model, probability_is_1):
    ''' Creates toy data set with 2 classes. With nr of users = users and nr of features = features '''
    if np.mod(model.users, model.groups) != 0:
        raise ValueError('users / groups must have remainder 0')
        exit

    toy_data = np.zeros([model.users, model.features], dtype=int)

    for group in xrange(model.groups):
        xrange(group, model.users/model.groups)
        for user in xrange(group*model.users/model.groups, model.users/model.groups*(group+1)):
            n, p = 1, 1-probability_is_1
            nr_r = np.random.binomial(n, p, model.features)
            toy_data[user, ] = nr_r

            n, p = 1, probability_is_1
            nr_r = np.random.binomial(n, p, model.features/model.groups)
            toy_data[user, model.features/model.groups*group:model.features/model.groups*(group+1)] = nr_r
    return toy_data


def step1(model, data):
    ''' calculate probability of each user being in group 0...n '''
    if(model.probability_data_given_parameters_log != None):
        model.last_probability_data_given_parameters_log = model.probability_data_given_parameters_log

    for row in xrange(model.users):
        model.probability_user_in_group_log[row] = (data[row,]*model.conditional_probabilities_log).sum(axis=1) + model.group_probabilities_log
        # Normalize probability_user_in_class
        model.probability_user_in_group_log[row] = model.probability_user_in_group_log[row]-logsumexp(model.probability_user_in_group_log[row,])
    return model


def step2(model, data):
    ''' use probability_user_in_class_log and the actual data to update the conditional_probabilities_log '''
    feature_counts = np.ones([model.features, model.groups], dtype=float)
    for col in xrange(model.features):
        z_norm = np.zeros(model.groups)
        for row in xrange(model.users):
            if data[row, col] == 1:
                feature_counts[col] += np.exp(model.probability_user_in_group_log[row])
            z_norm += np.exp(model.probability_user_in_group_log[row])
        model.conditional_probabilities_log[:,col] = np.log(feature_counts[col]) - np.log(z_norm)
    return model


def step3(model):
    ''' Update group probailities log '''
    for group in xrange(model.groups):
        model.group_probabilities_log[group] = logsumexp(model.probability_user_in_group_log[:,group])-np.log(model.users)
    return model


def step4(model, data):
    ''' Calculate probability of the entire data given current parameters '''
    model.probability_user_given_data_log = np.zeros(model.users, dtype=float)
    for user in xrange(model.users):
        for feature in xrange(model.features):
            if data[user][feature] == 1:
                p_user_given_group_log = np.empty(model.groups)
                p_user_given_group_log = model.conditional_probabilities_log[:,feature] + p_user_given_group_log
                model.probability_user_given_data_log[user] += logsumexp(p_user_given_group_log)
    model.probability_data_given_parameters_log = np.sum(model.probability_user_given_data_log)
    return model


def get_model_stats(model):
    ''' get model stastistics '''
    keep_going = 1
    status = 'Running'
    print 'Log Likelihood: ' + str(model.probability_data_given_parameters_log)
    print 'Raw Class Probabilities: ' + str(np.exp(model.group_probabilities_log))
    if (model.probability_data_given_parameters_log < model.last_probability_data_given_parameters_log):
        print 'Warning, loglikelihood decreases. LOCAL MINIMUM.'
        keep_going = 0
        status = 'Local Minimum'
    elif (round(model.probability_data_given_parameters_log,3) == round(model.last_probability_data_given_parameters_log,3)):
        print 'Model converged.'
        keep_going = 0
        status = 'Converged'
    return keep_going, status


def iterate_model(model, data):
    # Perform step1
    model = step1(model=model, data=data)
    # Perform step2
    model = step2(model=model, data=data)
    # Perfrom step3
    model = step3(model=model)
    # Perform step4
    model = step4(model=model, data=data)
    return model


def run_models(model_schemes_to_run, attempts_at_each_model, max_iterations, data):
    results_report = []
    for model_scheme in model_schemes_to_run:
        save_dir_name = 'groups_' + str(model_scheme.groups) + '_users_' + str(model_scheme.users) + '_features_' + str(model_scheme.features) + '_time_' + str(datetime.now()).replace(' ', '_')
        os.makedirs('results/' + save_dir_name)
        for attempt in xrange(attempts_at_each_model):
            model = model_scheme
            for iteration in xrange(max_iterations):
                model = iterate_model(model, data)
                keep_going, status = get_model_stats(model)
                if (keep_going == 0):
                    model.status = status
                    model.user_in_group = np.argmax(model.probability_user_in_group_log, axis=1)
                    save_file_name = 'model_' + str(attempt+1) + '_iterations_' + str(iteration+1) + '_groups_' + str(model.groups) + '_users_' + str(model.users) + '_features_' + str(model.features) + '_model_nr_' + str(attempt) + '_time_' + str(datetime.now()).replace(' ', '_')
                    np.save('results/' + save_dir_name + '/' + save_file_name, model)
                    results_report.append([model.last_probability_data_given_parameters_log, save_dir_name, save_file_name])
                    break
    results_report = np.asarray(results_report)
    report_file_name = 'report_' + str(datetime.now()).replace(' ', '_')
    np.save('results/' + report_file_name, results_report)
    return results_report


# Load best model
def get_best_model(results_report):
    best_model_index = np.argmax(results_report[:,0])
    best_model_file_name ='results/' +  results_report[best_model_index][1] + '/' + results_report[best_model_index][2] + '.npy'
    best_model = np.load(best_model_file_name)[()]
    return best_model


def logsumexp(a):
    ''' Log sum exp of an array of numbers '''
    a = np.asarray(a)
    a_max = a.max()
    return a_max + np.log((np.exp(a-a_max)).sum())
