# See http://nbviewer.ipython.org/gist/joernklinger/e4c4936ad09bc1560546

# Import modules
import numpy as np
import pdb
from datetime import datetime
import os
import copy


class ModelDict(dict):
    ''' Model dictionary class '''
    def __init__(self, groups=3, users=12, features=6):
        self['groups'] = groups
        self['users'] = users
        self['features'] = features

    def initialize(self):
        # Probabilities that a randomly drawn user is in class 0...n
        self['group_probabilities_log'] = np.empty(self['groups'])
        self['group_probabilities_log'].fill(np.log(1.0/self['groups']))

        # Probabilities that a randomly drawn user is in class 0...n
        self['group_probabilities_log'] = np.empty(self['groups'])
        self['group_probabilities_log'].fill(np.log(1.0/self['groups']))

        # Probabilities that a randomly drawn user is in class 0...n
        self['group_probabilities_log'] = np.empty(self['groups'])
        self['group_probabilities_log'].fill(np.log(1.0/self['groups']))

        # conditional probabilities for each feature by group
        self['conditional_probabilities_log'] = np.empty([self['features'], self['groups']], dtype=float)

        # for one class we initialize them all as log(1/groups)
        # for the other class we initialize them as a random close to log(1/groups)
        self['conditional_probabilities_log'][:, 0].fill(np.log(1.0/self['groups']))
        mu, sigma = 1.0/self['groups'], 0.01
        self['conditional_probabilities_log'][:, 1:self['groups']] = (np.log(np.random.normal(mu, sigma, [self['features'], self['groups']-1])))
        self['conditional_probabilities_log'] = np.transpose(self['conditional_probabilities_log'])

        # probability_user_in_group_log rows: users, cols: log([p(class0), p(class1)])
        self['probability_user_in_group_log'] = np.empty([self['users'], self['groups']], dtype=float)

        # Initialize last_probability_data_given_parameters_log as negative infinity
        self['probability_data_given_parameters_log'] = None
        self['last_probability_data_given_parameters_log'] = -np.inf

        # Model status
        self['status'] = None

        # Other attributes
        self['user_in_group'] = None
        self['iterations'] = 0

    def info(self):
        print 'groups: ' + str(self['groups']) + '\n' + 'users: ' + str(self['users']) + ' \n' + 'features: ' + str(self['features']) + '\n' + 'iteration: ' +  str(self['iterations']) + '\n' + 'status: ' + str(self['status'])


def create_toy_data(model, probability_is_1):
    ''' Creates toy data set with 2 classes. With nr of users = users and nr of features = features '''
    if np.mod(model['users'], model['groups']) != 0:
        raise ValueError('users / groups must have remainder 0')
        exit

    toy_data = np.zeros([model['users'], model['features']], dtype=int)

    for group in xrange(model['groups']):
        for user in xrange(group*model['users']/model['groups'], model['users']/model['groups']*(group+1)):
            n, p = 1, 1-probability_is_1
            nr_r = np.random.binomial(n, p, model['features'])
            toy_data[user, ] = nr_r

            n, p = 1, probability_is_1
            nr_r = np.random.binomial(n, p, model['features']/model['groups'])
            toy_data[user, model['features']/model['groups']*group:model['features']/model['groups']*(group+1)] = nr_r
    return toy_data


def step1(model, data):
    ''' calculate probability of each user being in group 0...n '''
    if(model['probability_data_given_parameters_log'] != None):
        model['last_probability_data_given_parameters_log'] = model['probability_data_given_parameters_log']

    if(model['status'] == None):
        model['status'] = 'Running'

    for user in xrange(model['users']):5
        model['probability_user_in_group_log'][user] = (data[user,]*model['conditional_probabilities_log']).sum(axis=1) + model['group_probabilities_log']
        # Normalize probability_user_in_class
        model['probability_user_in_group_log'][user] = model['probability_user_in_group_log'][user]-logsumexp(model['probability_user_in_group_log'][user,])
        model['user_in_group'] = np.argmax(model['probability_user_in_group_log'], axis=1)
    return model


def step2(model, data):
    ''' use probability_user_in_class_log and the actual data to update the conditional_probabilities_log '''
    feature_counts = np.ones([model['features'], model['groups']], dtype=float)
    for col in xrange(model['features']):
        z_norm = np.zeros(model['groups'])
        for row in xrange(model['users']):
            if data[row, col] == 1:
                feature_counts[col] += np.exp(model['probability_user_in_group_log'][row])
            z_norm += np.exp(model['probability_user_in_group_log'][row])
        model['conditional_probabilities_log'][:,col] = np.log(feature_counts[col]) - np.log(z_norm)
    return model


def step3(model):
    ''' Update group probailities log '''
    for group in xrange(model['groups']):
        model['group_probabilities_log'][group] = logsumexp(model['probability_user_in_group_log'][:,group])-np.log(model['users'])
    return model


def step4(model, data):
    ''' Calculate probability of the entire data given current parameters '''
    model['probability_user_given_data_log'] = np.zeros(model['users'], dtype=float)
    for user in xrange(model['users']):
        for feature in xrange(model['features']):
            if data[user][feature] == 1:
                p_user_given_group_log = np.zeros(model['groups'])
                p_user_given_group_log = model['conditional_probabilities_log'][:,feature] + p_user_given_group_log
                model['probability_user_given_data_log'][user] += logsumexp(p_user_given_group_log)
    model['probability_data_given_parameters_log'] = np.sum(model['probability_user_given_data_log'])
    return model


def get_model_stats(model):
    ''' get model stastistics '''
    keep_going = 1
    status = 'Running'
    print 'Log Likelihood: ' + str(model['probability_data_given_parameters_log'])
    print 'Raw Class Probabilities: ' + str(np.exp(model['group_probabilities_log']))
    print '\n'
    if (round(model['probability_data_given_parameters_log'],10) == round(model['last_probability_data_given_parameters_log'],10)):
        keep_going = 0
        status = 'Converged'
        print 'Converged.'
        print 'Log Likelihood: ' + str(model['probability_data_given_parameters_log'])
        print 'Raw Class Probabilities: ' + str(np.exp(model['group_probabilities_log']))
        print '\n'
    elif (model['probability_data_given_parameters_log'] < model['last_probability_data_given_parameters_log']):
        keep_going = 0
        status = 'Local Minimum'
        print 'Local Minimum'
        print 'Current LL: ' + str(model['probability_data_given_parameters_log'])
        print 'Last Log Likelihood: ' + str(model['last_probability_data_given_parameters_log'])
        print 'Difference: ' + str(abs(model['probability_data_given_parameters_log']-model['last_probability_data_given_parameters_log']))
        print '\n'
    return keep_going, status


def iterate_model(model, data):
    ''' Performs one iteration of the model'''
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
    ''' Runs models, saves models and result summaries '''
    results_temp = []
    for model_scheme in model_schemes_to_run:
        save_dir_name = 'groups_' + str(model_scheme['groups']) + '_users_' + str(model_scheme['users']) + '_features_' + str(model_scheme['features']) + '_time_' + str(datetime.now()).replace(' ', '_')
        os.makedirs('results/' + save_dir_name)
        print 'Model scheme with group: ' + str(model_scheme['groups']) + '\n'
        for attempt in xrange(attempts_at_each_model):
            model = copy.deepcopy(model_scheme)
            model.initialize()
            print 'Attempt: ' + str(attempt) + ' start ' + '\n'
            for iteration in xrange(max_iterations):
                print 'Attempt: ' + str(attempt) + ' iteration: ' + str(iteration) + '\n'
                backup_model = copy.deepcopy(model)
                model = iterate_model(model, data)
                keep_going, status = get_model_stats(model)
                if (keep_going == 0):
                    model = backup_model
                    model['status'] = status
                    model['user_in_group'] = np.argmax(model['probability_user_in_group_log'], axis=1)
                    model['iteartions'] = iteration+1
                    save_file_name = 'model_' + str(attempt+1) + '_iterations_' + str(iteration+1) + '_groups_' + str(model['groups']) + '_users_' + str(model['users']) + '_features_' + str(model['features']) + '_model_nr_' + str(attempt) + '_time_' + str(datetime.now()).replace(' ', '_')
                    np.save('results/' + save_dir_name + '/' + save_file_name, model)
                    results_temp.append([model['groups'], model['last_probability_data_given_parameters_log'], save_dir_name, save_file_name, model['status']])
                    break
    results_temp = np.asarray(results_temp)
    results = dict()
    results['groups'] = results_temp[:,0].astype(np.float)
    results['loglikelihood'] = results_temp[:,1].astype(np.float)
    results['save_dir_name'] = results_temp[:,2]
    results['save_file_name'] = results_temp[:,3]
    results['status'] = results_temp[:,4]

    results_file_name = 'results_' + str(datetime.now()).replace(' ', '_')
    np.save('results/' + results_file_name, results)
    return results


def get_best_models(results):
    ''' Get the best model for each number of groups '''
    best_models = []
    for groups_nr in list(set(results['groups'])):
        indices = np.where(results['groups'] == groups_nr)[0]
        best_model_index = indices[np.argmax(results['loglikelihood'][indices])]
        best_model_file_name ='results/' +  results['save_dir_name'][best_model_index] + '/' + results['save_file_name'][best_model_index] + '.npy'
        best_model_for_groups = np.load(best_model_file_name)[()]
        best_models.append(best_model_for_groups)
    print 'Best models for number of groups: \n'
    print range(len(best_models))
    print[ int(x) for x in list(set(results['groups'])) ]
    return best_models


def logsumexp(a):
    ''' Log sum exp of an array of numbers '''
    a = np.asarray(a)
    a_max = a.max()
    return a_max + np.log((np.exp(a-a_max)).sum())
