import tensorflow as tf
from tensorflow import keras
from learner_agent import LearnerAgent
import time
from geneticalgorithm2 import geneticalgorithm2 as geneticalgorithm
import numpy as np
import os, shutil
import json

parameters_ranges = {
    'exp_embedding_size': [0,6],
    'no_conv_layers' : [1,7],
    'start_exp_no_filters' : [0, 8],
    'final_exp_no_filters' : [0, 8], 
    'start_kernel_size' : [1, 7],
    'final_kernel_size' : [1, 7],
    'pooling_frequency' : [0, 4]
}


test_params = [3, 0.5, 5, 6, 3, 3, 2]

split = 100
data = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = data

x_train = x_train[:split]
y_train = y_train[:split]

x_test = x_test[:int(split/10)]
y_test = y_test[:int(split/10)]

data = (x_train, y_train), (x_test, y_test)


def append_log(f, key, value, add_index = False):

    if os.path.exists(f):
        with open(f) as json_file:
            data = json.load(json_file) 
    else:
        data = {}
    
    if add_index:
        key += '_{}'.format(len(data))
    data[key] = value
        
    with open(f, 'w') as outfile:
        json.dump(data, outfile)


class Evolution:

    def __init__(self, data, parameters_ranges, loss_weight = 1, train_time_weight = 1, log_dir = 'logs', train_batches = 512):

        self.data = data
        self.train_batches = train_batches

        self.loss_weight = loss_weight
        self.train_time_weight = train_time_weight

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)

        self.log_dir = log_dir

        self.generations = 1


    def fitness(self, parameters_array):
        
        initialisation = array_to_initialisation(parameters_array)

        learner_agent = LearnerAgent(initialisation)
        no_params = learner_agent.no_params

        start_time = time.time()
        loss_history = learner_agent.train(data, batch = self.train_batches, epochs = self.train_epochs)
        train_time = (time.time() - start_time) / self.train_epochs
        
        last_loss = loss_history[-1]
    
        #append_log(os.path.join(self.log_dir, 'Generation_{}.json'.format(self.generations)), 'Agent', 'Loss: {} TrainTime: {}'.format(last_loss, train_time), add_index = True)

        return self.loss_weight * (last_loss ** 2)  + self.train_time_weight * (train_time ** 2)



    def run(self, pop_size = 20, no_iterations = 50, start_train_epochs = 1, end_train_epochs = 30):

        self.train_epochs = start_train_epochs
        self.train_epochs_step = int((end_train_epochs - start_train_epochs) / no_iterations)

        algorithm_param = {
            'max_num_iteration': no_iterations,
            'population_size': pop_size,
            'mutation_probability':0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type':'uniform',
            'max_iteration_without_improv': None
        }

        varbound = np.array(initialisation_to_array(parameters_ranges))
    
        model = geneticalgorithm(
            function = self.fitness,
            dimension = 7,
            variable_type = 'int',
            variable_boundaries = varbound, 
            algorithm_parameters = algorithm_param,
            function_timeout = 1000
        )

        model.run(
            callbacks = [self.callback]
        )
        

    def callback(self, generation_number, report_list, last_population_as_2D_array, last_population_scores_as_1D_array):

        
        print('Completed generation {}'.format(generation_number))
        self.generations += 1
        self.train_epochs += self.train_epochs_step

        print(last_population_scores_as_1D_array.shape)
        append_log(os.path.join(self.log_dir, 'Evolution_Overview.json'), 'Generation_{}'.format(generation_number), 'AverageLoss: {}'.format(np.average(last_population_scores_as_1D_array)))
        append_log(os.path.join(self.log_dir, 'Best_Individual.json'), 'Generation_{}'.format(generation_number), str(last_population_as_2D_array[np.argmin(last_population_scores_as_1D_array)]))
        

def initialisation_to_array(initailisation_dict):
    
    array = []
    array.append(initailisation_dict.get('exp_embedding_size'))
    array.append(initailisation_dict.get('no_conv_layers'))
    array.append(initailisation_dict.get('start_exp_no_filters'))
    array.append(initailisation_dict.get('final_exp_no_filters'))
    array.append(initailisation_dict.get('start_kernel_size'))
    array.append(initailisation_dict.get('final_kernel_size'))
    array.append(initailisation_dict.get('pooling_frequency'))
    return array


def array_to_initialisation(array):
    initialisation = {
        'exp_embedding_size': array[0],
        'no_conv_layers' : array[1],
        'start_exp_no_filters' : array[2],
        'final_exp_no_filters' : array[3], 
        'start_kernel_size' : array[4],
        'final_kernel_size' : array[5],
        'pooling_frequency' : array[6]
    }
    return initialisation



if __name__ == '__main__': 
    evolution = Evolution(
        data, 
        parameters_ranges, 
        loss_weight = 1, 
        train_time_weight = 1, 
        log_dir = 'logs', 
        train_batches = 512
    )
    
    evolution.run(
        pop_size = 40,
        no_iterations = 50,
        start_train_epochs = 3, 
        end_train_epochs = 20 
    )
    