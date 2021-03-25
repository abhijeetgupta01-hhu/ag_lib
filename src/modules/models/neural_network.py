# coding:utf-8

__author__ = "Abhijeet Gupta (abhijeet.gupta@gmail.com)"
__copyright__ = "Copyright (C) 2021 Abhijeet Gupta"
__license__ = "GPL 3.0"

###############################################################################################################
#
#  IMPORT STATEMENTS
#
###############################################################################################################


# >>>>>>>>> Native Imports <<<<<<<<<<
import os

# >>>>>>>>> Package Imports <<<<<<<<<<

import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import regularizers, initializers
from keras import optimizers
from keras import backend as K

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.models.base_model import BaseModel

###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class NeuralNetwork(BaseModel):
    """
    Class to implement the Deep Neural Networks

    The class instance is directly invoked by the Model Contorller (model.py)

    """
###############################################################################################################
#
#  1. Basic initializations
#
###############################################################################################################

    def __init__(self):
        """
        Method for basic model initializations
        """
        self.__parameters = {'input_dim':0,
                             'output_dim':0,
                             'learning_rate':0.001,
                             'hl_sizes':[(100)],
                             'activation_funct':[("tanh")],
                             'output_activation_funct':"softmax",
                             'cost_funct':'categorical_crossentropy',
                             'l2reg':0.00001,
                             'l1reg':0.0,
                             'optimizer':"adam",
                             'max_epochs':1000,
                             'cost_delta':0.000001,
                             'dropout':0.5,
              }

        super(NeuralNetwork, self).__init__(parameters=self.__parameters)
        return

###############################################################################################################
#
#  2. Model Architecture
#
###############################################################################################################

    def build_model(self, model_file=None, model_dir=None, parameters=None):
        """
        CORE FUNCTION:
        The function takes as input the parameters required to build a (deep) model
        on the fly
        - The input layers, hidden layers and output layers are built dynamically
        - The model can be of any depth: as provided by the user
        - All related parameters are supplied at the time of program initialization
            in the dictionary self.__model_params[experiment]['parameters'][model_name]

        :model_file (str) : The name by which the trained data model will be stored
        :model_dir (str) : The path on which the model_file will be stored
        :parameters (dict(str:(any))) : The list of (hyper)parameters which have been passed
                                        to the model

        :returns (bool, model_file, model) :
                - bool : True, if the model was trained successfully; else False
                - model_file : The path & name of the file by which the model is stored
                - model (model_obj) : This is the trained model instance which has the
                                    weights and layers which can be directly loaded
                                    and used again

        """
        is_executed = True
        model_file = os.path.join(model_dir, model_file)


        ################################################################################
        # Building A SEQUENTIAL keras model
        model = Sequential()  # also creates the first input layer

        # hidden_layer_info : (list(tuple(hidden_units, activation_funct, dropout_value)))
        is_executed, network_layer_data = self.get_network_layer_data(parameters=parameters)


        if is_executed:

            for layer_no, (layer_units, activation_funct, dropout_val) in enumerate(network_layer_data, 1):
                # First layer
                if layer_no == 1:
                    model.add(
                                     Dense(units=layer_units,
                                           input_dim=parameters['input_dim'],
                                           activation=activation_funct,
                                           use_bias=True,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer='zeros', kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           kernel_constraint=None,
                                           bias_constraint=None)
                                     )
                else:
                    # Output layer
                    if dropout_val == -1.0:
                        model.add(
                                         Dense(units=layer_units,
                                               activation=activation_funct,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros', kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                               bias_regularizer=None,
                                               activity_regularizer=None,
                                               kernel_constraint=None,
                                               bias_constraint=None)
                                         )
                    # In-between layers
                    else:
                        model.add(Dropout(dropout_val))
                        model.add(
                                         Dense(units=layer_units,
                                               activation=activation_funct,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros', kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                               bias_regularizer=None,
                                               activity_regularizer=None,
                                               kernel_constraint=None,
                                               bias_constraint=None)
                                         )

        if parameters['optimizer'] == "adadelta":
            parameters['optimizer'] = optimizers.Adadelta(lr=1.0, rho=0.95,
                                                          epsilon=1e-08, decay=0.0)
        elif parameters['optimizer'] == "sgd":
            parameters['optimizer'] = optimizers.SGD(lr=parameters['learning_rate'],
                                                     decay=1e-6, momentum=0.9, nesterov=True)

        # COMPILE THE MODELS
        model.compile(loss=parameters['cost_funct'],
                      optimizer=parameters['optimizer'],
                      metrics=['accuracy'])

        # ADD CALLBACKS FOR EARLY STOPPING, ETC
        model.callback_list = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                    min_delta=parameters['cost_delta'],
                                                                    patience=3,
                                                                    verbose=1,
                                                                    mode='auto'
                                                                    ),
                                      keras.callbacks.EarlyStopping(monitor='acc',
                                                                    min_delta=0,
                                                                    patience=3,
                                                                    verbose=1,
                                                                    mode='auto'
                                                                    ),
                                      keras.callbacks.History(),
                                      keras.callbacks.ModelCheckpoint(model_file,
                                                                      monitor='acc',
                                                                      verbose=1,
                                                                      save_best_only=True,
                                                                      mode='max',
                                                                      period=1)
                                      ]
        #################################################################################

        return is_executed, model_file, model
