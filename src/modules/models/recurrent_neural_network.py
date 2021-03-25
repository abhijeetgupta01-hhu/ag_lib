#! /usr/bin/python3.7
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
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, TimeDistributed, SimpleRNN, Bidirectional, Masking
from keras import regularizers, initializers
from keras import optimizers
from keras import backend as K

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO
from modules.models.base_model import BaseModel


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class RecurrentNeuralNetwork(BaseModel):
    """
    Class to implement the Deep Recurrent Neural Networks

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
        self.io = IO()
        self.__parameters = {'input_dim':0,
                             'output_dim':0,
                             'sub_type':"lstm",
                             'learning_rate':0.1,
                             'hl_sizes':[(128)],
                             'activation_funct':[("tanh")],
                             'output_activation_funct':"softmax",
                             'recurrent_activation_funct':"sigmoid",
                             'cost_funct':"kullback_leibler_divergence",
                             'kernel_initializer':"glorot_uniform",
                             'recurrent_initializer':"glorot_uniform",
                             'l2reg':0.0,
                             'l1reg':0.0,
                             'optimizer':"adam",
                             'max_epochs':1000,
                             'cost_delta':1e-06,
                             'dropout':0.5,
                             'stateful':False,
                             'return_sequences':False,
                             'masking':False,
                             'mask_value':-1.0
                             }

        super(RecurrentNeuralNetwork, self).__init__(parameters=self.__parameters)
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

        sub_type = parameters['sub_type']

        # try:
        ################################################################################
        # Building A SEQUENTIAL keras model
        model = Sequential()
        if parameters['masking']:
            model.add(Masking(mask_value=self.__parameters['mask_value'],
                              input_shape=(self.train_input_shape[-2],
                                           self.train_input_shape[-1]
                                           )
                              )
                      )
        # hidden_layer_info : (list(tuple(hidden_units, activation_funct, dropout_value)))
        is_executed, network_layer_data = self.get_network_layer_data(parameters=parameters)


        if is_executed:

            for layer_no, (layer_units, activation_funct, dropout_val) in enumerate(network_layer_data, 1):

                # Last layer - which has dropout initialized to -1
                if dropout_val == -1:
                    if parameters['return_sequences']:
                        model.add(TimeDistributed(
                                     Dense(units=layer_units,
                                           activation=activation_funct,
                                           use_bias=True,
                                           kernel_initializer=parameters['kernel_initializer'],
                                           bias_initializer='zeros', kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                           )
                                     )
                              )
                    else:
                        model.add(Dense(units=layer_units,
                                           activation=activation_funct,
                                           use_bias=True,
                                           kernel_initializer=parameters['kernel_initializer'],
                                           bias_initializer='zeros', kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                           )
                                  )

                # In-between layers
                else:

                    if sub_type == "lstm":
                        model.add(LSTM(layer_units,
                                       activation=activation_funct,
                                       recurrent_activation=parameters['recurrent_activation_funct'],
                                       use_bias=True,
                                       kernel_initializer=parameters['kernel_initializer'],
                                       recurrent_initializer=parameters['recurrent_initializer'],
                                       bias_initializer='zeros',
                                       unit_forget_bias=True,
                                       kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                       recurrent_regularizer=regularizers.l2(parameters['l2reg']),
                                       dropout=dropout_val,
                                       recurrent_dropout=0,
                                       return_sequences=parameters['return_sequences'],
                                       )
                                  )

                    elif sub_type == "bilstm":
                        model.add(Bidirectional(LSTM(layer_units,
                                       activation=activation_funct,
                                       recurrent_activation=parameters['recurrent_activation_funct'],
                                       use_bias=True,
                                       kernel_initializer=parameters['kernel_initializer'],
                                       recurrent_initializer=parameters['recurrent_initializer'],
                                       bias_initializer='zeros',
                                       unit_forget_bias=True,
                                       kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                       recurrent_regularizer=regularizers.l2(parameters['l2reg']),
                                       dropout=dropout_val,
                                       recurrent_dropout=dropout_val,
                                       return_sequences=parameters['return_sequences'],
                                       )))

                    elif sub_type == "simplernn":
                        model.add(
                                SimpleRNN(layer_units,
                                          activation=activation_funct,
                                          use_bias=True,
                                          kernel_initializer=parameters['kernel_initializer'], recurrent_initializer=parameters['recurrent_initializer'],
                                          bias_initializer='zeros',
                                          kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                          recurrent_regularizer=regularizers.l2(parameters['l2reg']),
                                          dropout=dropout_val,
                                          recurrent_dropout=dropout_val,
                                          return_sequences=parameters['return_sequences'],
                                          )
                                )

                    elif sub_type == "gru":
                        model.add(GRU(layer_units,
                                      activation=activation_funct,
                                      recurrent_activation=parameters['recurrent_activation_funct'],
                                      use_bias=True,
                                      kernel_initializer=parameters['kernel_initializer'], recurrent_initializer=parameters['recurrent_initializer'],
                                      bias_initializer='zeros',
                                      kernel_regularizer=regularizers.l2(parameters['l2reg']),
                                      recurrent_regularizer=regularizers.l2(parameters['l2reg']),
                                      dropout=dropout_val,
                                      recurrent_dropout=dropout_val,
                                      return_sequences=parameters['return_sequences'],
                                      )
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
                             metrics=['acc'])

        # ADD CALLBACKS FOR EARLY STOPPING, ETC
        model.callback_list = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                    min_delta=parameters['cost_delta'],
                                                                    patience=5,
                                                                    verbose=1,
                                                                    mode='auto'
                                                                    ),
                                      keras.callbacks.EarlyStopping(monitor='acc',
                                                                    min_delta=0,
                                                                    patience=5,
                                                                    verbose=1,
                                                                    mode='auto'
                                                                    ),
                                      keras.callbacks.History(),
                                      keras.callbacks.ModelCheckpoint(model_file,
                                                                      monitor='acc',
                                                                      verbose=1,
                                                                      save_best_only=True,
                                                                      mode='auto',
                                                                      period=1)
                                      ]

        # except Exception as e:
        #     print("Error in building model << %s >> : %s" % (sub_type, str(e)))
        #     is_executed = False

        return is_executed, model_file, model
