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
import numpy as np
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, TimeDistributed, SimpleRNN, Bidirectional, Masking
from keras import regularizers, initializers
from keras import optimizers
from keras import backend as K

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO
from modules.vector_ops.vectors import VectorOps

###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class BaseModel(object):
    """
    Class to implement the base functions of all ML models
    - The class is an inherited class and contains all those functions which
    will be used by the ML models to create data models from a given dataset

    """

###############################################################################################################
#
#  1. Basic initializations
#
###############################################################################################################


    def __init__(self, **kwargs):
        """
        Method for basic common model initializations
        """
        self.io = IO()
        self.train_input = None
        self.train_output = None
        self.test_input = None
        self.test_output = None
        self.train_input_shape = None
        self.train_output_shape = None
        self.test_input_shape = None
        self.test_output_shape = None
        self.input_max_rows = None
        self.input_max_cols = None
        self.output_max_rows = None
        self.output_max_cols = None

        self.__parameters = None
        if 'parameters' in kwargs:
            self.__parameters = kwargs['parameters']

        self.__vector_ops = VectorOps()
        return

    def get_default_parameters(self):
        """
        Method to return the default set of initialized parameters to the ModelController class
        """
        return self.__parameters

###############################################################################################################
#
#  2. Data management functions - move to model controller
#
###############################################################################################################

    def set_model_data(self, dataset_type=None, experiment=None, input_data=None,
                       output_data=None, parameters=None, masking=None,
                       train_input_shape=None, train_output_shape=None,
                       padding=None):
        """
        Method to initialize the test data as CLASS variable and not instance variable
        - since multiple instances may be run for the same model

        :dataset_type (str) : the type of dataset --> train, dev or test
        :input_data (list(list)): list of input vectors
        :output_data (list(list)): list of output vectors

        :returns (bool): True if set,
                            False if not set
        """
        is_executed = True

        # DEL
        # input_data = input_data[0:2]
        # output_data = output_data[0:2]
        ###

        if dataset_type in ["dev","test"]:
            if self.train_input_shape is not None and self.train_output_shape is not None:
                if len(self.train_input_shape) == len(self.train_output_shape) == 2:
                    self.input_max_rows = self.train_input_shape[0]
                    self.input_max_cols = self.train_input_shape[1]
                    self.output_max_rows = self.train_output_shape[0]
                    self.output_max_cols = self.train_output_shape[1]
                if len(self.train_input_shape) == len(self.train_output_shape) == 3:
                    self.input_max_rows = self.train_input_shape[1]
                    self.input_max_cols = self.train_input_shape[2]
                    self.output_max_rows = self.train_output_shape[0]
                    self.output_max_cols = self.train_output_shape[1]

        # Getting the data in the right format
        # try:
        assert len(input_data) == len(output_data), "Initialization error: Input and Output data not of the same length"

        # Masking layer is enabled for RNN models
        if masking:
            if padding['input']:
                is_executed, input_data = self.__vector_ops.add_padding(data=input_data,
                                                                    pad_rows=True,
                                                                    pad_cols=False,
                                                                    pad_value=self.__parameters['mask_value'],
                                                                    max_row_len=self.input_max_rows,
                                                                    max_col_len=self.input_max_cols,
                                                                    )

            input_data = np.asarray([np.asarray([np.asarray(each_row) for each_row in each_block]) for each_block in input_data])

            print("\tInput data shape:", input_data.shape)
            self.input_max_rows = input_data.shape[1]
            self.input_max_cols = input_data.shape[2]
            assert input_data.ndim == 3, "Masking (rnn models) require a 3-Dimensional input data"

            if is_executed and padding['output']:
                print("Output data shape: (%d, %d)" % (len(output_data),max([len(each_row) for each_row in output_data])))
                is_executed, output_data = self.__vector_ops.add_padding(data=output_data,
                                                                         pad_rows=False,
                                                                         pad_cols=True,
                                                                         pad_value=self.__parameters['mask_value'],
                                                                         max_row_len=self.output_max_rows,
                                                                         max_col_len=self.output_max_cols,
                                                                         )


            if experiment == "int-cls":

                output_data = np.asarray([np.asarray(each_block) for each_block in output_data])
                self.output_max_rows = output_data.shape[0]
                self.output_max_cols = output_data.shape[1]
                assert output_data.ndim == 2, "Masking (rnn models) require a 2 -Dimensional output data"

            elif experiment == "slot-fill-cls":
                output_data = np.asarray([np.asarray(each_row) for each_row in output_data])
                output_data = output_data.reshape(len(output_data),len(output_data[0]),1)
                #output_data = np.asarray([np.asarray(each_block) for each_block in output_data])
                #output_data = output_data.reshape(output_data.shape[0],output_data.shape[1],1)
                self.output_max_rows = output_data.shape[1]
                self.output_max_cols = output_data.shape[2]
                assert output_data.ndim == 3, "Masking (rnn models) require a 3-Dimensional output data"

            print("\tOutput data shape:", output_data.shape)

        # LR / NN models requiring 2 dimensional data
        else:

            input_data = np.asarray([np.asarray(each_row) for each_row in input_data])
            output_data = np.asarray([np.asarray(each_row) for each_row in output_data])

            print("\tInput data shape:", input_data.shape)
            print("\tOutput data shape:", output_data.shape)

            self.input_max_rows = input_data.shape[0]
            self.input_max_cols = input_data.shape[1]
            self.output_max_rows = output_data.shape[0]
            self.output_max_cols = output_data.shape[1]

            assert input_data.ndim == 2, "LR / NN models require a 2-Dimensional input data"
            assert output_data.ndim == 2, "LR / NN models require a 2-Dimensional output data"

        # except Exception as e:
        #     print("Error (set_model_data): %s" % str(e))
        #     is_executed = False

        # Initializing the model data variables
        if is_executed:
            if dataset_type == "train":
                self.train_input = input_data
                self.train_output = output_data

                self.train_input_shape = self.train_input.shape
                self.train_output_shape = self.train_output.shape

            elif dataset_type in ["dev", "test"]:
                self.test_input = input_data
                self.test_output = output_data

                self.test_input_shape = self.test_input.shape
                self.test_output_shape = self.test_output.shape

        return is_executed

    def input_output_data_generator(self, input_data, output_data):
        """
        Method to generate input/output data sequences @ run-time during model_fit
        USED with model.fit_generate() where each "BLOCK" of data can be passed as input
        - A "BLOCK" of data implies SAME dimensionality of all samples within that block

        :input_data (list/np.ndarray/np.matrix): input data
        :output_data (list/np.ndarray/np.matrix): output data
        """
        if isinstance(input_data, list):
            assert len(input_data) == len(output_data), "input and output data need to be of the same length"

            while True:
                for i in range(len(input_data)):
                    input_seq = np.asarray(input_data[i])
                    input_seq = input_seq.reshape(1, input_seq.shape[0], input_seq.shape[1])
                    output_seq = np.asarray(output_data[i]).reshape(1,len(output_data[i]))
                    yield (input_seq, output_seq)

        elif isinstance(input_data, np.ndarray) or isinstance(input_data, np.matrix):
            assert input_data.shape[0] == output_data.shape[0], "input and output data need to be of the same length"

            while True:
                for i in range(input_data.shape[0]):
                    input_seq = input_data[i]
                    input_seq = input_seq.reshape(1, input_seq.shape[0], input_seq.shape[1])
                    output_seq = output_data[i]
                    output_seq = output_seq.reshape(1, output_seq.shape[0])
                    yield (input_seq, output_seq)

###############################################################################################################
#
#  3. Data modelling functions
#
###############################################################################################################

    def train(self, model_file=None, model_dir=None, parameters=None):
        """

        Method to train the model / create a data model from the data that is passed as input

            1. The model is built/compiled as an intial step, based on the parameters supplied as input.
            2. The built model is then trained (fit()) on the data that was read
                2.1 At each iteration, a CHECKPOINT will be created if loss & accuracy are better than the previous iteration. That checkpoint is stored as the BEST model

        The model can be saved, reloaded and used again for new data

        :model_file (str) : The name by which the trained data model will be stored
        :model_dir (str) : The path on which the model_file will be stored
        :parameters (dict(str:(any))) : The list of (hyper)parameters which have been passed
                                        to the model

        :returns (bool) : True, if the model was trained successfully; else False

        """

        is_executed = True

        checkpoint_file = model_file + ".mdl"
        trained_model_file = model_file + ".mdl"

        if is_executed:
            # The model_file is given to build_model() which creates a ModelCheckPoint to automatically store the best model at each model iteration
            # So, no need to store to model explicitely!!
            is_executed, model_file, model = self.build_model(model_file=checkpoint_file,
                                                       model_dir=model_dir, parameters=parameters)

            print("Model Configuration:\n\t", model.summary())

            if is_executed:

                # Train model
                try:
                    model_history = model.fit(self.train_input, self.train_output,
                                              batch_size=100,
                                              epochs=parameters['max_epochs'],
                                              verbose=1, shuffle=True,
                                              initial_epoch=0,
                                              callbacks=model.callback_list)

                    last_loss = str(model_history.history['loss'][-1])
                    if last_loss == "nan":
                        is_executed = False
                        print("Model exited with a NaN value!")
                    else:
                        print("Model trained successfully!")
                except Exception as e:
                    print("Error: Model failed to train!! << %s >> " % str(e))
                    is_executed = False

        del(model)
        return is_executed

    def test(self, trained_model=None, test_dir=None):
        """

        Method to run a trained model over test data
        - The model takes the path + filename of the trained data model
            and inputs the test data into the trained model to generate predictions
        - The predictions are stored in a specific test directory with the same
            name as the trained model
            - The predictions are pickled and gzipped for later evaluations

        :trained_model (str) : The path + filename of the trained model
        :test_dir (str) : The path of the test directory where the predictions will be saved

        :returns (bool) : True, if the predictions could be generated, else False

        """

        is_executed = True

        test_file = os.path.basename(trained_model) + ".pkl.gz"

        model = load_model(trained_model)

        try:
            predictions = model.predict(self.test_input, verbose=1)
        except Exception as e:
            print("Error: << %s >>.\n\tTry running 'train' and 'test' in sequence" % str(e))

        if predictions.shape == self.test_output.shape:
            is_executed = True
            if isinstance(predictions, np.matrix) or isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

            # SAVE THE PREDICTIONS
            if not self.io.write_file(path=test_dir, filename=test_file, format="pkl.gz", data=predictions):
                is_executed = False
        else:
            print("Prediction shape and Test output shapes do not match")

        return is_executed

    def get_network_layer_data(self, parameters=None):
        """
        Method to create a set of nested layers for the ML model
        - All layers except input layers are built here iteratively
        - All layer based information is supplied at the time of program initialization

        """
        is_executed = True
        network_layer_data = None

        if 'hl_sizes' not in parameters:
            print("The model has no hidden layers (hl_sizes) specified")
            is_executed = False
        elif not isinstance(parameters['hl_sizes'], int) and not isinstance(parameters['hl_sizes'], list) and not isinstance(parameters['hl_sizes'], tuple):
            print("The hidden layers (hl_sizes) parameter can only be of Type 'int' or 'list'")
            is_executed = False

        if isinstance(parameters['hl_sizes'], int):
            layer_units = [parameters['hl_sizes']] + [parameters['output_dim']]
            activations = [parameters['activation_funct']] + [parameters['output_activation_funct']]
        else:
            layer_units = [layer_unit for layer_unit in parameters['hl_sizes']] + [parameters['output_dim']]
            activations = [activation for activation in parameters['activation_funct']] + [parameters['output_activation_funct']]

        dropouts = [parameters['dropout']] * len(layer_units)

        # Just to make sure that even if the last layer has a 'dropout' BY MISTAKE, then
        # the value is 1.0, so that it does not affect the output of the last layer
        dropouts[-1] = -1

        try:
            assert len(layer_units) == len(activations) == len(dropouts), "len(layer_units) != len(activations) != len(dropouts) :: << %d,%d,%d >>" % (len(layer_units), len(activations), len(dropouts))

            network_layer_data = zip(layer_units, activations, dropouts)

        except Exception as e:
            print("Error: The network layer information parameters to not match in lengths.\n\t<< %s >>" % str(e))
            is_executed = False

        return is_executed, network_layer_data
