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
from collections import defaultdict, OrderedDict, Mapping

# >>>>>>>>> Package Imports <<<<<<<<<<
import numpy as np
from keras import optimizers

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO
from modules.data_ops.multiprocessor import Multiprocessor
from modules.data_ops.python_ops import PythonOps
from modules.models import LinearRegression, NeuralNetwork, RecurrentNeuralNetwork
from modules.vector_ops.evaluations import PredictionEvaluator

###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################


class ModelController(object):
    """
    Controller class for various ml model initializations
        - Can read/load datasets
        - Can initialize models and model parameters
        - Can create permutations of model parameters for testing
        - initialize training and testing of models
        - returning results of models

    :initialize_model()
        :__create_parameter_stack()
    :initialize_parameters()
    :initialize_datasets()
    :train()
    :test()
    :evaluate()


    """

    __data = defaultdict(dict)

    def __init__(self, multiprocessing=False):
        """
        Method for basic model initializations
        """
        self.io = IO()
        self.__multiprocessing = multiprocessing
        self.__model_obj = None
        self.__experiment = None
        self.__dataset_name = None
        self.__model_file_prefix = None
        self.__dataset_types = ["train", "dev", "test"]
        self.__dataset_content = ["vectors", "mappings", "one-hot-mappings"]
        self.__parameter_stack = None
        self.__multiprocessor = Multiprocessor()
        self.__python_ops = PythonOps()

        return

    def initialize_model(self, model=None, experiment=None, model_params=None,
                         dataset_name=None, result_dir=None):
        """
        Method to initialize the model according to experiment and model given by the main controller

        Creates a parameter stack recursively for EACH model so that various hyper-parameters can be tested to find th best model fit

        :returns (bool) : True, if model initialized successfully, else False
        """
        is_executed = True
        self.__model_name = model
        self.__experiment = experiment
        self.__model_execution_parameters = model_params[self.__experiment]['parameters'][self.__model_name]
        self.__experiment_padding_parameters = model_params[self.__experiment]['padding']
        self.__model_file_prefix = "_".join([dataset_name, experiment, model])
        self.__dataset_name = dataset_name

        if self.__model_name == "lr":
            self.__model = LinearRegression()
        elif self.__model_name == "nn":
            self.__model = NeuralNetwork()
        elif self.__model_name == "rnn":
            self.__model = RecurrentNeuralNetwork()

        self.__parameter_stack = {}

        # CREATING THE PARAMETER STACK - RECURSIVELY
        is_executed, self.__parameter_stack = self.__create_parameter_stack(supplied_param_dict=self.__model_execution_parameters,
                                    default_param_dict=self.__model.get_default_parameters(),
                                    model_name=self.__model_name)

        # STORING THE PARAMETERS CREATED INITIALLY FOR LATER EVALUATION
        # Only a part of these parameters will be executed.
        # Can be checked from train, test and evaluation runs
        if is_executed:
            # Check for the evaluation directory
            eval_dir = os.path.join(result_dir, "evaluations")
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)

            eval_file = self.__model_file_prefix + "_trained_model_parameters.json"

            if not self.io.write_file(path=eval_dir, filename=eval_file, format="json", data=self.__parameter_stack):
                is_executed = False

        return is_executed

    def __create_parameter_stack(self, supplied_param_dict=None, default_param_dict=None, model_name=None):
        """
        Module to create an iterable parameter stack which can be used to experiment with different parameters at one go!

        - Generate iterative param stack for the model => Cross product of all parameters
        stack = {0:(p11, p21, p31, .. pn1), 1:(p12, p22, p32, .. pn2)..  }
        tuple_map => Order of parameters in the tuple

        :supplied_param_dict (dict(str:list)): a list of parameters sent by the main controller
        :default_param_dict (dict(str:list)): default parameters extracted from the model instance
        :model_name (str) : name of the model

        returns (bool, stack_dict):
            - bool : True if the stack was created successfully
            - stack_dict (dict(int:dict)) : A dictionary of all possible combinations that could be created in the stack
                - int : represents the model number later on
                - dict : is the list of parameters (atleast) which is included in the model scripts __init__() function
        """
        is_executed = True
        if supplied_param_dict is not None:
            # Get the list of parameter names
            curr_param = 0
            stack_list = []
            param_list = list(supplied_param_dict.keys())
            total_params = len(param_list)

            # CREATING THE STACK
            # only curr_param and stack_list keep getting updated in this recursive call
            stack_list = self.__create_stack(curr_param, stack_list, total_params, param_list, supplied_param_dict)

            # Formating the extracted parameters in an easy to read dict() format
            stack_dict = {i:stack_list[i] for i in range(len(stack_list))}
            tuple_map = {param_list[i]:i for i in range(len(param_list))}

        else:
            print("NO ITERABLE PARAMETERS PASSED TO CREATE THE STACK.\n\tCREATING STACK ON DEFAULT PARAMETERS !!")
            stack_dict = {}
            tuple_map = {}
            is_executed = False

        # ONCE the stack is created, there might be some default parameters that might have been missed
        # at the time of input.
        # The following function takes a list of default parameters from the model instance and fills up the missing parameters, if required
        stack_dict = self.__refactor_stack_to_include_ALL_parameters(default_param_dict=default_param_dict,
                                                                     stack_dict=stack_dict, tuple_map=tuple_map)

        if not len(stack_dict) > 0:
            is_executed = False

        return is_executed, stack_dict

    def __create_stack(self, curr_param, stack_list, total_params, param_list, supplied_param_dict):
        """
        Module to create a prameter stack recursively. Aims to create a recursive stack of the parameters.
        """
        # Finished reading all parameters. The stack should be built by now.
        if curr_param == total_params:
            return stack_list

        # Get the current parameter values
        curr_values = supplied_param_dict[param_list[curr_param]]

        # If reading the 1st parameter - just make a list of it
        if curr_param == 0:
            stack_list = [[each_value] for each_value in curr_values]
        else:
            # Else, ADD new parameter values to the earlier parameters
            stack_list = [each_param_list + each_value if isinstance(each_value, list) else each_param_list + [each_value] for each_param_list in stack_list for each_value in curr_values]

        curr_param += 1
        stack_list = self.__create_stack(curr_param, stack_list, total_params, param_list, supplied_param_dict)
        return stack_list

    def __refactor_stack_to_include_ALL_parameters(self, default_param_dict=None, stack_dict=None,
                                                   tuple_map=None):
        """
        Module to create a new parameter stack where all default parameters also get included with the iterative parameters.

        Here, we create a new_param_dict by combining the default model parameters and the ones specified by the user.

        """

        new_stack_dict = {}
        if len(stack_dict) > 0:
            for row, param_tuple in stack_dict.items():
                new_param_dict = {}
                for eachParam in default_param_dict:
                    if eachParam in tuple_map:
                        new_param_dict[eachParam] = param_tuple[tuple_map[eachParam]]
                    else:
                        new_param_dict[eachParam] = default_param_dict[eachParam]
                new_stack_dict[row] = new_param_dict
        else:
            new_stack_dict[0] = default_param_dict

        return new_stack_dict

    def load_datasets(self, dataset_name=None, space_name=None, data_root_dir=None):
        """
        Method to load the vectorized Train, Test and Dev datasets for a given dataset

        :dataset_name (str) : name of the dataset
        :space_name (str) : name of the vector space
        :data_root_dir (str) : path of the root data directory

        :returns (bool) : True if the datasets were loaded successfully

        """
        is_executed = True
        dataset_types = self.__dataset_types
        dataset_content = self.__dataset_content
        dataset_search_str = "_".join([dataset_name, space_name, self.__experiment, self.__model_name])
        dataset_files = self.io.search_files(path=data_root_dir, search_string=dataset_search_str, match_type="partial")

        if len(dataset_files) < 1:
            is_executed = False
            print("Found no dataset files to load for the search string << %s >>" % dataset_search_str)

        if is_executed:
            for each_dataset_type in dataset_types:
                for each_dataset_content in dataset_content:
                    for each_file in dataset_files:
                        if each_file.find("_".join([each_dataset_type, each_dataset_content])) > -1:
                            self.__data[each_dataset_type][each_dataset_content] = self.io.read_file(path=os.path.dirname(each_file),
                                            filename=os.path.basename(each_file),
                                            format="pkl.gz")

                            if self.__data[each_dataset_type][each_dataset_content] is None:
                                is_executed = False
                                print("Error in reading file << %s >>" % each_file)

                        if not is_executed:
                            break
                    if not is_executed:
                        break
                if not is_executed:
                    break

        return is_executed

    def train(self, result_dir=None):
        """
        Method to train the model, based on a particular experiment and dataset

            - 'n' number of models can be trained, depending on the size of the parameter stack
            - The models can be executed in parallel by turning ON the multiprocessor switch from
                the command line arguments
            - The trained models are stored in a file in the results_dir/trained_models directory

        :result_dir (str) : The location of the result directory which stores the data

        :returns (bool) : True, if the models were trained successfully
        """
        is_executed = True
        if is_executed:
            # Load the training data
            for each_dataset_type in self.__dataset_types:
                for each_content_type in self.__dataset_content:
                    if each_content_type == "vectors" and each_dataset_type == "train":
                        is_executed = self.__model.set_model_data(dataset_type=each_dataset_type,
                            experiment=self.__experiment,
                            input_data=self.__data[each_dataset_type][each_content_type]['input'], output_data=self.__data[each_dataset_type][each_content_type]['output'],
                            masking=True if 'masking' in self.__model_execution_parameters and self.__model_execution_parameters['masking'] else False,
                            train_input_shape=None,
                            train_output_shape=None,
                            padding=self.__experiment_padding_parameters,
                            )

        # CREATE a revised stack of parameters by removing those elements from the stack
        # WHERE the parameters don't get validated
        if is_executed:
            if len(self.__parameter_stack) < 1:
                is_executed = False
                print("No parameters to initialize the model with in model.train()")
            else:
                revised_stack = {}

                for model_id, parameters in self.__parameter_stack.items():
                    valid_parameter, parameters = self.__validate_parameters(parameters=parameters,
                                            model_parameters=self.__model.get_default_parameters(), verbose=False)
                    if valid_parameter:
                        revised_stack[model_id] = parameters

                self.__parameter_stack = revised_stack

        # Create a parameter stack list, in case multiprocessing has to be used
        # ALSO, delete all those models which are going to be RE-RUN
        if is_executed:
            # Create a path to save the models
            is_executed, model_dir = self.io.make_directory(path=result_dir, dir_name="trained_models")
            if is_executed:
                model_argument_list = [("_".join([self.__model_file_prefix, str(model_id)]),
                                        model_dir,
                                        parameters
                                        ) for model_id, parameters in self.__parameter_stack.items()
                                       ]
                files = [os.path.join(model_dir, model_file + ".mdl") for (model_file, model_dir, parameters) in model_argument_list]
                is_executed = self.io.delete_files(file_list=files)

        # RUN the training module of the model instance
        if is_executed:
            # Normal mode
            if not self.__multiprocessing:
                for model_id, parameters in self.__parameter_stack.items():
                    model_file = "_".join([self.__model_file_prefix, str(model_id)])
                    is_executed = self.__model.train(model_file=model_file,
                                                     model_dir=model_dir,
                                                     parameters=parameters)
                    if not is_executed:
                        print("Error: in TRAINING model_id << %d >>" % model_id)

            # Multiprocessor mode
            else:
                return_values = self.__multiprocessor.run_pool(function=self.__model.train,
                                                               arguments=model_argument_list,
                                                               ret_val=True
                                                               )

                if not all(return_values):
                    is_executed = False

        return is_executed

    def __validate_parameters(self, parameters=None, model_parameters=None, verbose=False):
        """
        Method to validate the parameters passed as input for model training

        :parameters (dict(str:str|int|float)) : list of parameters from the parameter stack

        """
        is_executed = True


        if not all([True if each_param in parameters else False for each_param in model_parameters]):
            is_executed = False
            if verbose:
                print("Some necessary parameters have not been initialzied: ", model_parameters.keys())

        if is_executed:
            for param, param_value in parameters.items():
                if param == "input_dim":
                    if not (isinstance(param_value, int) or isinstance(param_value, float) or param_value.isdigit()):
                        if verbose:
                            print("'input_dim' can only be a int, float OR str (with digits)")
                        is_executed = False
                    else:
                        # Set value from the data
                        if param_value == 0:

                            if self.__model.train_input_shape is not None:
                                if len(self.__model.train_input_shape) == 2:
                                    param_value = self.__model.train_input_shape[1]
                                if len(self.__model.train_input_shape) == 3:
                                    param_value = self.__model.train_input_shape[1]
                            else:
                                print("Error: Input data has NOT been initialized!")
                        else:
                            param_value = int(param_value)

                if param == "output_dim":
                    if not (isinstance(param_value, int) or isinstance(param_value, float) or param_value.isdigit()):
                        if verbose:
                            print("'output_dim' can only be a int, float OR str (with digits)")
                        is_executed = False
                    else:
                        # Set value from the data
                        if param_value == 0:
                            if self.__model.train_output_shape is not None:
                                if len(self.__model.train_output_shape) == 2:
                                    param_value = self.__model.train_output_shape[1]
                                if len(self.__model.train_output_shape) == 3:
                                    param_value = self.__model.train_output_shape[2]
                            else:
                                print("Error: Output data has not been initialized")
                        else:
                            param_value = int(param_value)

                if param == "sub_type":
                    if not (isinstance(param_value, str) or isinstance(param_value, list)):
                        if verbose:
                            print("'sub_type' can only be an str OR list: [simplernn, rnn, bilstm, lstm, gru]")
                            is_executed = False
                    else:
                        sub_types = ["simplernn","bilstm","lstm","gru"]
                        if isinstance(param_value, str):
                            if not param_value in sub_types:
                                print("'sub_type' value can only be in [simplernn, bilstm, lstm, gru]")
                                is_executed = False
                        elif isinstance(param_value, list):
                            if not all([True if each_param_value in sub_types else False for each_param_value in param_value]):
                                is_executed = False
                                print("'sub_type' value can only be in [simplernn, rnn, bilstm, lstm, gru]")

                if param == "learning_rate":
                    if not (isinstance(param_value, float) or isinstance(param_value, int)):
                        if verbose:
                            print("'learning_rate' has to be a float OR int")
                        is_executed = False
                    else:
                        param_value = float(param_value)

                if param == "activation_funct":
                    if not isinstance(param_value, tuple):
                        if 'hl_sizes' in parameters and not isinstance(parameters['hl_sizes'], int):
                            if verbose:
                                print("'activation_funct' is None BUT there are hidden layers specified for the model.")
                            is_executed = False
                    else:
                        if 'hl_sizes' not in parameters:
                            if verbose:
                                print("'activation_funct' has values but no hidden layers initialized 'hl_sizes'")
                            is_executed = False
                        elif isinstance(parameters['hl_sizes'], int):
                            if verbose:
                                print("'activation_funct' has values but no hidden layers initialized 'hl_sizes'")
                            is_executed = False
                        elif len(parameters['hl_sizes']) != len(parameters[param]):
                            if verbose:
                                print("'activation_funct' and 'hl_sizes' lengths do not match")
                            is_executed = False

                if param == "output_activation_funct":
                    if param_value not in ['softmax', 'sigmoid', 'tanh','relu','leakyrelu'] or param_value is None:
                        if verbose:
                            print("'output_activation_funct' can only be in ['softmax', 'sigmoid', 'tanh','relu','leakyrelu']")
                        is_executed = False

                if param == 'cost_funct':
                    if param_value not in ['binary_crossentropy', 'crossentropy_softmax_1hot', 'crossentropy_categorical_1hot', 'mean_squared_error', 'negative_log_likelihood', 'categorical_crossentropy', 'kullback_leibler_divergence', 'cosine_proximity']:
                        if verbose:
                            print("'cost_funct' can only be in ['binary_crossentropy', 'crossentropy_softmax_1hot', 'crossentropy_categorical_1hot', 'mean_squared_error', 'negative_log_likelihood', 'categorical_crossentropy', 'kullback_leibler_divergence', 'cosine_proximity']")
                        is_executed = False

                if param == 'l2reg':
                    if not isinstance(param_value, float):
                        if verbose:
                            print("'l2reg' requries a float value")
                        is_executed = False

                if param == 'l1reg':
                    if not isinstance(param_value, float):
                        if verbose:
                            print("l1reg requires a float value")
                        is_executed = False

                if param == 'optimizer':
                    if param_value not in ["adadelta","sgd","adam"]:
                        if verbose:
                            print("'optimizer' can be in ['adadelta','sgd','adam']")
                        is_executed = False

                if param == 'max_epochs':
                    if not isinstance(param_value, int):
                        if verbose:
                            print("'max_epochs' can only be an int")
                        is_executed = False

                if param == 'cost_delta':
                    if not isinstance(param_value, float):
                        if verbose:
                            print("'cost_delta' can only be a float")
                        is_executed = False

                if param == "dropout":
                    if not isinstance(param_value, float):
                        if verbose:
                            print("'dropout' value can only be a float")
                        is_executed = False

                if param == 'return_sequences':
                    if not param_value in [True, False]:
                        if verbose:
                            print("'return_sequences' can only be boolean")
                        is_executed = False

                if param == 'masking':
                    if not param_value in [True, False]:
                        if verbose:
                            print("'masking' can only be boolean")
                        is_executed = False

                if "hl_sizes" in parameters:
                    if isinstance(parameters['hl_sizes'], tuple) or isinstance(parameters['hl_sizes'], list):
                        if len(parameters['hl_sizes']) != len(parameters['activation_funct']):
                            if verbose:
                                print("'hl_sizes' and 'activation_funct' are not of the same sizes")
                            is_executed = False

                if is_executed:
                    parameters[param] = param_value

        return is_executed, parameters

    def test(self, result_dir=None):
        """
        Method to run the test data on the trained model(s)
            - The module picks up all the trained models for a given dataset
            - It iteratively (or in multiprocessing mode) inputs the test data for
                EACH trained model and generates predictions on that model
            - The predictions are stored in the result_dir/test_models directory as pickled and gzipped files

        :result_dir (str) : Location of the result directory

        :returns (bool) : True if the test data ran on the data model, else False
        """
        is_executed = True

        # Check if there is a model directory
        model_dir = os.path.join(result_dir, "trained_models")
        if not os.path.isdir(model_dir):
            is_executed = False
            print("No 'trained_models' directory in the 'result_dir' @ << %s >>" % result_dir)
        else:
            dataset_files = self.io.search_files(path=model_dir, search_string=self.__model_file_prefix,
                                                  match_type="partial")
            model_files = self.io.search_files(path=model_dir, search_string=".mdl",
                                                  match_type="partial")

            trained_models = sorted(set(dataset_files).intersection(set(model_files)))

            if len(trained_models) < 1:
                print("No 'trained_models' directory in the 'result_dir' @ << %s >>" % result_dir)
                is_executed = False

        # Check if there is a directory to save results
        is_executed, test_dir = self.io.make_directory(path=result_dir, dir_name="test_results")
        if is_executed:
            files = [os.path.join(test_dir, os.path.basename(trained_model) + ".pkl.gz") for trained_model in trained_models]
            is_executed = self.io.delete_files(file_list=files)

        if is_executed:

            for each_dataset_type in self.__dataset_types:

                for each_content_type in self.__dataset_content:

                    if each_content_type == "vectors" and (each_dataset_type == "dev" or each_dataset_type == "test"):

                        if each_dataset_type in self.__data and each_content_type in self.__data[each_dataset_type]:

                            # Call and SET training data to get its shape and dimensions
                            #print(self.__model_execution_parameters)
                            if self.__model.train_input_shape is None or self.__model.train_output_shape is None:
                                is_executed = self.__model.set_model_data(dataset_type="train",
                                    experiment=self.__experiment,
                                    input_data=self.__data['train'][each_content_type]['input'], output_data=self.__data['train'][each_content_type]['output'],
                                    masking=True if 'masking' in self.__model_execution_parameters and self.__model_execution_parameters['masking'] else False,
                                    train_input_shape=None,
                                    train_output_shape=None,
                                    padding=self.__experiment_padding_parameters,
                                    )

                            # Set test data
                            is_executed = self.__model.set_model_data(dataset_type=each_dataset_type,
                                experiment=self.__experiment,
                                input_data=self.__data[each_dataset_type][each_content_type]['input'], output_data=self.__data[each_dataset_type][each_content_type]['output'],
                                masking=True if 'masking' in self.__model_execution_parameters and self.__model_execution_parameters['masking'] else False,
                                train_input_shape=self.__model.train_input_shape,
                                train_output_shape=self.__model.train_output_shape,
                                padding=self.__experiment_padding_parameters,
                                )

        # RUN the test data on the data model
        if is_executed:
            # Normal mode
            if not self.__multiprocessing:
                for each_trained_model in trained_models:
                    is_executed = self.__model.test(trained_model=each_trained_model,
                                                    test_dir=test_dir)
            else:
                # Create a multiprocessor argument list
                argument_list = [(each_trained_model, test_dir) for each_trained_model in trained_models]
                return_values = self.__multiprocessor.run_pool(function=self.__model.test,
                                                               arguments=argument_list,
                                                               ret_val=True
                                                               )

                if not all(return_values):
                    is_executed = False

        return is_executed

    def evaluate(self, result_dir=None, evaluation_actions=None):
        """
        Method to evaluate the predictions of the test data on the data models

            - The evaluations are stored in the result_dir/evaluations directory

        :result_dir (str) : location of the result directory where the trained and tested models are stored
        :evaluation_actions (list(tuple)): A list of (model, evaluation) supplied from the main function
                                to execute a relevant evaluation on the predictions

        :returns (bool) : True if the evaluations could be executed, else False
        """
        is_executed = True

        if self.__dataset_name.find("atisnlu") > -1:

            masking=True if 'masking' in self.__model_execution_parameters and self.__model_execution_parameters['masking'] else False

            if os.path.isfile(os.path.join(result_dir, self.__dataset_name + "_MinMaxScaler.pkl")):
                minmaxscaler_obj = self.io.read_file(path=result_dir,
                filename=self.__dataset_name + "_MinMaxScaler.pkl", format="pkl")


        # Check if there is a model directory
        model_dir = os.path.join(result_dir, "trained_models")
        if not os.path.isdir(model_dir):
            is_executed = False
            print("No 'trained_models' directory in the 'result_dir' @ << %s >>" % result_dir)

        # Check if there is a directory to save results
        pred_dir = os.path.join(result_dir, "test_results")
        if not os.path.isdir(pred_dir):
            is_executed = False
            print("No 'test_results' directory in the 'result_dir' @ << %s >>" % result_dir)

        # Get the prediction / test data files form the test results directory
        if is_executed:
            pred_data_files = self.io.search_files(path=pred_dir,
                                                   search_string=self.__model_file_prefix,
                                                   match_type="partial")

            pred_data_files = sorted(pred_data_files)

            if len(pred_data_files) < 1:
                is_executed = False

        test_output_data = None
        test_mappings = None
        test_one_hot_mappings = None

        # READ the ORIGINAL test data
        if is_executed:
            for each_dataset_type in self.__dataset_types:
                for each_content_type in self.__dataset_content:
                    if (each_dataset_type == "dev" or each_dataset_type == "test") and each_dataset_type in self.__data:
                        if each_content_type == "vectors" and each_content_type in self.__data[each_dataset_type]:
                            test_output_data = self.__data[each_dataset_type][each_content_type]['output']
                        if each_content_type == "mappings" and each_content_type in self.__data[each_dataset_type]:
                            test_mapping = self.__data[each_dataset_type][each_content_type]['output']
                        if each_content_type == "one-hot-mappings" and each_content_type in self.__data[each_dataset_type]:
                            test_one_hot_mapping = self.__data[each_dataset_type][each_content_type]['output']

        # Validate the test data superficially
        if (test_output_data and test_mapping and test_one_hot_mapping) is None:
            is_executed = False

        # INVOKE the evaluation class
        if is_executed:
            evaluator = PredictionEvaluator()

            # Determine which evaluations have to be executed for the experiment and model
            actions = [evaluation_action for (experiment, evaluation_action) in evaluation_actions if experiment == self.__experiment]
            evaluation_dict = OrderedDict()

            # For each action, read the prediciton data and call a relevant evaluation
            for each_action in actions:
                if each_action not in evaluation_dict:
                    evaluation_dict[each_action] = {}

                # Read the prediction data
                for each_file in pred_data_files:
                    pred_data = self.io.read_file(path=os.path.dirname(each_file),
                                                  filename=os.path.basename(each_file),
                                                  format="pkl.gz")

                    # Validate the prediciton and test data superficially
                    if not len(pred_data) > 0 or len(pred_data) != len(test_output_data):
                        is_executed = False
                        print("Error: Difference in prediciton and test data << %s >>" % each_file)
                        print("Predictions", len(pred_data))
                        print("Test data", len(test_output_data))

                    else:

                        # PREPARE the evaluation parameters and data
                        arg_dict = {}
                        if each_action == "accuracy-one-hot":
                            arg_dict = {"metric":each_action,
                                        "predictions":pred_data,
                                        "observed":test_output_data
                                        }

                        if each_action == "fscore":
                            arg_dict = {"metric":each_action,
                                        "predictions":pred_data,
                                        "observed":test_output_data,
                                        "mapping":test_mapping,
                                        "dataset_name":self.__dataset_name
                                        }
                            if self.__dataset_name.find("atisnlu") > -1:
                                arg_dict["masking"] = masking
                                arg_dict["scaler_obj"] = minmaxscaler_obj

                        # RUN the evaluation based on the evaluation 'metric'
                        is_executed, results = evaluator.evaluate(**arg_dict)

                    if not is_executed:
                        print("Error: Could not evaluate action << %s, %s >>" % (self.__experiment, each_action))
                    else:
                        # CREATE / STORE THE EVALUATION DATA IN A VARIABLE
                        # Eg. file: atisnlu-dev_int-cls_lr_0.mdl.pkl.gz
                        dataset, experiment, model, model_id = os.path.basename(each_file).split(".")[0].split("_")

                        model_id = int(model_id)

                        if self.__dataset_name is None:
                            self.__dataset_name = dataset
                        if experiment not in evaluation_dict[each_action]:
                            evaluation_dict[each_action][experiment] = {}
                        if model not in evaluation_dict[each_action][experiment]:
                            evaluation_dict[each_action][experiment][model] = []

                        evaluation_dict[each_action][experiment][model].append((results,model_id))

                        evaluation_dict[each_action][experiment][model] = sorted(evaluation_dict[each_action][experiment][model], reverse=True)

        if len(evaluation_dict) < 1:
            is_executed = False

        # STORE THE EVALUATIONS
        if is_executed:
            eval_dir = os.path.join(result_dir, "evaluations")
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)

            eval_file = self.__model_file_prefix + "_eval.json"

            prev_eval_dict = None
            if os.path.isfile(os.path.join(eval_dir, eval_file)):
                prev_eval_dict = self.io.read_file(path=eval_dir, filename=eval_file, format="json")

            if prev_eval_dict is not None:
                prev_eval_dict = self.__python_ops.dict_merge(prev=prev_eval_dict,
                                                              new=evaluation_dict)

                if not self.io.write_file(path=eval_dir, filename=eval_file,
                                          format="json",  data=prev_eval_dict):
                    is_executed = False
            else:
                if not self.io.write_file(path=eval_dir, filename=eval_file, format="json",  data=evaluation_dict):
                    is_executed = False

        return is_executed
