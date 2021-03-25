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
from collections import defaultdict, OrderedDict, Counter

# >>>>>>>>> Package Imports <<<<<<<<<<
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO
from modules.data_ops.atisnlu_data_reader import AtisnluDataReader
from modules.vector_ops.vectors import VectorOps

###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class DatasetBuilder(object):
    """
    Class to build vectorized datasets from a dataset and a vector space
        - The class can receive as input a dataset and a vector space file
        - Thereafter, based on the models and experiments specified as command line arguments it will build the datasets for those models and experiments
            - There are certain parameters like input, output types of the datasets, percentage of division, etc.
            - These parameters have to be initialized from the main program (or a file)


    :build_dataset() : method to build a space from a collection of vectors and the dataset vocabulory

    """

    def __init__(self):
        """
        Method to initialize basic class instance variables
        """
        self.io = IO()
        self.__vector_ops = VectorOps()
        self.__dataset_reader = None
        self.__built_dataset = {}
        self.__space = None
        self.__dataset_stats = {}

        ### Data mod variables
        self.__normalize = normalize
        self.__minmaxscaler_obj = MinMaxScaler()
        return

    def build_dataset(self, dataset_name=None, data=None, space_file=None, data_params=None,
                      dataset_params=None, model_params=None, experiments_allowed=None):
        """
        Method to build a dataset
            - Assumption: the dataset has be 'read'; a 'space' from the dataset vocabulory has been reated
                - The experiment and models are known
                - A vectorized dataset will be built for a specific dataset, which might have some dataset specific functionality

        :dataset_name (str) : name of the dataset
        :data (dict(str:list(str|list(list)))) : the dataset's data, read into a dictionary
        :space_file (str) : the file which stores the vectors
        :data_params dict(str:str|list|dict) : the data parameters initialized from command line arguments
        :dataset_params dict(str:str|list|dict) : the dataset specifications
        :model_params dict(str:str|list|dict) : the model parameters initialized from command line arguments

        returns (bool) : True, if the dataset could be created
                            False, if the dataset could not be created

        """
        is_executed = True

        # Initialize basic variables
        experiments = model_params['experiments']
        models = model_params['models']
        space_name = os.path.splitext(os.path.basename(space_file))[0].split("_")[1]

        # Load the vector space for this dataset
        self.__space = self.io.read_file(path=os.path.dirname(space_file), filename=os.path.basename(space_file),
                                         format="pkl")
        if self.__space is not None:
            is_executed = True

        if is_executed:
            # Invoke a dataset builder module according to the dataset name
            if dataset_name.find("atisnlu") > -1:
                self.__dataset_reader = AtisnluDataReader()
                is_executed = self.__build_dataset_atisnlu(data=data, data_params=data_params,
                                                           dataset_name=dataset_name, space_name=space_name,
                                                           dataset_params=dataset_params, experiments=experiments,
                                                           models=models, experiments_allowed=experiments_allowed)

        # If the datasets are created, then SAVE the dataset statistics
        if is_executed:
            if len(self.__dataset_stats) > 0:
                dataset_stat_filename = "Vectorized_Dataset_Statistics.json"
                save_path = data_params['result_statistics_dir']
                if os.path.exists(os.path.join(save_path, dataset_stat_filename)):
                    prev_dataset_stats = self.io.read_file(path=save_path, filename=dataset_stat_filename,
                                                           format="json")
                    prev_dataset_stats.update(self.__dataset_stats)
                    if not self.io.write_file(path=save_path, filename=dataset_stat_filename,
                                              format="json", data=prev_dataset_stats):
                        is_executed = False
                else:
                    if not self.io.write_file(path=save_path, filename=dataset_stat_filename,
                                              format="json", data=self.__dataset_stats):
                        is_executed = False

        if not is_executed:
            print("Failed to build the vectorized dataset for << %s >>" % dataset_name)
        return is_executed

###############################################################################################################
#
#  1. ATISNLU dataset creation modules
#
###############################################################################################################

    def __build_dataset_atisnlu(self, data=None, dataset_name=None, space_name=None, data_params=None,
                                dataset_params=None, experiments=None, models=None, experiments_allowed=None):
        """
        Method to build the vectorized dataset for the ATISNLU dataset

        :dataset_name (str) : name of the dataset
        :data (dict(str:list(str|list(list)))) : the dataset's data, read into a dictionary
        :space_name (str) : the name of the vector space
        :data_params dict(str:str|list|dict) : the data parameters initialized from command line arguments
        :dataset_params dict(str:str|list|dict) : the dataset specifications
        :experiments (list(str)) : The experiments for which the datasets have to be created
        :models (list(str)) : The models for which the datasets have to be created

        :returns (bool) : True if the datasets were successfully created
                            False, if the datasets are not created

        """
        is_executed = True
        total_slots = []
        total_intents = []
        self.__temp_data = {}
        if os.path.isfile(os.path.join(data_params['result_dir'],dataset_name + "_MinMaxScaler.pkl")):
            self.__minmaxscaler_obj = self.io.read_file(path=data_params['result_dir'],
                                                        filename=dataset_name + "_MinMaxScaler.pkl",
                                                        format="pkl"
                                                        )

        # Get the complete list of slots and intents for one-hot-mapping
        for dataset_type, dataset_data in data.items():
            (sents, slots, intents) = self.__dataset_reader.get_data(data=dataset_data)
            total_slots += slots
            total_intents += intents
            self.__temp_data[dataset_type] = (sents, slots, intents)

        # Process each dataset_type and its data i.e. train, dev or test AND their respective data
        for dataset_type, (sents, slots, intents) in self.__temp_data.items():
            # Process for each model
            for each_model in models:
                for each_experiment in experiments:
                    if (each_experiment, each_model) in experiments_allowed:
                        print("Building dataset for << %s : %s >>" % (each_experiment, each_model))
                        # Get the input dataset and mapping
                        is_executed, input_matrix, input_mapping = self.__build_dataset_create_input_data(input_data=sents, input_params=dataset_params['models'][each_model])

                        if is_executed:
                            # Process for each experiment

                            if each_experiment == "int-cls":
                                output_data = intents
                                one_hot_data = total_intents
                            elif each_experiment == "slot-fill-cls":
                                output_data = slots
                                one_hot_data = total_slots

                            one_hot_mapping = None
                            # Get the output dataset, mapping and one-hot mapping
                            is_executed, output_matrix, output_mapping, one_hot_mapping = self.__build_dataset_create_output_data(output_data=output_data,
                            one_hot_data=one_hot_data, output_params=dataset_params['experiments'][each_experiment],
                            model_params=dataset_params['models'][each_model])

                            # IF THE DATASET is created successfully,
                            # STORE IT in a file
                            if is_executed:
                                files_to_save = ["vectors", "mappings", "one-hot-mappings"]
                                for each_file_type in files_to_save:
                                    file_name = "_".join([dataset_name, space_name, each_experiment, each_model, dataset_type, each_file_type + ".pkl.gz"])
                                    if each_file_type == "vectors":
                                        input_data = input_matrix
                                        output_data = output_matrix
                                    elif each_file_type == "mappings":
                                        input_data = input_mapping
                                        output_data = output_mapping
                                    elif each_file_type == "one-hot-mappings":
                                        input_data = None
                                        output_data = one_hot_mapping

                                    if not self.io.write_file(path=data_params['vector_datasets'],
                                                              filename=file_name, format="pkl.gz",
                                                              data={"input":input_data, "output":output_data}):
                                        is_executed = False
                                    else:
                                        # Create dataset statistics if a file is stored sucessfully
                                        self.__dataset_stats[file_name] = {}
                                        self.__dataset_stats[file_name]['for_experiment'] = each_experiment
                                        self.__dataset_stats[file_name]['for_model'] = each_model
                                        self.__dataset_stats[file_name]['dataset_name'] = dataset_name
                                        self.__dataset_stats[file_name]['space_name'] = space_name
                                        if each_file_type == "vectors":
                                            self.__dataset_stats[file_name]['input_data_shape'] = (len(input_data),self.__build_dataset_get_dimensions(input_data))
                                            self.__dataset_stats[file_name]['output_data_shape'] = (len(output_data),self.__build_dataset_get_dimensions(output_data))
                                        elif each_file_type == "mappings":
                                            self.__dataset_stats[file_name]['input_data_shape'] = (len(input_data),0)
                                            self.__dataset_stats[file_name]['output_data_shape'] = (len(output_data),0)
                                        elif each_file_type == "one-hot-mappings":
                                            self.__dataset_stats[file_name]['input_one-hot-mapping'] = None
                                            self.__dataset_stats[file_name]['output_one-hot-mapping'] = (len(output_data),1)

        try:
            val = self.__minmaxscaler_obj.data_max_

        except AttributeError as e:
            if not self.io.write_file(path=data_params['result_dir'],
                                      filename=dataset_name + "_MinMaxScaler.pkl",
                                      format="pkl",
                                      data=self.__minmaxscaler_obj):
                is_executed = False

        return is_executed

    def __build_dataset_get_dimensions(self, data):
        """
        Method to find out the column dimensionality of a matrix
        """
        rows = len(data)
        cols = 0
        dimensionality = None
        for i in range(len(data)):
            if i == 0:
                data_len = len(data[i])
                continue
            if len(data[i]) != data_len:
                dimensionality = "Variable"
                break
            else:
                dimensionality = len(data[i])
        return dimensionality

    def __build_dataset_create_input_data(self, input_data=None, input_params=None):
        """
        Method to create the input data for the dataset
            - Also, this method applies any transformations which the input_matrix might require on the vectors

        :input_data (list(str|list(str))) : the input data from which the vectorized dataset will be created
        :input_params (dict(str:(str|bool))) : the dataset_parameters specific to a model

        :returns (bool) : True, if the vectors were extrated into the input_matrix successfully
                            False, if the input_matrix could not be created
                - input_matrix (list(list)) : each row represents 1 input sample
                - input_mapping (dict(str:(any))) : With each row as key value, mapping of the rows to the textual data

        """
        is_executed = True
        input_matrix = []
        input_mapping = {}

        # Get all the vocab from the input data and get the vectors for the vocab
        for line_no, sentence_word_list in enumerate(input_data):
            sentence_vectors = []
            found_words = []
            for each_word in sentence_word_list:
                try:
                    idx = self.__space['rows'].index(each_word)
                    sentence_vectors.append(self.__space['cooccurrence_matrix'][idx])
                except Exception as e:
                    sentence_vectors.append([0.0] * len(self.__space['cooccurrence_matrix'][0]))
                found_words.append(each_word)

            input_mapping[line_no] = found_words

            # ADD EMBEDDINGS to the input_matrix
            if len(sentence_vectors) > 0:
                if input_params['input_vector'] == "single":
                    sentence_vector = []

                    for each_vector in sentence_vectors:
                        sentence_vector += each_vector

                    # if input_params['input_vector_transformation'] == "pca":
                    #     sentence_vector = self.__vector_ops.apply_vector_transformations(transformation_params={'pca':True}, matrix=[sentence_vector], n_components=input_params['input_vector_dim'])

                    input_matrix.append(sentence_vector)

                elif input_params['input_vector'] == "seq":
                    input_matrix.append(sentence_vectors)
                    assert len(sentence_vectors) == len(found_words)

        # PERFORM data normalization on the embeddings extracted, if required
        if input_params['input_vector'] == "single" and input_params['input_vector_transformation'] == "pca":
            max_idx = max([len(row) for row in input_matrix])
            input_matrix = [row + [0.1e-10] * (max_idx-len(row)) for row in input_matrix]
            is_executed, input_matrix = self.__vector_ops.apply_vector_transformations(transformation_params={'pca':True}, matrix=input_matrix, n_components=300)

        return is_executed, input_matrix, input_mapping

    def __build_dataset_create_output_data(self, output_data=None, one_hot_data=None,
                                           output_params=None, model_params=None):
        """
        Method to create the output data for the dataset
            - ONE-HOT-MAPPING might be created, typically used for classification tasks

        :output_data (list(str|list(str))) : the input data from which the vectorized dataset will be created
        :one_hot_data (list(str)) : list of unique vocabulary for making a one hot mapping
        :output_params (dict(str:(str|bool))) : the dataset_parameters specific to a model

        :returns (bool, output_matrix, output_mapping, one_hot_mapping) :

            - bool :  True, if the vectors were extrated into the output_matrix successfully
                        False, if the input_matrix could not be created
            - output_matrix (list(list)) : each row represents 1 output sample
            - output_mapping (dict(str:(any))) : With each row as key value, mapping of the rows to the
                                                textual data
            - one-hot-mapping (dict(str:(any))) : represents the one-hot mapping of output values (component values) to the textual data
        """
        is_executed = True
        one_hot_mapping = None
        output_matrix = []
        scaled_obj = None

        # IF output_data is dependent on one-hot-mapping
        if output_params['output_vector'] == "one-hot-vector":
            # Get mapping
            one_hot_mapping = self.__build_dataset_create_one_hot_mapping(one_hot_data)
        elif output_params['output_vector'] == 'categorical_min-max-scaled':
            one_hot_mapping = self.__build_dataset_create_categorical_min_max_scaled_mapping(one_hot_data)
            # Get data
        output_matrix, output_mapping = self.__build_dataset_create_output_matrix(output_data,
                                                                                      one_hot_mapping)

        if one_hot_mapping is None or output_matrix is None:
            is_executed = False

        return is_executed, output_matrix, output_mapping, one_hot_mapping

    def __build_dataset_create_one_hot_mapping(self, one_hot_data):
        """
        Method to create a one-hot-mapping vector so that the output_matrix can have a set of one_hot_vectors
        NOTE: one-hot-mapping should be created over the ENTIRE dataset

        """
        one_hot_mapping = None

        # Create a list of unique values from the output data
        if not isinstance(one_hot_data[0], list):
            unique_vals = sorted(set(one_hot_data))
        else:
            unique_vals = sorted(set([each_word for each_line in one_hot_data for each_word in each_line]))

        # Create a one-hot-mapping dict
        one_hot_mapping = {unique_vals[idx].strip():idx for idx in range(len(unique_vals))}

        return one_hot_mapping

    def __build_dataset_create_categorical_min_max_scaled_mapping(self, one_hot_data):
        """
        Method to create a categorical vector so that the output_matrix
        can have a set of categorical vectors for multi-lable classification.

        :one_hot_data (list(str)) : list of unique values to create scaling

        :returns dict(str:int/float), MinMaxScaler (categorical scaling object)
        """
        categorical_mapping = None

        # Create a list of unique values from the output data
        if not isinstance(one_hot_data[0], list):
            unique_vals = sorted(set(one_hot_data))
        else:
            unique_vals = sorted(set([each_word for each_line in one_hot_data for each_word in each_line]))
        unique_idx = np.matrix(sorted({unique_vals[idx].strip():idx for idx in range(len(unique_vals))}.values()))
        # Create a categorical data mapping dict
        ########################################
        # Initialize a list of random numbers
        # This list is a normal gaussian distribution of numbers

        # Scale these random numbered array to a range of [0,1]
            # Take a transpose BECAUSE the scaling happens by columns and we want to scale by rows
            # Reverse-transpose to get back the original shape (i.e. row-wise matrix)
            # Convert to a scaled list
        try:
            val = self.__minmaxscaler_obj.data_max_
            scaled_vals = self.__minmaxscaler_obj.transform(unique_idx.T).T.round(4).tolist()
        except AttributeError as e:
            scaled_vals = self.__minmaxscaler_obj.fit_transform(unique_idx.T).T.round(4).tolist()

        # Assert that all data is of the same length
        assert len(scaled_vals[0]) == len(unique_vals), "ids to be mapped and unique list should be of the same length"
        # Create the categorical mapping
        categorical_mapping = {term:value for (term, value) in zip(unique_vals,scaled_vals[0])}

        return categorical_mapping

    def __build_dataset_create_output_matrix(self, output_data, data_mapping):
        """
        Method to create an output matrix
            - each datapoint is a one-hot-vector OR a categorical vector
            - each one-hot-vector is created from the one-hot-mapping
                resp. for categorical vector --> from categorical mapping
        """
        output_matrix = None
        output_mapping = {}

        # CREATE the output_matrix from the one-hot-mapping
        if not isinstance(output_data[0], list):
            output_matrix = [[1 if each_mapped_word==each_word else 0 for each_mapped_word, idx in data_mapping.items()] for each_word in output_data]
            output_mapping = {line_no:each_word for line_no, each_word in enumerate(output_data)}
        else:
            output_matrix = [[data_mapping[each_word.strip()] for each_word in each_line] for each_line in output_data]
            output_mapping = {line_no:[each_word.strip() for each_word in each_line] for line_no, each_line in enumerate(output_data)}

        return output_matrix, output_mapping
