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
from collections import defaultdict, OrderedDict

# >>>>>>>>> Package Imports <<<<<<<<<<
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO

###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class VectorOps(object):
    """
    Class to run vector based operations
        - Can perform vector operations (transformations) on individual or a vector batch
        - Can also create, load and save spaces

    :build_space() : method to build a space from a collection of vectors and the dataset vocabulory
        - called from the vector_ops instance

    :save_space() : method to save a built/transformed space
        - called from the vector_ops instance

    :load_space() : method to load an existing space
        - called from the vector_ops instance

    :apply_space_transformations(): method to apply transformation operations on space created / loaded
        - called from the vector_ops instance

    :apply_vector_transformations_l2norm: method to apply l2 normalization operation on a vector / matrix
        - called from vector_ops instance

    :add_padding() : method to add row/column based padding for an experiment
        - called from the model instance
    """

    space_params = OrderedDict()

    def __init__(self):
        """
        Method to initialize basic class-instance variables
        """
        self.io = IO()
        self.__atisnlu_reader = AtisnluDataReader()
        self.__dataset_embeddings = {}
        self.__space = None
        return

###############################################################################################################
#
#  1. SPACE RELATED MODULES
#
###############################################################################################################

    def build_space(self, data=None, dataset_name=None, space_path=None, space_file_name=None,
                    space_name=None,space_extension=None):
        """
        Method to extract embeddings from a (downloaded) vector space

        :data (dict(str:list(str))) : contains the orignal/pre-processed dataset which has been read
        :dataset_name (str) : name of the dataset
        :space_name (str) : name of the space
        :space_extension (str) : extension of the space file

        :return bool,
        """
        is_executed = True

        # read embeddings
        try:
            embedding_data = self.io.read_file(path=space_path,
                                               filename=space_file_name,
                                               format=space_extension)
        except Exception as e:
            print("Error in reading the embedding file: %s" % os.path.join(space_path, space_file_name))
            is_executed = False

        # get vocabulory
        if is_executed:
            vocab = self.__get_vocabulory(dataset_name=dataset_name, data=data)
            if not len(vocab) > 0:
                is_executed = False

        # extract dataset embeddings
        if is_executed:
            self.__dataset_embeddings = self.__extract_embeddings(vocab=vocab, embeddings=embedding_data,
                                                                  space_name=space_name, dataset_name=dataset_name)

        # Initializing space from the extracted embeddings
        del(embedding_data)
        if len(self.__dataset_embeddings) > 0:
            self.__space = {'rows':[],'cooccurrence_matrix':[], 'transformations':[]}
            for word, embedding in self.__dataset_embeddings.items():
                self.__space['rows'].append(word)
                self.__space['cooccurrence_matrix'].append(embedding)

            del(self.__dataset_embeddings)
            self.__dataset_embeddings = {}
        else:
            is_executed = False
            raise ValueError("No embeddings extracted for this dataset << %s >> from space << %s >>" % (dataset_name, space_name))

        return is_executed

    def __get_vocabulory(self, dataset_name=None, data=None):
        """
        Method to return the vocabulory of a dataset based on the dataset name

        :data (dict(str:list(str))) : contains the orignal/pre-processed dataset which has been read
        :dataset_name (str): name of the dataset

        :return (list(str)) : extracted vocabulory
        """
        vocab = set()

        # dataset specific extraction
        if dataset_name.find("atisnlu") > -1:
            for dataset_type, dataset_data in data.items():
                (sents, slots, intents) = self.__atisnlu_reader.get_data(data=dataset_data)
                for i in range(len(sents)):
                    for j in range(len(sents[i])):
                        if not sents[i][j].isdigit():
                            vocab.add(sents[i][j].strip())

        return list(vocab)

    def __extract_embeddings(self, vocab=None, embeddings=None, space_name=None, dataset_name=None):
        """
        Method to extract the embeddings specific to a dataset vocab

        :vocab (list(str)) : vocabulory of the dataset
        :embeddings list(str) : the label and embeddings rows stored in the embedding file
        :space_name (str) : the name of the embedding file / source from which the embeddings have been extracted

        :returns dict(str:list(float)): a dictionary of vocab and its embeddings
        """
        extracted_embeddings = {}
        extracted_freq_order = []
        total_embeddings = None
        dimensions = None
        min_component_val = 0.0
        max_component_val = 0.0

        vocab = {each_word.lower().strip():0 for each_word in vocab}

        for line_no, each_line in enumerate(embeddings):

            # CONVERT to utf-8, if data is in binary format
            try:
                each_line = each_line.decode().lower().strip()
            except Exception as e:
                each_line = each_line.lower().strip()

            # Skipping the first line of the below mentioned spaces
            if line_no == 0:
                if space_name.find("crawl-300d") > -1 or space_name.find("GoogleNews2013") > -1 or space_name.find("CoNLL2017") > -1:
                    total_embeddings, dimensions = map(int, each_line.split())
                    continue
                else:
                    total_embeddings = len(embeddings)
                    dimensions = len(each_line.split()[1:])

            # Processing the current line
            parts = each_line.split()
            # The embedding line may contain multiple words (separated by spaces) followed by features
            # A split point needs to be found, so that we convert only the features to float values
            # AND consider all words of the multi-word expression as a part of 1 token
            split_at = 1
            found = False
            while not found:
                try:
                    to_float = list(map(float, parts[split_at:]))
                    found = True
                except ValueError as e:
                    split_at += 1
                    if split_at == len(parts):
                        found = True

            token = " ".join(parts[:split_at]).strip()

            # Removing the POS tag for google news space from the token name
            if space_name.find("GoogleNews2013") > -1:
                token = token.split("_")[0].strip()

            # Extracting the embeddings by vocab
            try:
                val = vocab[token]
                vocab[token] = 1
                extracted_embeddings[token] = list(map(float, parts[split_at:]))
                min_val = min(extracted_embeddings[token])
                max_val = max(extracted_embeddings[token])
                min_component_val = min_val if min_val < min_component_val else min_component_val
                max_component_val = max_val if max_val > max_component_val else max_component_val
                extracted_freq_order.append((line_no, token))
            except KeyError as e:
                # print("Not found: ", token, " -- ", str(e))
                continue

        # update space parameters - statistics for the data being read
        self.space_params[dataset_name] = {space_name:{"original_vocab":total_embeddings,
                                                       "embedding_dimensions":dimensions,
                                                       "min_max_component_val":(min_component_val, max_component_val),
                                                       "dataset_vocab":len(vocab),
                                                       "vocab_found":sum([1 for each_word, flag in vocab.items() if flag==1]),
                                                       "vocab_skipped_count":sum([1 for each_word, flag in vocab.items() if flag==0]),
                                                       "vocab_skipped":[each_word for each_word, flag in vocab.items() if flag==0],
                                                       "vocab_found_inDecreasing_Frequency":extracted_freq_order,
                                                       }
                                           }

        # print(self.space_params)
        # input("vectors .. space params")
        return extracted_embeddings

    def save_space(self, space_path=None, space_name=None):
        """
        Method to save a space created by the self.build_space() method OR loaded by using the self.load_space() method.
        - IF a previous space is found with the same name then:
            - that previous space will be loaded
            - ALL the missing rows/embeddings NOT in self.__space will be added
            - transformations of previous space and current space will be re-applied

        :space_path (str) : the path on which the space will be saved
        :space_name (str) : the name of the space (without the extension)

        :returns (bool) : True, if the space is saved
                            False, if the space is not saved
        """
        is_executed = True
        space_name += "_space.pkl.gz"

        # Check if there is no space already loaded in the same class instance
        if self.__space is not None:
            # Check if there is a previous space to which the new data can be added
            if os.path.isfile(os.path.join(space_path, space_name)):
                prev_space = self.io.read_file(path=space_path, filename=space_name, format="pkl.gz")
                if prev_space is not None:
                    # IF YES, then add the data and apply prev. space's transformations
                    is_executed = self.__update_space_with_previous_space(prev_space=prev_space)
            # Save space
            if is_executed and not self.io.write_file(path=space_path, filename=space_name, format="pkl.gz", data=self.__space):
                is_executed = False
        # Nothing to write
        else:
            is_executed = False
            print("There seems to be no space loaded / created for this class instance")

        return is_executed

    def __update_space_with_previous_space(self, prev_space=None):
        """
        Method to update a newly built / loaded space with a previous space.
        - The method updates the new space with the rows of the previously existing space
        - Applies space transformations of the previous space to the new space

        #TODO: Check if the new space has transformations that can be applied to the previous space as well

        :prev_space (dict(str:list(str|float))): rows and cooccurrence matrix of the previous space

        :returns (bool): True, if the new space is updated with the existing space
                            False, if the update failed
        """
        is_executed = True

        # Updating the new space with previous space
        space_vocab = {each_word:0 for each_word in self.__space['rows']}

        for i in range(len(prev_space['rows'])):
            try:
                val = space_vocab[prev_space['rows'][i]]
            except KeyError as e:
                self.__space['rows'].append(prev_space['rows'][i])
                self.__space['cooccurrence_matrix'].append(prev_space['cooccurrence_matrix'][i])

        # Figuring out the transformations to apply
        new_space_transformations = {transformation:True for transformation in prev_space['transformations'] if transformation not in self.__space['transformations']}

        if not self.apply_space_transformations(transformation_params=new_space_transformations):
            is_executed = False

        return is_executed

    def load_space(self, space_path=None, space_name=None):
        """
        Method to load an existing space created from a dataset vocabulory
            - The space can be updated over multiple datasets over time
            - Just load the space, build_space( on new dataset) AND save space

        :space_path (str) : path of the space to load
        :space_name (str) : name of the space to search and find the corresponding file

        :returns (bool) : True, if the existing space was loaded successfully
                            False, if the space could not be loaded
        """
        space_file = None
        self.__space = None
        is_executed = True

        # Search for the file
        files = self.io.search_files(path=space_path, search_string=space_name+"_space", match_type="partial")

        # Make sure that there is exactly one file for the dataset and space
        try:
            assert len(files) == 1
            space_file = files[0]
        except Exception:
            is_executed = False
            print("There should be exactly 1 file for the space. FOUND (%d):\n\t" % len(files), files)

        # Make sure that the file name is a valid file string and file
        if len(space_file) < 0 or not isinstance(space_file, str):
            print("Does not seem to be a valid string for a file << %s >>" % space_file)
            is_executed = False

        if not os.path.isfile(space_file):
            print("Does not seem to be a valid file << %s >>" % space_file)
            is_executed = False

        # Load space
        if is_executed:
            self.__space = self.io.read_file(path=os.path.dirname(space_file),
                                            filename=os.path.basename(space_file),
                                            format="pkl")
            if self.__space is None:
                is_executed = False
                print("Could not read the space file << %s >>" % space_file)

        return is_executed

###############################################################################################################
#
#  3. SPACE TRANSFORMATION MODULES
#
###############################################################################################################

    def apply_space_transformations(self, transformation_params=None):
        """
        Method to apply space transformation operations to a build/loaded space

        :transformation_params (dict(str:bool)) : a dictionary of parameters with boolean values to execute OR not execute a transformation

        :returns (bool): True, if the transformation could be executed
                            False, if the transformation could not be executed
        """
        is_executed = True
        if self.__space is None:
            is_executed = False
            print("Cannot read a NoneType space instance. Load / Create a space first")

        if is_executed:
            for transformation, bool_value in transformation_params.items():
                if bool_value:

                    if transformation == "l2norm":
                        is_executed, self.__space['cooccurrence_matrix'] = self.__apply_vector_transformations_l2norm(matrix=self.__space['cooccurrence_matrix'])
                        if is_executed and not transformation in self.__space['transformations']:
                            self.__space['transformations'].append(transformation)

                    if transformation == "pca":
                        is_executed, self.__space['cooccurrence_matrix'] = self.__apply_vector_transformations_pca(matrix=self.__space['cooccurrence_matrix'])
                        if is_executed and not transformation in self.__space['transformations']:
                            self.__space['transformations'].append(transformation)

                    if not is_executed:
                        print("Could not apply space transformations on the parameter << %s >>" % transformation)
        return is_executed

    def apply_vector_transformations(self, transformation_params=None, matrix=None, n_components=None):
        """
        Method to apply vector transformations to a 2D vector matrix.
        Based on the transformation given in the transformation parameters, a relevant transformation function is called
            - For transforming a single vector -> pass it within a list

        :transformation_params (dict(str:bool)) : a dict of transformations to be performed on the dataset
        :matrix (list(list)): a 2D matrix "not np.ndarray" which contains the vectors
        :n_components (int): a parameter used by many of the space reduction functions which will restrict the size of a transformed vector to the value of n_components

        :returns (bool, matrix):
            - bool : True, if the transformation was successful;
                        False, if the transformation did not happen
            - matrix : Transformed vectors
        """
        is_executed = True

        if not isinstance(matrix, np.matrix):
            matrix = np.matrix(matrix)

        # check if the passed object is a vector, IF YES, then make it into a matrix
        orig_dims = np.ndim(matrix)

        if orig_dims == 1:
            matrix = [matrix]
        elif orig_dims == 2:
            pass
        else:
            orig_dims = 0
            is_executed = False

        if is_executed:
            for transformation, bool_value in transformation_params.items():
                if bool_value:
                    if transformation == "l2normalize":
                        is_executed, matrix = self.__apply_vector_transformations_l2norm(matrix=matrix)
                    if transformation == "pca":
                        is_executed, matrix = self.__apply_vector_transformations_pca(matrix=matrix,
                                                                                      n_components=n_components)

                    if not is_executed:
                        print("Could not apply space transformations on the parameter << %s >>" % transformation)

        # Convert back to 1 vector, IF 1 vector was passed
        if is_executed:
            matrix = matrix.tolist()
            if orig_dims == 1:
                matrix = matrix[0]

        return is_executed, matrix

    def __apply_vector_transformations_l2norm(self, matrix=None):
        """
        Method to apply an l2 normalization to a vector or matrix
            - the l2 normalization is applied to each element (row) individually (axis=1)
            - to apply l2 normalization on the entire matrix / set of vectors ==> (axis=0)

        :matrix (list(list)) : a 2 dimensional matrix

        :returns (bool, l2normalized_matrix) :
            - bool: True, False depending on the success of transformation
            - l2 normalized matrix
        """
        is_executed = True


        # Compute the l2 normalization (by row; each row individually)
        if is_executed:
            try:
                matrix = normalize(matrix, norm="l2", axis=1)
                assert int(np.sum(matrix ** 2, axis=1).sum()) == len(matrix)
            except Exception as e:
                is_executed = False
                print(int(np.sum(matrix ** 2, axis=1).sum()))
                print(len(matrix))
                print("Could not l2 normalize the matrix")

        return is_executed, matrix

    def __apply_vector_transformations_pca(self, matrix=None, n_components=None):
        """
        Method to apply PCA to reduce dimensionality of a matrix

        :matrix (list(list)) : a 2 dimensional matrix
        :n_components : number of components to keep in the reduced dimensionality

        :returns (bool, pca_matrix) :
            - bool: True, False depending on the success of transformation
            - pca reduced matrix
        """
        is_executed = True

        # Compute the pca reduction
        if is_executed:
            pca_obj = PCA(n_components=n_components)
            try:
                matrix = pca_obj.fit_transform(matrix)
            except Exception as e:
                is_executed = False
                print("Could not execute PCA: %s" % str(e))

        return is_executed, matrix

###############################################################################################################
#
#  3. ADDITIONAL VECTOR OPERATION MODULES
#
###############################################################################################################

    def add_padding(self, data=None, pad_rows=None, pad_cols=None, pad_value=0.0,
                    max_row_len=None, max_col_len=None):
        """
        Method to add padding to the data

        :data (list(list(list(float)))) : The input vectors given to create a mask for data
                # [sentences[sentence[word_vectors(float)]]]

        :data (list(list(float))) : The output vectors given to create a mask
                # [one_hot_vector(float)]

        :returns (bool, data):
            - bool : True if the padding was added; else False
            - data : The 2D OR 3D matrix -- with added padding

        """
        is_executed = True

        # If ROWS have to be padded in a matrix
            # IF data to be padded is a matrix, both MAX ROWS and COLS have to be computed
            # Generally applicable to input matrices
        if pad_rows:
            if isinstance(data[0][0], list):
                # Find rows
                if max_row_len is None:
                    max_rows = max([len(each_block) for each_block in data])
                else:
                    max_rows = max_row_len
                # Find cols
                if max_col_len is None:
                    max_columns = max([len(each_row) for each_block in data for each_row in each_block])
                else:
                    max_columns = max_col_len

                # FOR 2D Padding
                data = [each_block + [[pad_value] * max_columns] * (max_rows - len(each_block)) if len(each_block) < max_rows else each_block for each_block in data]

                is_executed = all([True if len(each_block) == max_rows else False for each_block in data])

        # IF ONLY COLS have to be padded
            # In data to be padded are rows individual rows
                # only MAX cols are applicable
            # Generally applicable for output matrices
        if pad_cols:
            if isinstance(data[0], list):

                if max_row_len is None:
                    max_rows = len(data)
                else:
                    max_rows = max_row_len

                if max_col_len is None:
                    max_columns = max([len(each_block) for each_block in data])
                else:
                    max_columns = max_col_len

                # FOR A 2D padding
                # data = [[each_row + [pad_value] * (max_columns - len(each_row))] * max_columns if len(each_row) < max_columns else [each_row] * max_columns for each_row in data]

                # FOR A 1D padding
                data = [each_row + [0.0] * (max_columns - len(each_row)) if len(each_row) < max_columns else each_row for each_row in data]

                is_executed = all([True if len(each_row) == max_columns else False for each_row in data])

        return is_executed, data
