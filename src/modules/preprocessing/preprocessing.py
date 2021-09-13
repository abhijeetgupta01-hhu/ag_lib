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
from collections import defaultdict

# >>>>>>>>> Package Imports <<<<<<<<<<


# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO
# Test
# Test 1
###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class DatasetPreprocessor(object):
    """
    Class to pre-processs data

    :run() : The main function to invoke the data pre-processor from an external module


    """

    def __init__(self):
        """
        Method to initialize basic class-instance variables
        """
        self.io = IO()
        self.__dataset_reader = None
        self.__dataset_name = None

        return

    def run(self, data=None, dataset_type=None, pre_processing_params=None, data_params=None):
        """
        Main method to execute the data pre-processor for a dataset

        The module accepts the data read from the dataset file and carries out specific pre-processing operations according to the dataset

        :data (list(str)) : The dataset, as read from the file
        :pre_processing_params (dict(str:bool)) : A specifications of which parameters have to be executed for the dataset
        :data_params: The data parameters of the program, as read from the command line arguments

        :returns (bool, list(str)):
            bool: True, False - depending on if the pre-processing could be carried out successfully
            list(str) : The pre-processed data
        """

        is_executed = True
        self.__dataset_name = data_params['dataset_name']
        if self.__dataset_name.find("atisnlu") > -1:
            self.__dataset_reader = AtisnluDataReader()

        pre_processed_data = data

        for each_parameter, bool_value in pre_processing_params.items():

            if bool_value:
                if each_parameter == "convert_to_lower_case":
                    is_executed, returned_data = self.__convert_to_lower_case(pre_processed_data)

                elif each_parameter == "convert_to_utf8":
                    is_executed, returned_data = self.__convert_to_utf8(pre_processed_data)

                elif each_parameter == "check_text_vs_slot_data":
                    is_executed, returned_data = self.__check_text_vs_slot_data(pre_processed_data)

                elif each_parameter == "remove_apostrophe":
                    is_executed, returned_data = self.__remove_apostrophe(pre_processed_data)

                elif each_parameter == "generate_statistics":
                    is_executed = self.__generate_statistics(data=pre_processed_data,
                                                             dataset_type=dataset_type,
                                                             data_params=data_params)

                if is_executed:
                    pre_processed_data = returned_data
                else:
                    print("Error in data pre-processing parameter << %s >>" % each_parameter)

        return is_executed, pre_processed_data

###############################################################################################################
#
#  1. Data Pre-processing functions
#
###############################################################################################################

    def __convert_to_lower_case(self, input_data):
        """
        Method to convert the data into lowercase data

        :input_data : data coming from the previous stage of pre-processing OR intial dataset

        :returns bool, list(str) : pre-processed data of the function
        """
        return True, [each_line.lower() for each_line in input_data]

    def __convert_to_utf8(self, input_data):
        """
        Method to convert the data into utf8 data

        :input_data : data coming from the previous stage of pre-processing OR intial dataset

        :returns bool, list(str) : pre-processed data of the function
        """

        pre_processed_data = []

        for each_line in input_data:
            try:
                each_line.decode('utf-8')
            except AttributeError as e:
                each_line = each_line.encode('utf-8')
            pre_processed_data.append(each_line)

        return True, pre_processed_data

    def __check_text_vs_slot_data(self, input_data):
        """
        Method to convert the data into lowercase data

        :input_data : data coming from the previous stage of pre-processing OR intial dataset

        :returns bool, list(str) : pre-processed data of the function
        """

        is_executed = True
        pre_processed_data = []
        skipped_count = 0

        if self.__dataset_name.find("atisnlu") > -1:
            (sents, slots, intents) = self.__dataset_reader.get_data(data=input_data)
            for i in range(len(sents)):
                try:
                    assert len(sents[i]) == len(slots[i])
                    assert len(intents[i]) > 0
                    pre_processed_data.append(input_data[i].strip())
                except Exception as e:
                    print("Exception is words vs slot (%d vs %d) count @ line no: %d" % (len(sents[i]), len(slots[i]), i))
                    skipped_count += 1

            if skipped_count > 0:
                print("Skipped %d rows of the dataset due to sent vs. slot mismatch" % skipped_count)
                # input("Continue...")
        return is_executed, pre_processed_data

    def __remove_apostrophe(self, input_data):
        """
        Method to remove speific apostrophe-ied words from the corpora

        :input_data : data coming from the previous stage of pre-processing OR intial dataset

        :returns bool, list(str) : pre-processed data of the function
        """

        is_executed = True
        pre_processed_data = []
        skipped_count = 0

        if self.__dataset_name.find("atisnlu") > -1:
            for each_line in input_data:
                sent, rest = each_line.strip().split("\t")
                sent = sent.strip().replace("'s","").replace("'re","").replace("'ve","").replace("'ll","")
                pre_processed_data.append("\t".join([sent, rest]))

        return is_executed, pre_processed_data

    def __generate_statistics(self, data=None, dataset_type=None, data_params=None):
        """
        Method to generate statistics over a dataset under consideration

        :data (list(str)) : dataset
        :dataset_type (str) : type of dataset (train, dev, test) - currently under processing
        :data_params (dict(str:str|list)) : the data parameters read from the command line arguments

        :return bool

        :OUTPUT: a statistics file based on the dataset being processed
        """
        is_executed = True

        if self.__dataset_name.find("atisnlu") > -1:

            agg_words = defaultdict(int)
            agg_slots = defaultdict(int)
            agg_intents = defaultdict(int)
            sent_len = defaultdict(int)

            (sents, slots, intents) = self.__dataset_reader.get_data(data=data)
            for i in range(len(sents)):
                sent_len[i] = len(sents[i])
                for j in range(len(sents[i])):
                    agg_words[sents[i][j]] += 1
                    if slots[i][j].strip() != "O":
                        agg_slots[slots[i][j]] += 1
                agg_intents[intents[i]] += 1

            stat_data = {}
            stat_data[self.__dataset_name] = {dataset_type:
                                       {'Total Sentences':len(sents),
                                        'Max_Sentence_Length':max([sent_len[i] for i in sent_len]),
                                        'Total Words':sum([count for word, count in agg_words.items()]),
                                        'Unique Words':len(agg_words),
                                        'Total Slots':sum([count for slot, count in agg_slots.items()]),
                                        'Unique Slots':len(agg_slots),
                                        'Total Intents':sum([count for intent, count in agg_intents.items()]),
                                        'Unique Intents':len(agg_intents),
                                        }
                                       }

        # Saving some statistics about the dataset
        result_dir = data_params['result_statistics_dir']
        stat_file_name = self.__dataset_name + "_statistics.json"

        if os.path.exists(os.path.join(result_dir,stat_file_name)):
            prev_stat_data = self.io.read_file(path=result_dir, filename=stat_file_name, format="json")
            prev_stat_data[self.__dataset_name].update(stat_data[self.__dataset_name])
            if not self.io.write_file(path=result_dir, filename=stat_file_name, format="json", data=prev_stat_data):
                is_executed = False
        else:
            if not self.io.write_file(path=result_dir, filename=stat_file_name, format="json", data=stat_data):
                is_executed = False

        return is_executed

###############################################################################################################
#
#  Misc functions
#
###############################################################################################################

    def build_vocab(sents):
        """
        Builds a vocab2ID dict from the sentences - does not remove punctiation or anythingself.

        :sents (list[str]) : The list of full sentences

        :returns (dict, dict, int) : returns the word2id dict, the count per word dict, total count of "unique" words
        """

        vocab = {}
        analysis = {}
        counter = 0

        for sent in sents:
            words = sent.split()

            for word in words:
                if not vocab.get(word):
                    vocab[word] = counter
                    counter += 1

                if analysis.get(word):
                    analysis[word] += 1
                else:
                    analysis[word] = 1


        return vocab, analysis, counter








#
