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
import multiprocessing
from multiprocessing import Process, Pool

# >>>>>>>>> Package Imports <<<<<<<<<<


# >>>>>>>>> Local Imports <<<<<<<<<<


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class Multiprocessor():
    """
    CLASS to invoke the multiprocessing environment for a particular module
    """

    def __init__(self):

        self.__process_object = None
        self.__pool_object = None
        self.__cpu_count = multiprocessing.cpu_count()
        self.__return_values = []
        return

    def cpu_count(self):
        """
        Method to return the number of cpus on the execution machine
        """
        return self.__cpu_count

    def run_process(self, function=None, arguments=None):
        """
        Execute each argument as a separate process explicitely.
        Gives more control over initializing the multiprocessor units
        """
        assert(isinstance(arguments, list))
        for eachArgument in arguments:
            if not isinstance(eachArgument, tuple):
                eachArgument = tuple(eachArgument)
            self.__process_object = Process(target=function, args=eachArgument)
            self.__process_object.start()
            self.__process_object.join()
        return

    def run_pool(self, function=None, arguments=None, ret_val=False):
        """
        Let the Multiprocessing library manage all internal multiprocessing initialization tasks

        :function (function) : The link to a function which has to be run in the multiprocessing environment
        :arguments (list(tuple)) : A list of arguments that the function will take as input
        :ret_val (bool) : Flag for the function to either return the results OR not

        :return (results || NULL) : if ret_val is True, the return values of
                                    the 'function' are returned back to the calling program
                                     
        """
        self.__return_values = []
        self.__pool_object = Pool(self.__cpu_count)
        assert(isinstance(arguments, list))
        if ret_val is False:
            self.__pool_object.starmap(function, arguments)
            return
        else:
            self.__return_values.append(self.__pool_object.starmap(function, arguments))
            return self.__return_values[0]
