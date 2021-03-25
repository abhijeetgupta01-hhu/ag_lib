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
from collections import Mapping


# >>>>>>>>> Package Imports <<<<<<<<<<


# >>>>>>>>> Local Imports <<<<<<<<<<


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################
class PythonOps(object):
    """
    Class to perform various data manipulations in python objects

    :dict_merge() : Given an old and a new dictionary representing data from the same process; it recursively updates the old dictionary with new values

    """

    def dict_merge(self, prev=None, new=None):
        """
        Method to combine various evaluations perrformed on a dataset
        Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.

        :prev_dict : previous evaluation dict
        :eval_dict : current evaluation dict

        # FORMAT: evaluation_dict[each_action][experiment][model] = [()]

        :returns (dict) : updated evaluation dict
        """

        for k, v in new.items():
            if (k in prev and isinstance(prev[k], dict)
                    and isinstance(new[k], Mapping)):
                self.dict_merge(prev[k], new[k])
            else:
                prev[k] = new[k]

        return prev
