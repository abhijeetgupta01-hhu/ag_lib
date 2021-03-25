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
import sklearn.metrics as eval

# >>>>>>>>> Local Imports <<<<<<<<<<
from modules.data_ops.io import IO


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################

class PredictionEvaluator(object):
    """
    Class to compute various evaluations to be executed on various test predictions
    - The evaluations can be specified through command line parameter strings
    - Ideally the functions can be mapped to a NamedTuple so that the functions get exectued automatically based on the 'string' command line parameter

    :evaluate() : The main evaluation function called from an external class which checks the basic input parameters given for evaluation and invokes a specific evaluation function

    :accuracy_one_hot(): Compute the accuracy of predicitons vs observed data
    :fscore(): Computes the Fscores
    :precision(): Compute the precision values (TP and FP)
    :recall(): Computes the recall values (TP and FN)

    """

    def __init__(self):
        """
        Method to initialize basic class instance variables
        """
        self.__execute = {'accuracy-one-hot':self.__accuracy_one_hot,
                          'fscore':self.__compute_fscore,
                          'precision':self.__compute_precision,
                          'recall':self.__compute_recall,
                          }
        return

###############################################################################################################
#
#  3. EXECUTING MODELLING SPECIFIC ACTIONS
#
###############################################################################################################

    def evaluate(self, **kwargs):
        """
        Method to invoke a specific evaluation from a calling class
            - Also checks all the parameters passed as input before executing an evaluation

        :kwargs (dict(str:(any))) : A dict of parameters which contains the
            'metric' - evaluation action
            'predictions' - predicted data
            'observed' - observed data (dev/test) data
            'mapping' - the vector mappings / labels of the prediction and observed dimensions
            (AND OTHER ON-DEMAND parameters)

        :returns (bool, results):
            - bool : True, if the evaluation could be done; else False
            - results : The evaluation results (in the order of prediction & observed data)

        """
        is_executed = True
        results = None

        # Check if a metric is supplied
        if not 'metric' in kwargs or not kwargs["metric"] in self.__execute:
            print("<< %s >> is not an evaluation action in the Evaluation class instance" % kwargs["metric"])
            is_executed = False

        # See of the predicitons and observed data is of equal size (superficially)
        try:
            metric = kwargs["metric"]
            assert len(kwargs['predictions']) == len(kwargs['observed']), "The predictions and observed matrices are of different sizes"
        except Exception as e:
            print("Error: << %s >>" % str(e))
            is_executed = False

        # Call the evaluation function
        if is_executed:
            is_executed, results = self.__execute[kwargs["metric"]](**kwargs)

        return is_executed, results

###############################################################################################################
#
#  3. EXECUTING MODELLING SPECIFIC ACTIONS
#
###############################################################################################################
    def __accuracy_one_hot(self, **kwargs):
        """
        Method to compute the accuracy of ONE-HOT vectors from the prediction and observed data

        :kwargs (same as above)

        :returns (bool, results)
        """
        is_executed = True
        results = None

        predictions = kwargs['predictions']
        observed = kwargs['observed']

        # Extract the indices with the maximum values from each row
        # In case of one hot vectors, the model would be performing a binary classification
            # With 1 being the correct class; the rest will be 0
        pred_idx = np.argmax(predictions, axis=1)
        observed_idx = np.argmax(observed, axis=1)

        # Compute the accuracy
        # If the indexes are the same, then its a correct prediction
        # Else a wrong prediciton has happened
        try:
            results = float(sum([1 if pred_idx[i] == observed_idx[i] else 0 for i in range(len(pred_idx))]))/len(pred_idx)
        except Exception as e:
            print("Error (accuracy_one_hot) : << %s >>" % str(e))
            is_executed = False

        return is_executed, results

###############################################################################################################
#
#  3. EXECUTING MODELLING SPECIFIC ACTIONS
#
###############################################################################################################

    def __compute_fscore(self, **kwargs):
        """
        Method to compute fscores from the predictions vs. the given observations

        :kwargs (same as above)

        :returns (bool, results)
        """
        is_executed = True
        results = None

        # DATASET specific data compilation
        # Data compilation:
            # 1. Extracting the mappings
            # 2. IDENTIFING the POSITIVE and NEGATIVE observed classes from the observed data by using mappings
            # 3. Creating a separate list of INDICES for POSITIVE and NEGATIVE examples from the observed data
            # 4. COMPUTE TP, FN ; TN, FP
        print("Compiling evaluation data")
        is_executed, kwargs = self.__precompute_metric_data(**kwargs)

        # Compute the True Positive and False Negatives
        # Compute the precision
        if is_executed:
            print("Computing precision")
            is_executed, precision = self.__compute_precision(**kwargs)

        # Compute the True Negatives and False Positives
        # Compute the recall
        if is_executed:
            print("Computing Recall")
            is_executed, recall = self.__compute_recall(**kwargs)

        # Compute the FSCORE
        if is_executed:
            print("Computing Fscore")
            try:
                fscore = float(2 * precision * recall) / (precision + recall)

                results = {}
                results['fscore'] = fscore
                results['precision'] = precision
                results['recall'] = recall
            except Exception as e:
                print("Error (compute_fscore) : << %s >>" % str(e))

        return is_executed, results

    def __compute_precision(self, **kwargs):
        """
        Method to compute the precision on prediction and observed data

        :kwargs (same as above)

        :returns (bool, results)

        """
        is_executed = True
        precision = None

        # Check if TP and FP are computed
        if ('tp' and 'fp') not in kwargs:
            is_executed = False
            print("Error (compute_precision): 'tp' or 'fp' not found")

        if is_executed:

            tp = kwargs['tp']
            fp = kwargs['fp']

            # Compute precision
            try:
                precision = float(tp)/(tp+fp)
            except Exception as e:
                print("Error (evaluation-fscore-precision) << %s >>" % str(e))
                is_executed = False

        return is_executed, precision

    def __compute_recall(self, **kwargs):
        """
        Method to compute the recall on prediction and observed data

        :kwargs (same as above)

        :returns (bool, results)

        """
        is_executed = True
        recall = None

        # Validate if TP and FN have been computed
        if ('tp' and 'fn') not in kwargs:
            is_executed = False
            print("Error (compute_recall): 'tp' or 'fn' not found")

        if is_executed:

            tp = kwargs['tp']
            fn = kwargs['fn']

            # Compute Recall
            try:
                recall = float(tp)/(tp+fn)
            except Exception as e:
                print("Error (evaluation-fscore-recall) << %s >>" % str(e))
                is_executed = False

        return is_executed, recall

###############################################################################################################
#
#  3. EXECUTING MODELLING SPECIFIC ACTIONS
#
###############################################################################################################
    def __precompute_metric_data(self, **kwargs):
        """
        Method to precompute some DATASET SPECIFIC data for the evaluation functions
            # Dataset specific compilation:
                # 1. Extracting the mappings
                # 2. IDENTIFING the POSITIVE and NEGATIVE observed classes from the observed data by using mappings
            # 3. Creating a separate list of INDICES for POSITIVE and NEGATIVE examples from the observed data
            # 4. COMPUTE TP, FN ; TN, FP

        :kwargs : same as above

        :returns (bool, results) : same as above
        """
        is_executed = True

        if 'dataset_name' in kwargs and kwargs['dataset_name'].find("atisnlu") > -1:
            is_executed, kwargs = self.__atisnlu_compile_data(**kwargs)
            if is_executed:
                print("Computing Positive/Negative features")
                is_executed, kwargs = self.__atisnlu_extract_positive_negative_feature_map(**kwargs)

        # compute the necessary data from the positives and negatives
        # Requires row-wise POSITIVE and NEGATIVE indices PER sample in a matrix
        if kwargs['metric'] == "fscore":
            if is_executed:
                print("Computing TP/FN")
                is_executed, kwargs = self.__compute_tp_fn(**kwargs)
            if is_executed:
                print("Computing TN/FP")
                is_executed, kwargs = self.__compute_tn_fp(**kwargs)

        return is_executed, kwargs

    def __compute_tp_fn(self, **kwargs):
        """
        Method to compute True positive and False negatives from the positive data

        :kwargs (same as above)

        :returns (bool, results)
        """
        is_executed = True

        pos_idx = kwargs['pos_feature_idx']
        predictions = kwargs['predictions']
        observed = kwargs['observed']

        tp = []
        fn = []

        # Compute the TP and FN from the positive data
        for i in range(len(pos_idx)):
            pos_indices = pos_idx[i]
            pred_row = np.asarray(predictions[i])
            ob_row = np.asarray(observed[i])

            tp += [1 if pred_val == ob_val else 0 for (pred_val, ob_val) in zip(pred_row[pos_indices],
                                                                               ob_row[pos_indices])]
            fn += [1 if pred_val != ob_val else 0 for (pred_val, ob_val) in zip(pred_row[pos_indices],
                                                                               ob_row[pos_indices])]

        kwargs['tp'] = sum(tp)
        kwargs['fn'] = sum(fn)

        return is_executed, kwargs

    def __compute_tn_fp(self, **kwargs):
        """
        Method to compute True negatives and False postives from the negative data

        :kwargs (same as above)

        :returns (bool, results) : same as above
        """
        is_executed = True

        neg_idx = kwargs['neg_feature_idx']
        predictions = kwargs['predictions']
        observed = kwargs['observed']

        tn = []
        fp = []

        # Compute the TN and FP from the negative data
        for i in range(len(neg_idx)):
            neg_indices = neg_idx[i]
            pred_row = np.asarray(predictions[i])
            ob_row = np.asarray(observed[i])

            tn += [1 if pred_val == ob_val else 0 for (pred_val, ob_val) in zip(pred_row[neg_indices],
                                                                               ob_row[neg_indices])]
            fp += [1 if pred_val != ob_val else 0 for (pred_val, ob_val) in zip(pred_row[neg_indices],
                                                                               ob_row[neg_indices])]

        kwargs['tn'] = sum(tn)
        kwargs['fp'] = sum(fp)

        return is_executed, kwargs

###############################################################################################################
#
#  3. EXECUTING MODELLING SPECIFIC ACTIONS
#
###############################################################################################################

    def __atisnlu_compile_data(self, **kwargs):
        """
        Method to extract compile the datapoints for computation of evaluation metrics
        for the ATISNLU DATASET

        :kwargs : (same as above)

        :returns (bool, kwargs): (same as above)
        """
        is_executed = True

        # Validate the parameters require to process the data
        if ('predictions' and 'observed' and 'mapping' and 'masking' and 'scaler_obj') not in kwargs:
            is_executed = False

        if is_executed:
            predictions = kwargs['predictions']
            observed = kwargs['observed']
            mapping = kwargs['mapping']
            masking = kwargs['masking']
            scaler_obj = kwargs['scaler_obj']

            predictions = np.asarray([np.asarray(each_row).flatten() if isinstance(each_row, list) else each_row for each_row in predictions]).tolist()

            # If the data has padding
            if masking:
                # Remove the padding
                predictions = self.__atisnlu_remove_prediction_masks(predictions=predictions,
                                                             observed=observed)

            # Re-scale the vectors back to their original values
            # The re-scaling model is the same as the model to scale
            predictions = self.__atisnlu_inverse_transform(matrix=predictions, scaler_obj=scaler_obj)
            observed = self.__atisnlu_inverse_transform(matrix=observed, scaler_obj=scaler_obj)
            # Convert the mapping from a dict to a row based matrix for easy access
            mapping = [mapping[i] for i in range(len(mapping))]

            # Convert all values to "int"
            predictions = [list(map(int,each_row)) for each_row in predictions]
            observed = [list(map(int,each_row)) for each_row in observed]

            kwargs['predictions'] = predictions
            kwargs['observed'] = observed
            kwargs['mapping'] = mapping

        return is_executed, kwargs

    def __atisnlu_remove_prediction_masks(self, predictions=None, observed=None):
        """
        Method to remove the masks from the prediction data
        """
        return [predictions[i][:len(observed[i])] for i in range(len(observed))]

    def __atisnlu_inverse_transform(self, matrix=None, scaler_obj=None):
        """
        Method to inverse transform the data to its original values
        """
        return [scaler_obj.inverse_transform(np.asarray([each_row]).T).T.round(4).tolist()[0] for each_row in matrix]

    def __atisnlu_extract_positive_negative_feature_map(self, **kwargs):
        """
        Method to extract to positive and negative datapoints from the predictions and observed data
        based on the mapping information
        """

        is_executed = True

        predictions = kwargs['predictions']
        observed = kwargs['observed']
        mapping = kwargs['mapping']

        pos = []
        neg = []
        for each_row_map in mapping:
            pos.append([i for i in range(len(each_row_map)) if each_row_map[i] != "O" ])
            neg.append([i for i in range(len(each_row_map)) if each_row_map[i] == "O" ])

        kwargs["pos_feature_idx"] = pos
        kwargs["neg_feature_idx"] = neg

        return is_executed, kwargs
