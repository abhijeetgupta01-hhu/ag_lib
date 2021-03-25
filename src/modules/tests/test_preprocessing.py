###############################################################################################################
#
#  IMPORT STATEMENTS
#
###############################################################################################################

# >>>>>>>>> Native Imports <<<<<<<<<<
import os
import sys


path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/..")


# >>>>>>>>> Package Imports <<<<<<<<<<


# >>>>>>>>> Local Imports <<<<<<<<<<
from preprocessing import preprocessing


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################


def test_vocab_builder():

    sents = ["A B C","B C D"]

    voc2id, analysis, counter = preprocessing.build_vocab(sents)

    assert counter == 4
    assert analysis["A"] == 1
    assert analysis["B"] == 2
    assert analysis["A"]+analysis["D"] == analysis["C"]





###############################################################################################################
#
#  MAIN
#
###############################################################################################################


if __name__=="__main__":

    sents = ["A B C","B C D"]

    voc2id, analysis, counter = preprocessing.build_vocab(sents)

    print(voc2id)
    print(analysis)
    print(counter)




#
