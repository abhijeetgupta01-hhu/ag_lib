#######################################

DESCRIPTION:

Currently, this module is meant to be used as a stand-alone library, which can be easily used to BUILD deep-learning models 

This code can be customized very easily to work with any dataset, vector space, ML tasks and evaluation.
The code is modular and OOPS oriented and can be scaled to add new classes, features and parameters easily.

There are 2 broad divisions at the top level that this module expects:

  # EXTERNAL DATA INPUT FOLDER  
  # which contains -
      1. Datasets
      2. Vector spaces created from the dataset, if required
      3. Vectorized datasets (created from the text datasets)
      4. Results - an all purpose folder to store statistics, analysis, data models and other intermediate outputs
      5. Statistics

  # INTERNAL SOURCE CODE CONTAINED WITHIN THIS REPOSITORY
	NOTE: The src structure can be created into a package and used further. 
	
	It contains -
        1. data_loaders: a set of scripts that deal with data processing
        2. preprocessing: a set of scripts that deal with pre-processing
        3. vector_ops: a set of scripts for vector operations and dataset creation for data modelling
        4. models: a set of scripts which contains the ML models
        5. tests: scripts which should contain the unit tests for code and data integrity
	
########################################

TO BEGIN:

A. Installing external dependencies:

# IF REQUIRED AND EXISTS: 
	Install the 'requirements.txt' in a VIRTUAL ENVIRONMENT

# Change the current working directory to the "root" folder of this repository and run:

      python3 setup.py develop

      - To use the libraries on a CPU (if there is no GPU), install a tensorflow binary which will allow the libraries to read data in a OS specific format
      (Note that this binary may make your tensorflow version revert back to 1.12.0.
      Get the latest binary for the latest version of tensorflow!)

      MacOS:
      pip3 install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.12.0-macOS-mojave-ubuntu16.04-py2-py3/tensorflow-1.12.0-cp37-cp37m-macosx_10_13_x86_64.whl

      Other OS:
      https://github.com/lakshayg/tensorflow-build

B. To run the code:
Change the current working directory to atisnlu/src/modules and run:
      python3 run_program.py <parameters>
      OR
      python3 run_program.py --help to see the parameters to be used.


#### EXAMPLES ######

1. Running the code to create vectorized datasets from a given dataset:

      python3 run_proram.py -exp <exp1> -d <dataset_path> -dt train dev -da read preprocess build_dataset -m rnn

      Here, (all arguments to the parameters are separated by space)
      -exp : specifies the experiment name EG: int-cls or slot-fill-cls (for intent classification or slot filling classification, respectively) 

      -d : location of the dataset for which learning models have to be created

      -dt : the type of data within this dataset

      -da : actions to be performed on this dataset
            - read
            - preprocess
            - extract_embeddings, in case a smaller vector space should be created for the dataset for quicker executions)
            - build_dataset, in case we want to build the vectorized datasets

      -m : list of models for which the datasets have to be created. Each model might have a different requirement in terms of input-output data

      Create all dataset:
      -------------------
        python3 run_proram.py -exp <exp1> -d <dataset_path> -dt train dev -da build_dataset -dsf <space_file.pkl> -m lr nn rnn -ea fscore -mp False -gpu False

	- dsf : directory of the semantic space file. The current expected file is a pickle file 
	
	- ea  : evaluation action -- fscore, accuracy, etc 
	
	- mp  : multi-processing activation 

	- gpu : Whether to use the GPU or not. It should be TRUE by default 


2. Running the code to run the models on vectorized datasets:

      python3 run_proram.py -exp <exp1> -m rnn -ma train test evaluate -drd ../../data/ -dn atisnlu-dev -sn GoogleNew2013

      Here,
      -exp : as above

      -m : models

      -ma : model actions

      -drd : the location of the data root directory

      -dn : dataset name, useful to identify the dataset files to load

      -sn : space name, useful to identify the dataset files to load

      Run ALL models in the multiprocessing mode:
      -------------------------------------------
        python3 run_proram.py -exp int-cls slot-fill-cls -drd ../../data -dn atisnlu-dev -sn GoogleNews2013 -m lr nn rnn -ma train test evaluate -ea accuracy-one-hot fscore -mp True -gpu False

########################################

CHOICE OF MODELS :
------------------

                                  ** General **

  1. The choice of activation functions depends on the loss function being used.
     The choice of loss function, in turn, depends on the output.
        - Cross entropy goes well with Binary classification
        - Mean-Squared error goes well with Numeric outputs ()
      The activation function is also intuitively decided on the range of the component values of the input vectors.
        - [-1, 1] : tanh
        - [0, 1] : relu, softmax, sigmoid

  2. The number of hidden layers and their units / neurons depends upon the total feature space of the model. This would be an intuitive process, given that the input vectors are generally reduced (dimensionally) with no feature information

                                    ** MODELS **

########################################

  MODEL ARCHITECTURE:

  The Keras toolkit has been used here to create all the models.

  - The models are NOT hard-coded in the sense that the number of layers and all related and relavent parameters can be declared at program initialization and the Model Controller object can create 'n' layered deep models on demand

  - The model parameters, presently, can be declared, when the program is initialized by the

                SOMEController.__init__() module:

                          lr':{'input_dim':[0],
                              'output_dim':[0],
                              'learning_rate':[0.01, 0.0001, 0.00001],
                              'activation_funct':[("")],
                              'output_activation_funct':["softmax"],
                              'cost_funct':["binary_crossentropy"],
                              'l2reg':[0.0, 1e-05, 1e-09],
                              'l1reg':[0.0],
                              'optimizer':["adadelta", "sgd", "adam"],
                              'max_epochs':[1000],
                              'cost_delta':[1e-06],
                              },

        * EACH parameter can have MULTIPLE "tentative" values which can be given as a list.
        * The model will create a PARAMETER STACK from all these parameters by taking a cross-product of all parameter values
        * This STACK can then be executed in parallel to find the best fit amongst all the models specified through these parameters

########################################

  EVALUATION:

  	TBD

########################################

  STATISTICS: ../data/results/statistics
    - Contains the statistics related to datasets, spaces and vectorized datasets
