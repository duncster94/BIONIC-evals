[![DOI](https://zenodo.org/badge/340116033.svg)](https://zenodo.org/badge/latestdoi/340116033)

# BIONIC-evals
Evaluation library for [BIONIC](https://github.com/bowang-lab/BIONIC). This library contains code to reproduce the co-annotation prediction, module detection, and gene function prediction evaluations from Fig. 2a, 3a, 4 and 5.

**NOTE:** The module detection and gene function prediction evaluations take a considerable amount of time to complete (on the order of 10s of hours). This can be sped up with multiprocessing. If you'd like me to implement this functionality, please open an [issue](https://github.com/duncster94/BIONIC-evals/issues).

## :gear: Installation
The library can be installed using [Poetry](https://python-poetry.org).


1. First, [install Poetry](https://python-poetry.org/docs/#installation).

2. Create a virtual Python **3.8** environment using [conda](https://docs.anaconda.com/anaconda/user-guide/getting-started/):
        
        $ conda create -n bionic-evals python=3.8
    
3. Make sure your virutal environment is active for the following steps:
        
        $ conda activate bionic-evals

4. Clone this repository by running

       $ git clone https://github.com/duncster94/BIONIC-evals.git

5. Make sure you are in the same directory as the `pyproject.toml` file. Install the `bioniceval` library as follows:

       $ poetry install

6. Test `bioniceval` is installed properly by running

       $ bioniceval --help
       
    You should see a help message.

## :zap: Usage

You can run `bioniceval` by simply passing in a config file as follows:

    $ bioniceval path/to/config/file.json

### Configuration File
`bioniceval` runs by passing in a configuration file: a [JSON](https://www.w3schools.com/whatis/whatis_json.asp) file containing all the relevant file paths and evaluation parameters. You can have a uniquely named config file for each evaluation scenario you want to run. An example config file can be found [here](https://github.com/duncster94/BIONIC-evals/blob/main/bioniceval/config/fig2a_config.json).

The configuration keys are as follows:

Argument | Description
--- | ---
**Input files** |
`networks.name` | Name for the given network.
`networks.path` | Filepath to input network.
`networks.delimiter` | Delimiter of network file.
`features.name` | Name for the given feature set.
`features.path` | Filepath to input feature set.
`features.delimiter` | Delimiter of feature file.
**Evaluation standards** | 
`standards.name` | Name for the given standard.
`standards.task` | The type of evaluation task. Valid values are `"coannotation"`, `"module_detection"`, and `"function_prediction"`
`standards.path` | Filepath to standard.
`standards.delimiter` | Delimiter of standard file.
**Module detection specific parameters**
`standards.samples` | Number of flat module set samples to perform evaluations for.
`standards.methods` | A list of valid linkage methods to perform clustering for. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for more information.
`standards.metrics` | A list of valid distance metrics to perform clustering for. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist) for more information.
`standards.thresholds` | Number of clustering thresholds to extract clusters for and evaluate.
**Function prediction specific parameters**
`standards.test_size` | Held-out test size. A value of 0.1 corresponds to test set of 10% of genes.
`standards.folds` | Number of folds to perform cross validation on.
`standards.trials` | Number of trials to repeat function prediction evaluations for.
`standards.gamma.minimum` | Lower bound of radial basis function kernel coefficient.
`standards.gamma.maximum` | Upper bound of radial basis function kernel coefficient.
`standards.gamma.samples` | Number of coefficients to sample from the range defined by `minimum` and `maximum` arguments.
`standards.regularization.minimum` | Lower bound of regularization parameter (`C` in scikit-learn [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)).
`standards.regularization.maximum` | Upper bound of regularization parameter.
`standards.regularization.samples` | Number of regularization parameters to sample from the range defined by `minimum` and `maximum` arguments.
**Miscellaneous** |
`consolidation`| Whether to consolidate differences in gene sets between datasets by extending datasets to the union of genes (`"union"`) or reducing datasets to the intersection of genes (`"intersection"`). `union` was used for analyses in the BIONIC manuscript.
