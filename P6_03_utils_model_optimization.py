"""Utilitary functions used for model optimization in Project 6"""

# Import des librairies

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV


#                    HYPER-PARAMETER TUNING                          #
# --------------------------------------------------------------------
def gridsearch_pipe(
    estimator, grid_params, cv, scoring, refit, return_train_score, n_jobs
):
    """
    Description: pipeline hyperparameter tuning by gridsearchCV

    Args:
        - estimator: estimator to use
        - grid_params (dict): dictionnary containing hyperparameters' names
        as keys and ranges to look at
        - cv (int): number of cross-validation folds
        - scoring (str): strategy to evaluate the performance
        of the cross-validated model on the test set.
        - refit (str): refit an estimator using the
        best found parameters on the whole dataset.
        - return_train_score (bool): if False, the cv_results_ attribute
        will not include training scores.
        - n_jobs (int): number of jobs to run in parallel.
        -1 means using all processors.

    Return :
        - A dict with keys as column headers and values as columns,
        that can be imported into a pandas DataFrame
    """
    gridsearch_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=grid_params,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring,
        refit=refit,
        return_train_score=return_train_score,
        error_score="raise",
    )

    return gridsearch_pipeline


def get_modelCV_output(model):
    """
    Description: get hyperparameters set and metrics scores
    corresponding to the best model refited by gridSearchCV

    Args:
        - model: model entered as estimator in gridSearchCv (can be a pipeline)

    Return :
        - scores on training and cross-validation sets of the best
        model found by gridSearchCV
    """

    result = pd.DataFrame(model.cv_results_)
    best_res = result.loc[model.best_index_]
    print(best_res.params)
    return best_res


def update_gridsearch_summary(
    summary,
    feature,
    preprocess_name,
    vectorization,
    dim_reduction,
    pipeline_best_result,
):
    """
    Description: update summary of modeling results following
    tuning hyperparameter by gridsearchCV

    Args:
        - summary (dataframe): sumary of modeling results
        - model_name (str): name of the model
        - pipeline_best_result : best model hyperparameters
            chosen by gridsearchCV
        - preprocess_param (dict): dictionnary of preprocessing data

    Return :
        - updated summary of modeling results
    """

    summary = summary.append(
        {
            "preprocess": preprocess_name,
            "feature_name": feature,
            "algorithm": vectorization,
            "reduction_dimension": dim_reduction,
            "best_param": pipeline_best_result.params,
            "ARI": pipeline_best_result.mean_test_score,
            "execution_time": pipeline_best_result.mean_fit_time,
        },
        ignore_index=True,
    )
    return summary


def run_hyperparameter_tuning(
    pipe_name,
    grid_params,
    feature,
    vectorization,
    dim_reduction,
    model_summary,
    X,
    labels,
    cv=1,
    scoring="adjusted_rand_score",
    refit="adjusted_rand_score",
    return_train_score=True,
    n_jobs=-1,
):
    """
    Description: run hyperparameter tuning process and summary update

    Args:
        - pipe_name: estimator to use
        - grid_params (dict): dictionnary containing
            hyperparameters' names (as keys) and ranges to look at
        - model_name (str): name of the model
        - preprocess_param (dict): dictionnary of preprocessing data
        - model_summary (dataframe): sumary of modeling results
        - X (dataframe) : explanatory variable
        - labels (array) : arrays of target values
        - cv (int): number of cross-validation folds
        - scoring (str): strategy to evaluate the performance of the
            cross-validated model on the test set (default: 'r2')
        - refit (str): refit an estimator using the best found
            parameters on the whole dataset (default: 'r2').
        - return_train_score (bool): if False, the cv_results
            attribute will not include training scores (default: True.
        - n_jobs (int): number of jobs to run in parallel.
            Default: -1 means using all processors.

    Return :
        - updated summary of hyperparameter tuning
    """

    # Build pipeline
    gridPipeline = gridsearch_pipe(
        pipe_name, grid_params, cv, scoring, refit, return_train_score, n_jobs
    )

    # Fit
    gridPipeline.fit(X, labels)

    # Recupère les résultat
    best_res = get_modelCV_output(gridPipeline)

    # Mise à jour du dataframe de résultats
    model_summary = update_gridsearch_summary(
        model_summary, feature, vectorization, dim_reduction, best_res
    )
    return model_summary


def plot_piechart_from_gridsearch(params, var_list, figsize=(20, 5)):
    # Setupt layout
    fig, axs = plt.subplots(1, len(var_list), figsize=figsize)
    # get values
    for i, var in enumerate(var_list):
        var_values = [params[i].get(var) for i in range(0, len(params))]
        count = pd.Series(var_values).value_counts()
        # Plot
        axs[i].pie(count, labels=count.index)
        axs[i].set_title(var)

    plt.show()
