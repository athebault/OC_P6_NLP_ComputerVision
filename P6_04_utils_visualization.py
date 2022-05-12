"""Utilitary functions for vizualisation used in Project 6"""

# Import des librairies
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from collections import Counter
from PIL import Image
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from wordcloud import WordCloud

DPI = 300
RAND_STATE = 42


#                            EXPLORATION                             #
# --------------------------------------------------------------------
def plot_wordcloud(token_list, cat=None, figsave=None):
    """
    Description: plot wordcloud of most important tokens from a list of tokens

    Arguments:
        - token_list (list): list of token lists
        - cat (str): categorie name for plot title
        - figsave (str) : name of the figure if want to save it

    Returns :
        - Wordcloud of tokens, based on tokens counts
    """
    wc = WordCloud(background_color="white", width=1000, height=500)
    wordcloud = wc.generate_from_text(" ".join(token_list))

    plt.figure(figsize=(12, 6))
    plt.suptitle(cat, fontsize=24, fontweight="bold")
    plt.imshow(wordcloud)
    plt.axis("off")

    if figsave:
        plt.savefig(figsave, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_barplot_of_tags(
    tags_list,
    nb_of_tags,
    xlabel="Nombre d'occurences",
    ylabel="",
    figsave=None,
    figsize=(10, 30),
):
    """
    Description: plot barplot of tags count (descending order) from a list of tags

    Arguments:
        - tags_list (lsit): list of tags
        - nb_of_tags (int) : number of tags to plot in barplot (default=50)
        - xlabel, ylabel (str): labels of the barplot
        - figsize (list) : figure size (default : (10, 30))

    Returns :
        - Barplot of nb_of_tags most important tags

    """
    tag_count = Counter(tags_list)
    tag_count_sort = dict(tag_count.most_common(nb_of_tags))

    plt.figure(figsize=figsize)
    sns.barplot(
        x=list(tag_count_sort.values()),
        y=list(tag_count_sort.keys()),
        orient="h",
        palette="viridis",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if figsave:
        plt.savefig(figsave, bbox_inches="tight")
    plt.show()


#                    REDUCTION DIMENSION                             #
# --------------------------------------------------------------------
class TSNE_wrapper(BaseEstimator, TransformerMixin):
    """
    Description: wrapper to use TSNE embedding in pipeline
    """

    def __init__(self, n_components=2, perplexity=30, init="random", random_state=None):

        self.n_components = n_components
        self.perplexity = perplexity
        self.init = init
        self.random_state = random_state

    def fit(self, X, y=None):
        ts = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            init=self.init,
            random_state=self.random_state,
        )
        self.X_tsne = ts.fit_transform(X)
        return self

    def transform(self, X, y=None):
        ts = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            init=self.init,
            random_state=self.random_state,
        )
        return ts.fit_transform(X)


def reduce_dimension(
    method,
    data,
    tsne_comp=2,
    pca_comp=0.8,
    lsa_comp=100,
    lda_comp=10,
    nmf_comp=10,
    tsne_perplexity=30,
):
    """
    Description: reduce data dimension

    Arguments:
        - method (str): reduction method. Can be one of:
            - 'TSNE', 'PCA/LSA',
            - 'NMF' for Non-Negative Matrix Factorization
            - 'LDA' for LatentDirichletAllocation
            In case of 'PCA/LSA', PCA will be done if possible,
            but if data matrix is sparse, TruncatedSVD will be performed
        - data (dataframe): data on which apply reduction dimension
        - tsne_comp (int) : number of components for TSNE
        - pca_comp (int, float) : number of components for PCA.
            If 0 < pca_comp < 1, select the number of components
            such that the amount of variance that needs to be
            explained is greater than the percentage specified
            by pca_comp.
        - lsa_comp (int): number of components for LSA
        - lda_comp (int): number of components for Latent Dirichlet Allocation
        - nmf_comp (int): number of components for NMF
        - tsne_perplexity (int) : perplexity value for TSNE

    Returns :
        - dimension reduced data
    """

    # check if matrix is sparse:
    sparse = scipy.sparse.issparse(data)

    # check minimal value of the matrix
    if hasattr(data, "min") and data.min() < 0:
        is_neg = True
    elif not hasattr(data, "min") and np.asarray(data).min() < 0:
        is_neg = True
    else:
        is_neg = False

    # Reduction dimension:
    explained_var = "nan"

    if method == "TSNE":
        n_comp = tsne_comp
        tsne = TSNE(
            n_components=n_comp, perplexity=tsne_perplexity, random_state=RAND_STATE
        )
        x_red = tsne.fit_transform(data)

    elif method == "PCA/LSA":
        if sparse:
            n_comp = lsa_comp
            pca = TruncatedSVD(n_components=n_comp, random_state=RAND_STATE)
        else:
            n_comp = pca_comp
            pca = PCA(n_components=n_comp, random_state=RAND_STATE)
        x_red = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_.sum()

    else:
        if is_neg:
            raise ValueError(
                "data contains negative values - Cannot perform LDA or NMF"
            )
        else:
            if method == "LDA":
                n_comp = lda_comp
                lda = LatentDirichletAllocation(
                    n_components=n_comp,
                    learning_method="online",
                    random_state=RAND_STATE,
                )
                x_red = lda.fit_transform(data)

            elif method == "NMF":
                n_comp = nmf_comp
                nmf = NMF(n_components=n_comp, random_state=RAND_STATE)
                x_red = nmf.fit_transform(data)

    return x_red, explained_var, n_comp


#                      METRICS COMPUTATION                           #
# --------------------------------------------------------------------
def compute_metrics(data, true_labels, pred_labels, round=4):
    """
    Description: compute clustering metrics: ARI and silhouette score

    Arguments:
        - data (dataframe): data on which apply reduction dimension
        - true_labels (array1D) : true labels
        - pred_labels (array1D) : predicted labels
        - round (int) : number of decimals (default:4)

    Returns :
        - ARI value and mean silhouette score between predicted and true labels
    """

    # Calcul des métriques
    ARI = np.round(metrics.adjusted_rand_score(true_labels, pred_labels), round)
    silhouette = np.round(silhouette_score(data, pred_labels), round)

    return ARI, silhouette


def get_labels_from_retrainedCNN(model, val_ds, class_names):
    """
    Description: retrieve labels from cnn validation dataset
    Arguments:
        - model : CNN model
        - val_ds (array1D) : validation set
        - class_names (array1D) : list of class names,
            as defined in train_ds
    Returns :
        - true_labels_num: predicted classes
        - true_labels_cat: predicted labels
        - pred_labels_num: true classes
        - pred_labels_cat: true labels
        - image_val: image of the validation set, as numpy array
    """

    # Get predictions
    predictions = model.predict(val_ds)
    pred_labels_cat = [np.nan] * len(predictions)
    pred_labels_num = [np.nan] * len(predictions)
    true_labels_cat = [np.nan] * len(predictions)
    for i, pred in enumerate(predictions):
        pred_labels_cat[i] = class_names[np.argmax(pred)]
        pred_labels_num[i] = np.argmax(pred)

    # Get true labels
    true_labels_num = np.concatenate([labels for images, labels in val_ds], axis=0)
    image_val = np.concatenate([images for images, labels in val_ds], axis=0)

    for i, lab in enumerate(true_labels_num):
        true_labels_cat[i] = class_names[lab]

    return (
        true_labels_num,
        true_labels_cat,
        pred_labels_num,
        pred_labels_cat,
        image_val,
    )


def compute_ARI_fromCNN(model, val_ds, class_names, plot=False):
    """
    Description: compute ARI on CNN classification ouptut
    Arguments:
        - model : CNN model
        - val_ds (array1D) : validation set
        - class_names (array1D) : list of class names,
            as defined in train_ds
        - plot (bool): if True plot confusion matrix
            between true and predicted labels

    Returns :
        - ARI value and potentially confusion matrix
            between predicted and true labels
    """
    # Get predictions
    true_labels_num, _, pred_labels_num, _, _ = get_labels_from_retrainedCNN(
        model, val_ds, class_names
    )

    # Compute ARI
    ARI = np.round(metrics.adjusted_rand_score(true_labels_num, pred_labels_num), 4)

    if plot:
        res = confusion_matrix(true_labels_num, pred_labels_num)
        sns.heatmap(res, annot=True, fmt="d", cmap="rocket_r")
        print("ARI: ", ARI)

    return ARI


def plot_cnn_metrics_dynamics(
    model_history,
    train_metrics="sparse_categorical_accuracy",
    test_metrics="val_sparse_categorical_accuracy",
):
    """
    Description: plot training and validation accurarcy dynamics over epochs
    Arguments:
        - model_history : history of the model (contained in model.history)
        - train_metrics (str) : metric used for training set evaluation
            (default: sparse_categorical_accuracy)
        - test_metrics (str): metric used for validation set evaluation
            (default: val_sparse_categorical_accuracy)

    Returns :
        - plots of model accuracy and loss for both training and validation sets
    """
    # Visualisation du modele
    acc = model_history[train_metrics]
    val_acc = model_history[test_metrics]

    loss = model_history["loss"]
    val_loss = model_history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


#                           VISUALISATION                            #
# --------------------------------------------------------------------
def catplot_summary(
    summary,
    var="ARI",
    y="feature_name",
    hue="reduction_dimension",
    col="algorithm",
    col_wrap=2,
    savefig=None,
):

    """
    Plot values of 'var' accross combinaisons
    of y, hue and col.

    Arguments:
        - summary (dataframe) : dataframe where results are stored.
            It must contain the following columns:
                - "feature_name"
                - "reduction_dimension"
                - "algorithm"
        - var (str) : name of the variable to plot
            (usually "ARI" (default) or "silhouette_score")
        - y (str) : variable to plot on y axis (default: "feature_name")
        - hue (str) : variable to use as hue (default: "reduction_dimension")
        - col (str) : variable to use to separate plots (default: "algorithm")
        - col_wrap (int) : number of columns
        - savefig (str): name of the saved plot

    Returns:
        - a catplot of var values for all combinaisons of y, hue and col.
    """
    sns.catplot(
        x=var, y=y, hue=hue, col=col, kind="bar", col_wrap=col_wrap, data=summary
    )

    if savefig:
        plt.savefig(savefig, bbox_inches="tight", dpi=DPI)


def get_best_results(summary):
    """
    Get rows containing either higher ARI, or higher silhouette score
    or higher (ARI + Silhouette_score) from a summary dataframe

    Arguments:
        - summary (dataframe) : dataframe where results are stored.
            It must contain the following columns:
                - "ARI"
                - "Silhouette_score"
    Returns:
        - idx : list of indexes of combinaisons with higher ARI,
            higher silhouette score and higher (ARI + Silhouette_score).
        - best_res : dataframe with combinaisons with higher ARI,
            higher silhouette score and higher (ARI + Silhouette_score).
    """
    idx = []
    best_res = pd.DataFrame()

    # maximum ARI
    max_ari = summary.loc[summary.loc[:, "ARI"] == summary.loc[:, "ARI"].max()]
    print("Maximum ARI index: ", max_ari.index.values)
    idx += max_ari.index.values
    best_res = best_res.append(max_ari)

    # maximum silhouette_score
    max_silhouette = summary.loc[
        summary.loc[:, "Silhouette_score"] == summary.loc[:, "Silhouette_score"].max()
    ]
    print("Maximum Silhouette score index: ", max_silhouette.index.values)
    idx += max_silhouette.index.values
    best_res = best_res.append(max_silhouette)

    # maximum meta-score (ARI + Silhouette score)
    optim = summary.loc[
        summary.loc[:, ["ARI", "Silhouette_score"]].sum(axis=1)
        == summary.loc[:, ["ARI", "Silhouette_score"]].sum(axis=1).max()
    ]
    print("Optimal model index: ", optim.index.values)
    idx += optim.index.values
    best_res = best_res.append(optim)

    return idx, best_res


###                   EMBEDDING VISUALISATION                      ###
# --------------------------------------------------------------------#
def visualize_w2v(w2v_model, n_words=100):
    """
    Visualize word2vec word similarity on PCA

    Arguments:
        - w2v_model : word2vec model
        - n_words (int): number of words to plot (default: 100)

    Returns:
        - Plot of n_words embbed with word2vec model
    """
    words = w2v_model.wv.index_to_key[:n_words]
    X = w2v_model.wv[words]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(10, 10))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()


###                   CLUSTERING VISUALISATION                      ###
# --------------------------------------------------------------------#
# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(x_tsne, y_cat_num, categ_names, labels, ARI, title):
    """
    Description: Visualize true and predicted categories on TSNE projection

    Arguments:
        - x_tsne (list): list of token lists
        - y_cat_num (str): list of true labels for all samples
        - categ_names (str): list of true labels names
        - labels (list of str): list of predicted labels
        - ARI (float, optional): if given, print Adjusted Rand Index value
            between true and predicted labels
        - title (str, optional): if given, suptitle of the figure

    Returns :
        - Plots of projected true and predicted labels on TSNE projection.
            Also print ARI value if given
    """

    fig = plt.figure(figsize=(15, 15))
    if title:
        plt.suptitle(title, fontsize=24, fontweight="bold")

    ax1 = fig.add_subplot(211)
    scatter = ax1.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_cat_num, cmap="Set1")
    ax1.legend(
        handles=scatter.legend_elements()[0],
        labels=categ_names,
        frameon=True,
        bbox_to_anchor=(1, 1),
        loc="best",
        title="Categorie",
    )
    plt.title("Représentation des produits par catégories réelles")

    ax2 = fig.add_subplot(212)
    scatter = ax2.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap="Set1")
    ax2.legend(
        handles=scatter.legend_elements()[0],
        labels=set(labels),
        frameon=True,
        bbox_to_anchor=(1, 1),
        loc="best",
        title="Clusters",
    )
    plt.title("Représentation des produits par clusters")

    plt.show()
    if ARI:
        print("ARI : ", ARI)


# visualisation du graph des silhouettes
def show_silhouette(clust, data):
    """
    Description: plot silhouette plot

    Arguments:
        - clust (clusterMixin): clustering model
        - data (dataframe): data used for clustering

    Returns :
        - silhouette plot
    """
    visualizer = SilhouetteVisualizer(clust)
    visualizer.fit(data)
    visualizer.finalize()


# visualisation de la matrice de confusion
def plot_contingency_table(
    truth,
    labels,
    title="Contingency Table",
    set_aspect=True,
    figsize=(10, 5),
    savefig=None,
):
    """
    Description: plot confusion matrix between true and predicted labels

    Arguments:
        - truth (1Darray): true labels
        - labels (1Darray)): predicted labels
        - title (str): plot title
        - set_aspect (bool): aspect (height/width),
            if True (default), set to width=height
        - figsize : size of the figure (default (10,5))
        - savefig : name of the saved plot

    Return :
        - Contingency table of the two variables (no annotation)
    """

    # Create dataframe of results
    res = pd.DataFrame({"Categories": truth, "Prediction": labels})

    # Create contingency table
    cont_table = pd.crosstab(
        index=res.loc[:, "Categories"], columns=res.loc[:, "Prediction"], margins=False
    )

    # Plot table
    fig = plt.figure(num=None, figsize=figsize, facecolor="w", edgecolor="k")
    ax1 = fig.add_subplot(111)
    if set_aspect:
        ax1.set_aspect(1)
    sns.heatmap(
        cont_table, cmap="rocket_r", vmin=0, annot=True, fmt="d",
    )
    plt.title(title, fontsize=20)

    if savefig:
        plt.savefig(savefig, bbox_inches="tight", dpi=DPI)

    plt.show()


def visualize_from_summary(summary, idx, y_cat_num, true_labels):
    """
    Description: retrieve results of index in summary table and plot results:
        silhouette plot, TSNE 2D projection and
        confusion matrix between true and predicted labels

    Arguments:
        - summary (pd.DataFrame): dataframe with data to plot.
            Must contain the following columns:
                - 'algorithm'
                - 'ARI'
                - 'outputs' which should contain : 'x_red', 'labels' and 'cls'

        - idx (int): index of the summary to plot
        - y_cat_num (array1D): true labels (numerical values)
        - true_labels (array1D): true labels (categorical values)

    Return :
        - Silhouette plot, TSNE 2D projection and
            confusion matrix between true and predicted labels

    """

    # recuperation données
    method = summary.loc[idx, "algorithm"]
    ari = summary.loc[idx, "ARI"]
    outputs = summary.loc[idx, "outputs"]
    x_red = outputs.get("x_red")
    labels = outputs.get("labels")
    model = outputs.get("cls")

    # Get true category names
    categ_names = set(true_labels.unique())

    # Visualisation
    show_silhouette(model, x_red)
    TSNE_visu_fct(x_red, y_cat_num, categ_names, labels, ari, title=method)
    plot_contingency_table(true_labels, labels, title=method)


def process_features(
    features,
    n_clusters,
    true_labels,
    categ_names,
    plot=True,
    title="",
    pca=False,
    n_comp_pca=0.95,
    tsne=True,
    n_comp_tsne=2,
    conf_matrix=True,
):
    """
    Description: process features, i.e dimension reduction and clustering and computes metrics

    Arguments:
        - features (Array): feature matrix. Must be a dense matrix.
        - n_clusters (int): number of clusters for clustering
        - true_labels (array1D): array of true labels
        - categ_names (list if str): list of category names
        - plot (bool): if true, plot TSNE vizualisation and silhouette plot
        - title (str): title of the plots
        - pca (bool): if true perform a PCA as a first step of dimension reduction (default: False)
        - n_comp_pca (int or float): number of components (if int) or percentage
            of variance represented by components (if float <1, default 0.95) for PCA
        - tsne (bool): if true (default), perform a TSNE for dimension reduction
        - n_comp_tsne (int): number of components for TSNE (default: 2)
        - conf_matrix (bool): if true (default), plot confusion matrix between
            true and predicted labels

    Return :
        - ARI: adjusted rand Index between true and predicted labels
        - silhouette: mean silhouette score of clustering

    """

    if pca:
        # Dimensionaliy reduction
        pca = PCA(n_components=n_comp_pca, random_state=22)
        pca.fit(features)
        x_red = pca.transform(features)
    else:
        x_red = features

    if tsne:
        tsne = TSNE(
            n_components=n_comp_tsne,
            perplexity=30,
            n_iter=2000,
            init="random",
            random_state=6,
        )
        x_red = tsne.fit_transform(x_red)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(x_red)
    labels = kmeans.labels_

    # Compute metrics
    ARI, silhouette = compute_metrics(x_red, true_labels, labels)

    if plot:
        # Visualisation des résultats
        TSNE_visu_fct(x_red, true_labels, categ_names, labels, ARI, title)

        show_silhouette(kmeans, x_red)
        print("Mean silhouette score : ", silhouette)

        if conf_matrix:
            plot_contingency_table(true_labels, labels, title=title)

    return ARI, silhouette


def get_misclassed_images(
    categ, df, labels_dict, wordcloud=True, plot=True, img_path=""
):
    """
    Description: show details on misclassed samples for a given category

    Arguments:
        - categ (str): category.
        - df (pd.dataFrame): dataframe
        - labels_dict (dict): concordance between predicted labels (numerical)
            and most probable category (str)
        - wordcloud (bool): if true (default), plot wordcloud of misclassed samples description
        - plot (bool): if true, show a sample of misclassed images of the category
        - img_path (str): path to the directory where images are stored

    Returns :
        - Wordcloud of tokens present in description of the misclassed products
            of the category and sample of misclassed images
    """

    # Get dictionary keys and values lists
    key_list = list(labels_dict.keys())
    val_list = list(labels_dict.values())

    # Get numerical value for predicted category
    pred_num_categ = key_list[val_list.index(categ)]

    # Find misclassed product
    categ_df = df.loc[df.loc[:, "true_categ"] == categ]
    misclassed = categ_df.loc[categ_df.loc[:, "labels"] != pred_num_categ].reset_index(
        drop=True
    )

    if plot:
        plot_misclassed(img_path, misclassed, categ, labels_dict)

    if wordcloud:
        plot_wordcloud(misclassed["tokens"], cat=categ, figsave=None)


def plot_misclassed(img_path, misclassed, categ, labels_dict):
    """
    Description: show photos of 6 randomely sampled misclassed products within a category

    Arguments:
        - img_path (str): path to the directory where images are stored
        - misclassed (pd.DataFrame): dataframe containing misclassed
            products of the given category
        - categ (str): category name.
        - labels_dict (dict): concordance between predicted labels (numerical)
            and most probable category (str)

    Returns :
        - Show photos of 6 misclassed products of the category
    """

    n = len(misclassed)
    temp = misclassed.copy()
    nb_pict = 6
    plt.figure(figsize=(8, 8))
    for p in range(nb_pict):
        ax = plt.subplot(2, 3, p + 1)
        plt.suptitle(f"True category : {categ}")
        i = np.random.randint(0, n)
        img_name = temp.loc[i, "image"]
        print("image: ", img_name)
        pred_label = labels_dict[temp.loc[i, "labels"]]
        plt.imshow(Image.open(os.path.join(img_path, img_name)))
        plt.title(f"Pred. Categ : {pred_label}")
        temp = temp.drop(index=i).reset_index(drop=True)
        n -= 1
