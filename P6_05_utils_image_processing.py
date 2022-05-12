"""Utilitary functions used for image processing in Project 6"""

# Import des librairies
import os
import time

import cv2
import numpy as np
import pickle
import tensorflow as tf

from PIL import Image, ImageFilter, ImageOps
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img

from P6_04_utils_visualization import *

os.environ["TF_KERAS"] = "1"

# Check versions
print("tensorflow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("is tf built with cuda:", tf.test.is_built_with_cuda())

RAND_STATE = 42


#                       IMAGE PROCESSING                             #
# --------------------------------------------------------------------
def resize_image(img, size=(224, 224), background_color="white"):
    """
    Description: resize input image to requested size, without deformation

    Arguments:
        - img: PIL loaded image
        - size (tuple): size of the returned image
        - background_color (str): color of the background image.
            Can be either 'white' (default) or 'black'
    Returns:
        An image of the required size, without transformation.
        A white (default) or black background is added to avoid deformation
    """

    img_copy = img
    img_copy.thumbnail(size, Image.ANTIALIAS)
    if background_color == "white":
        background = Image.new("RGB", size, (255, 255, 255, 0))
    elif background_color == "black":
        background = Image.new("RGB", size, (0, 0, 0, 0))
    else:
        raise ValueError(
            f'Check background_color. Must be "white" or "black". You provided "{background_color}"'
        )
    background.paste(
        img_copy,
        (int((size[0] - img_copy.size[0]) / 2), int((size[1] - img_copy.size[1]) / 2)),
    )
    return background


def preprocess_image(
    img,
    grayscale=True,
    equalize=True,
    autocontrast=True,
    noise=True,
    filter_type=["boxblur", "gaussianblur"],
):
    """
    Description: process image

    Arguments:
        - img: PIL loaded image
        - grayscale (bool) : convert image to grayscale (default:True).
        - equalize (bool): equalize the image histogram,
            i.e create a uniform distribution of grayscale in the image (default:True).
        - autocontrast (bool): normalized the image (default:True)
        - noise (bool): add noise to the image (default: True)
        - filter_type (str) : type of noise added.
            Can be either 'boxblur', 'gaussianblur' or 'medianblur'
    Returns:
        A processed image
    """

    if autocontrast:
        img_processed = ImageOps.autocontrast(image=img, cutoff=2, ignore=255)
    else:
        img_processed = img

    if equalize:
        img_processed = ImageOps.equalize(img_processed, mask=None)

    if noise:
        for filter in filter_type:
            if filter == "gaussianblur":
                img_processed = img_processed.filter(ImageFilter.GaussianBlur(3))

            elif filter == "boxblur":
                img_processed = img_processed.filter(ImageFilter.BoxBlur(3))

            elif filter == "medianblur":
                img_processed = img_processed.filter(ImageFilter.MedianFilter(5))

            else:
                raise ValueError(
                    f'Check filter_type. Must be "boxblur", "gaussianblur" or "medianblur". You provided "{filter}"'
                )

    if grayscale:
        img_processed = img_processed.convert("L")

    return img_processed


def apply_preprocess(
    input_path,
    output_path,
    list_photos,
    grayscale=True,
    equalize=True,
    autocontrast=True,
    noise=True,
    filter_type=["boxblur", "gaussianblur"],
):
    """
    Description: process batch of image and save them in a specified directory

    Arguments:
        - input_path (str): path to the directory where images are stored
        - output_path (str): name of the directory where to save processed image.
            The directory will be created if needed
        - list_photos: list of image names to process within the input_path
        - grayscale (bool) : convert image to grayscale (default:True).
        - equalize (bool): equalize the image histogram,
            i.e create a uniform distribution of grayscale in the image (default:True).
        - autocontrast (bool): normalized the image (default:True)
        - noise (bool): add noise to the image (default: True)
        - filter_type (str) : type of noise added.
            Can be either 'boxblur', 'gaussianblur' or 'medianblur'
    Returns:
        A processed image
    """

    # Create new directory if unexistent
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Apply preprocess
    for img_file in list_photos:
        img = Image.open(input_path + img_file)
        img_processed = preprocess_image(
            img,
            grayscale=grayscale,
            equalize=equalize,
            autocontrast=autocontrast,
            noise=noise,
            filter_type=filter_type,
        )
        img_processed.save(output_path + img_file)


#                    IMAGE FEATURE EXTRACTIONS                       #
# --------------------------------------------------------------------
def build_histogram(kmeans, des, image_num):
    """
    Description: build histogram of descriptors

    Arguments:
        - kmeans (clusterMixin): clustering method to apply to descriptors
        - des (array): list of descriptors
        - image_num: list of image names
    Returns:
        - histogram of descriptors
    """

    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des = len(des)
    if nb_des == 0:
        print("problème histogramme image: ", image_num)
    for i in res:
        hist[i] += 1.0 / nb_des
    return hist


def process_SIFT(img_path, list_photos):
    # identification of key points and associated descriptors
    """
    Description: identification of keypoints and associated descriptors

    Arguments:
        - img_path (str): path to the directory where images are stored
        - list_photos (array): list of photos to deal with
    Returns:
        - im_features: dense matrix of features for all images in list_photos
        - sift_keypoints_all.shape : number of keypoints found
    """
    sift_keypoints = []
    temps1 = time.time()
    sift = cv2.SIFT_create()

    for img in list_photos:
        # load image
        image = cv2.imread(img_path + img)
        # gest keypoints and descriptors
        kp, des = sift.detectAndCompute(image, None)
        sift_keypoints.append(des)

    print("creating keypoints arrays ...")
    sift_keypoints_by_img = np.asarray(sift_keypoints)
    sift_keypoints_all = np.concatenate(sift_keypoints_by_img, axis=0)

    print()
    print("Nombre de descripteurs : ", sift_keypoints_all.shape)

    duration1 = time.time() - temps1
    print("temps de traitement SIFT descriptor : ", "%15.2f" % duration1, "secondes")

    # Determination number of clusters
    temps1 = time.time()

    k = int(round(np.sqrt(len(sift_keypoints_all)), 0))
    print("Nombre de clusters estimés : ", k)
    print("Création de", k, "clusters de descripteurs ...")

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=k, init_size=3 * k, random_state=0)
    kmeans.fit(sift_keypoints_all)

    duration1 = time.time() - temps1
    print("temps de traitement kmeans : ", "%15.2f" % duration1, "secondes")

    # Creation of histograms (features)
    temps1 = time.time()

    # Creation of a matrix of histograms
    hist_vectors = []

    for i, image_desc in enumerate(sift_keypoints_by_img):
        hist = build_histogram(kmeans, image_desc, i)  # calculates the histogram
        hist_vectors.append(hist)  # histogram is the feature vector

    im_features = np.asarray(hist_vectors)

    duration1 = time.time() - temps1
    print("temps de création histogrammes : ", "%15.2f" % duration1, "secondes")

    return im_features, sift_keypoints_all.shape


def extract_features_with_cnn(file, cnn_model, height=224, width=224):
    """
    Description: extract features of a given image thanks,
        using a Convolutional Neural Network
        (already adapted for features extraction)

    Arguments:
        - file (str): name of the image
        - cnn_model: convolutional Neural Network to use
            (eg: VGG16, ResNet50, EfficientNetB0, ...),
            already adapted for features extraction
        - height (int): height of the images (default: 224)
        - width (int): width of the images default: 224)

    Returns:
        - features: dense matrix of features the image
    """

    # load the image as a 224x224 array
    img = load_img(file, target_size=(height, width))
    # convert image to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = cnn_model.predict(imgx, use_multiprocessing=True)

    return features


def process_cnn(
    image_path, list_photos, cnn_model, img_size=224, error_file="error.pkl"
):
    """
    Description: extract features for a set of images,
        using a Convolutional Neural Network
        (already adapted for features extraction)

    Arguments:
        - image_path (str): path to the directory where images are stored
        - list_photos (array): list of photos to deal with
        - cnn_model: convolutional Neural Network to use
            (eg: VGG16, ResNet50, EfficientNetB0, ...),
            already adapted for features extraction
        - img_size (tuple): height and width of the images (default: 224))
        - error_file (str): name of the created file summarizing errors
            (default: "error.pkl")

    Returns:
        - features: dense matrix of features for the list of images
    """

    # extract feature
    data = {}
    # loop through each image in the dataset
    for img in list_photos:
        # try to extract the features and update the dictionary
        try:
            feat = extract_features_with_cnn(
                image_path + img, cnn_model, img_size, img_size
            )
            data[img] = feat
        # if something fails, save the extracted features as a pickle file
        except:
            with open(str(image_path + "/" + error_file), "wb") as file:
                pickle.dump(data, file)

    # get a list of just the features
    features = np.array(list(data.values()))

    # reshape so that there are 1050 samples of n vectors
    features = features.reshape(-1, features.shape[2])

    return features


def retrained_CNN(
    cnn_model,
    n_classes,
    class_names,
    trainable_layers,
    optimizer_list,
    learning_rate,
    training_set,
    validation_set,
    batch_size,
    summary,
    summary_filename,
    saving_base_name,
    input_path,
    img_size=224,
    epochs=10,
):
    """
    Description: perform supervised classification based on
        transfer learning from a Convolutional Neural Network
        and compute metrics to assess quality of the classification

    Arguments:

        - cnn_model: convolutional Neural Network to use
            (eg: VGG16, ResNet50, EfficientNetB0, ...),
            already adapted for features extraction
        - n_classes (int): number of classes for classification
        - class_names(array): list of class names, as defined in training set
        - trainable_layers (str): can either be 'top' of one only wants to remove
            top layers and retrained new top layers (fully connected with n_classes units)
            or 'inside' if one wants to retrain hidden layers
            (in this case, batchNormalization layers (if exist) will not be retrained,
            neither the first convolutional layers of first blocks)
        - optimizer_list (array): list of optimizers to use (can be "Adam" and/or "SGD")
        - learning_rate (array): list of learning rate values to use
        - training_set: training set as defined by tensorflow.keras.utils.image_dataset_from_directory
        - validation_set: validation set as defined by tensorflow.keras.utils.image_dataset_from_directory
        - batch_size (int): batch size
        - summary (pd.dataFrame): dataframe where to store performance results
        - summary_filename (str): name to use to save summary as a a csv file
        - saving_base_name (str): base name to use to save models
            (will be append with model parameters and 'checkpoint', 'history')
        - input_path (str): path to the directory where the images are stored
        - img_size (int): size (width=height) of the images (default: 224)
        - epochs (int): number of epochs to run for model training (default: 10)

    Returns:
        - summary: appended summary with models performances
        - Each model, checkpoint and history is saved in the current directory :
            - model_name : cnn_model_optimizer_learning-rate_trainLayers_saving-base-name
                (ex : ResNet50_ADAM_0.01_top_classif_29042022)
            - history : model_name_history.npy
            - checkpoints : model_name_checkpoints
    """
    for train_layers in trainable_layers:
        for optimiz in optimizer_list:
            for lr in learning_rate:

                temps1 = time.time()

                # Load model
                print(f"Building model with {train_layers} trainable layers ...")
                classif_cnn = cnn_model(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(img_size, img_size, 3),
                )

                # mark which loaded layers are trainable
                if train_layers == "top":
                    for layer in classif_cnn.layers:
                        layer.trainable = False

                elif train_layers == "inside":
                    # Freeze batch normalization layers
                    for layer in classif_cnn.layers:
                        if isinstance(layer, BatchNormalization):
                            layer.trainable = False
                    # mark some layers as not trainable
                    classif_cnn.get_layer("conv1_conv").trainable = False
                    classif_cnn.get_layer("conv2_block1_1_conv").trainable = False
                    classif_cnn.get_layer("conv2_block1_2_conv").trainable = False
                    classif_cnn.get_layer("conv2_block2_1_conv").trainable = False
                    classif_cnn.get_layer("conv2_block2_2_conv").trainable = False
                    classif_cnn.get_layer("conv2_block3_1_conv").trainable = False
                    classif_cnn.get_layer("conv2_block3_2_conv").trainable = False
                    classif_cnn.get_layer("conv3_block1_1_conv").trainable = False
                    classif_cnn.get_layer("conv3_block1_2_conv").trainable = False
                    classif_cnn.get_layer("conv3_block2_1_conv").trainable = False
                    classif_cnn.get_layer("conv3_block2_2_conv").trainable = False
                    classif_cnn.get_layer("conv3_block3_1_conv").trainable = False
                    classif_cnn.get_layer("conv3_block3_2_conv").trainable = False
                    classif_cnn.get_layer("conv3_block4_1_conv").trainable = False
                    classif_cnn.get_layer("conv3_block4_2_conv").trainable = False
                    classif_cnn.get_layer("conv4_block1_1_conv").trainable = False
                    classif_cnn.get_layer("conv4_block1_2_conv").trainable = False
                    classif_cnn.get_layer("conv4_block2_1_conv").trainable = False
                    classif_cnn.get_layer("conv4_block2_2_conv").trainable = False
                    classif_cnn.get_layer("conv5_block1_1_conv").trainable = False
                    classif_cnn.get_layer("conv5_block1_2_conv").trainable = False

                # Ajout des couches pour la classification
                drop1 = Dropout(0.3)(classif_cnn.layers[-1].output)
                flat1 = Flatten()(drop1)
                class1 = Dense(1024, activation="relu")(flat1)
                output = Dense(n_classes, activation="softmax")(class1)

                # Définiton du nouveau modele
                classif_cnn = Model(inputs=classif_cnn.inputs, outputs=output)

                # Compilation
                print(f"Model compilation with {optimiz}, learning rate : {lr} ...")
                if optimiz == "SGD":
                    opt = optimizers.SGD(learning_rate=lr, momentum=0.9)

                elif optimiz == "ADAM":
                    opt = optimizers.Adam(learning_rate=lr)

                # Compilation
                classif_cnn.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer=opt,
                    metrics=["sparse_categorical_accuracy"],
                )

                # Entrainement
                print("Model training ...")
                history = classif_cnn.fit(
                    training_set,
                    validation_data=validation_set,
                    batch_size=batch_size,
                    epochs=epochs,
                )

                # Compute ARI
                print("Computing ARi ...")
                ARI = compute_ARI_fromCNN(
                    classif_cnn, validation_set, class_names, plot=False
                )
                print("ARI: ", ARI)

                # Compute execution time
                time2 = time.time() - temps1

                # Appending summary
                print("Appending summary ...")
                summary = summary.append(
                    {
                        "image_path": input_path,
                        "method": str("train " + train_layers + " layers"),
                        "algo": cnn_model.__name__,
                        "ARI": ARI,
                        "execution_time": time2,
                        "params": {"optimizer": optimiz, "learning rate": lr},
                        "training_accuracy": history.history[
                            "sparse_categorical_accuracy"
                        ][-1],
                        "validation accuracy": history.history[
                            "val_sparse_categorical_accuracy"
                        ][-1],
                    },
                    ignore_index=True,
                )

                print("Save summary file")
                print("--------------------")
                summary.to_csv(summary_filename)
                summary.to_pickle(str("p_" + summary_filename))

                # Sauvegarde de l'historique et du modèle
                model_name = str(
                    cnn_model.__name__
                    + "_"
                    + optimiz
                    + "_"
                    + str(lr)
                    + "_"
                    + train_layers
                    + "_"
                    + saving_base_name
                )

                print("Saving model ")
                np.save(str(model_name + "_history.npy"), history.history)
                classif_cnn.save(model_name)
                classif_cnn.save_weights(str(model_name + "_checkpoint"))

    return summary
