import os
import argparse
import random
import copy
import pickle
from collections import Counter

import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold, StratifiedGroupKFold
from sklearn.utils import resample, shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

from feature_preprocessing import FeaturePreprocessing

"""
augmented - "upsample": duplacation, "augmented": using EDA
"""

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def prediction_model(seed, opt_model, input, context, gender, pretest,
                     hidden, epochs, batch, preprocessing, pca, features, user, seqdrop, collab, count, dr, seqdrop_rate, attention_type):

    ############################
    # (1) Load Data
    ###########################
    n_splits = 10

    ######Total Data######
    X_file_path = f"data/{input}_X_context={context}_pre={preprocessing}_pca={pca}_{features}_collab_{collab}.pkl"
    X_user_file_path = f"data/{input}_User_context={context}_pre={preprocessing}_pca={pca}_{features}_collab_{collab}.pkl"
    X_user_idx_file_path = f"data/{input}_User_idx_context={context}_pre={preprocessing}_pca={pca}_{features}_collab_{collab}.pkl"

    X_user_feature_file_path = "data/{}_User_Feature_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(input, context, preprocessing, pca, features, collab)
    X_collab_file_path = "data/{}_Collab_Feature_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(input, context, preprocessing, pca, features, collab)

    y_file_path = "data/{}_Labels_context={}_pre={}_pca={}_{}.pkl".format(input, context, preprocessing, pca, features)
    groups_file_path= "data/{}_Groups_context={}_pre={}_pca={}_{}.pkl".format(input, context, preprocessing, pca, features)
    cv_split_path = "data/cv_split_{}_{}_seed_{}.pkl".format(n_splits, seed)

    if os.path.isfile(X_file_path) and os.path.isfile(X_collab_file_path) and os.path.isfile(X_user_file_path) and os.path.isfile(X_user_idx_file_path) and os.path.isfile(X_user_feature_file_path) and os.path.isfile(y_file_path) and os.path.isfile(groups_file_path):
        print("Files Exist!!")
        X = pickle.load((open(X_file_path, "rb")))
        X_user = pickle.load((open(X_user_file_path, "rb")))
        X_user_idx = pickle.load((open(X_user_idx_file_path, "rb")))
        X_collab = pickle.load((open(X_collab_file_path, "rb")))
        #X_user_feature = pickle.load((open(X_user_feature_file_path, "rb")))
        y = pickle.load((open(y_file_path, "rb")))
        groups = pickle.load((open(groups_file_path, "rb")))

    else:
        print("No Files, creating one...")

        vectorizer = FeaturePreprocessing(input, preprocessing, pca, features)

        X, X_user, X_user_idx, y, groups, X_user_feature, X_collab = vectorizer.create_data(context, gender, pretest, collab)

    X = np.array(X)
    X_user = np.array(X_user)
    X_user_idx = np.array(X_user_idx)
    X_collab = np.array(X_collab)
    #X_user_feature= np.array(X_user_feature)
    y = np.array(y)
    groups = np.array(groups)

    ############################
    # (2) Create Group-CV splits
    ###########################

    random_state = 22

    if os.path.isfile(cv_split_path):
        print("CV splits exists!")
        cv_splits = pickle.load(open(cv_split_path, "rb"))
    else:
        print("CV splits NOT exists, creating one...")

        group_kfold = StratifiedGroupKFold(n_splits=n_splits)#, random_state=random_state, shuffle=True)

        cv_splits = []
        for train_idx, test_index in group_kfold.split(X, y, groups):

            ##For validation-set
            sss = StratifiedShuffleSplit(n_splits=1, test_size=len(test_index))

            for train_index_splitted, valid_idx in sss.split(X[train_idx], y[train_idx]):
                train_index = train_idx[train_index_splitted]
                valid_index = train_idx[valid_idx]

                print(train_index)
                print(valid_index)

                break

            cv_splits.append([np.array(train_index), np.array(valid_index), np.array(test_index)])


        pickle.dump(cv_splits, open(cv_split_path, 'wb'), -1)


    print("seed={}, model={}, input={}, context={}".format(
        seed, opt_model, input, context))

    """ Setting a Random Seed """
    if seed:
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    else:
        print("No seed applied!")

    # To display all contents
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    # date_md = datetime.now().strftime('%m%d_%H%M')

    if not seed and opt_model == 'lstm':
        print("NO Seed & Using LSTM!")

    i = 0
    columns = ['Fold', 'Type', 'Model', 'Input', 'Context',
               'Accuracy', 'Precision', 'Recall', 'Fmeasure', 'AUC', 'PR_AUC']

    performances = pd.DataFrame(columns=columns)

    # List of classic machine learning methods
    classic_ml = ['lr', 'gaussian_nb', 'multinomial_nb', 'svm', 'rf']

    ############################
    # (2) Define a model
    ###########################

    cv_splits_iter = cv_splits

    precisions_all_folds = []
    recalls_all_folds = []
    thresholds_all_folds = []


    # >> Group-level Leave One Out Cross Validation (18 folds)
    for fold_idx, (train_idx, valid_index, test_idx) in enumerate(cv_splits_iter):

        print("\nFold {}, Start >>".format(fold_idx + 1))

        print(np.shape(train_idx))
        print(np.shape(test_idx))

        y_train = y[train_idx]
        X_train = X[train_idx]
        X_user_train = X_user[train_idx]
        X_user_idx_train = X_user_idx[train_idx]
        X_collab_train = X_collab[train_idx]
        #X_user_feature_train = X_user_feature[train_idx]

        y_valid = y[valid_index]
        X_valid = X[valid_index]
        X_user_valid = X_user[valid_index]
        X_user_idx_valid = X_user_idx[valid_index]
        X_collab_valid = X_collab[valid_index]

        y_test = y[test_idx]
        X_test = X[test_idx]


        X_user_test = X_user[test_idx]
        X_user_idx_test = X_user_idx[test_idx]
        X_collab_test = X_collab[test_idx]

        # >> Split input into Train & Valid & Test

        if task == 'Disruptive':
            train_data = pd.DataFrame(zip(X_train, y_train), columns=["X", "target"])
            train_non_disruptive = train_data.loc[train_data.target == 0].X.tolist()
            train_disruptive = train_data.loc[train_data.target == 1].X.tolist()

            print("  - # Non-Disruptive = {}, # Disruptive = {}".format(len(train_non_disruptive), len(train_disruptive)))

        X_train = np.array(X_train)
        X_user_train = np.array(X_user_train)
        X_user_idx_train = np.array(X_user_idx_train)
        X_collab_train = np.array(X_collab_train)
        #X_user_feature_train = np.array(X_user_feature_train)
        y_train = np.array(y_train)
        y_train_lstm = tf.keras.utils.to_categorical(y_train, num_classes=2)

        X_valid = np.array(X_valid)
        X_user_valid = np.array(X_user_valid)
        X_user_idx_valid = np.array(X_user_idx_valid)
        X_collab_valid = np.array(X_collab_valid)
        #X_user_feature_train = np.array(X_user_feature_train)
        y_valid = np.array(y_valid)
        y_valid_lstm = tf.keras.utils.to_categorical(y_valid, num_classes=2)

        X_test = np.array(X_test)
        X_user_test = np.array(X_user_test)
        X_user_idx_test = np.array(X_user_idx_test)
        X_collab_test = np.array(X_collab_test)

        #X_user_feature_test = np.array(X_user_feature_test)
        y_test = np.array(y_test)
        y_test_lstm = tf.keras.utils.to_categorical(y_test, num_classes=2)

        print("\n 2) Displaying shapes >>")
        print("  - X_train.shape = {}".format(X_train.shape))
        print("  - X_valid.shape = {}".format(X_valid.shape))
        print("  - X_test.shape = {}".format(X_test.shape))
        print("  - y_train.shape = {}".format(y_train.shape))
        print("  - y_valid.shape = {}".format(y_valid.shape))
        print("  - y_test.shape = {}".format(y_test.shape))

        train_pred = []
        train_actual = []
        test_pred = []
        test_pred_prob = []
        test_actual = []

        # Total standard error
        accuracy = 0

        # >> Pad & Reshape inputs
        max_length = max(max([len(each) for each in X_train]), max([len(each) for each in X_train]),
                         max([len(each) for each in X_test]))

        print("\n 3) Padding and Reshaping >>")  # Input vectors are in the same length if used Bert, or PCA
        print("  - max_length: {}".format(max_length))

        print("*********************************************************X_train_shape : ", X_train.shape)


        X_train = sequence.pad_sequences(X_train, maxlen=max_length, truncating='pre')
        X_user_train = sequence.pad_sequences(X_user_train, maxlen=max_length, truncating='pre')
        X_user_idx_train = sequence.pad_sequences(X_user_idx_train, maxlen=max_length, truncating='pre')
        X_valid = sequence.pad_sequences(X_valid, maxlen=max_length, truncating='pre')
        X_user_valid = sequence.pad_sequences(X_user_valid, maxlen=max_length, truncating='pre')
        X_user_idx_valid= sequence.pad_sequences(X_user_idx_valid, maxlen=max_length, truncating='pre')
        X_test = sequence.pad_sequences(X_test, maxlen=max_length, truncating='pre')
        X_user_test = sequence.pad_sequences(X_user_test, maxlen=max_length, truncating='pre')
        X_user_idx_test = sequence.pad_sequences(X_user_idx_test, maxlen=max_length, truncating='pre')


        print("*********************************************************X_train_shape : ", X_train.shape)
        X_train = X_train.astype('float32')
        X_user_train = X_user_train.astype('float32')
        X_collab_train = X_collab_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_user_valid = X_user_valid.astype('float32')
        X_collab_valid = X_collab_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_user_test = X_user_test.astype('float32')
        X_collab_test = X_collab_test.astype('float32')


        print("*********************************************************X_train_shape : ", X_train.shape)

        # ''' Deep learning Model '''
        if opt_model == 'lstm:

            print("\n  Result >")
            print("   - X_train.shape = {}".format(X_train.shape))
            print("   - X_test.shape = {}".format(X_test.shape))
            print("   - y_train.shape = {}".format(y_train.shape))
            print("   - y_test.shape = {}".format(y_test.shape))

            print("\n 4) Setting-up a model")

            ################################################################################################################
            # User-focused
            ###############################################################################################################
            if user:

                ###Group###
                ###user###
                input2 = Input(shape=(max_length, X_user_train.shape[2]))
                x2 = LSTM(hidden, recurrent_dropout=dr)(input2)
                input1 = Input(shape=(max_length, X_train.shape[2]))

                if attention_type:
                    ## Attention
                    activations = LSTM(hidden, recurrent_dropout=dr, return_sequences=True)(input1)
                    attention = Dot(axes=(1, 2))([x2, activations])
                    attention = tf.keras.layers.Activation('softmax')(attention)
                    attention = tf.keras.layers.RepeatVector(hidden)(attention)
                    attention = tf.keras.layers.Permute([2, 1])(attention)
                    #
                    sent_representation = tf.keras.layers.Multiply()([activations, attention])
                    x1 = tf.keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(hidden,))(sent_representation)

                x = Concatenate(axis=1)([x1, x2])
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=[input1, input2], outputs=output)

                model.summary()

                if seqdrop:
                    optimizer = keras.optimizers.Adam()
                    loss_fn = keras.losses.BinaryCrossentropy()
                    min_val_loss = 1
                    from_last_min = 0

                    for epoch in range(epochs):
                        print("\nStart of epoch %d" % (epoch,))

                        X_train_seqdrop = []
                        #### Random Dropout without considering the current speaker#####
                        for idx, datapoint in enumerate(X_train):
                            ##### Random Dropout without considering the current speaker#####

                            target_user = set(X_user_idx[idx])

                            if seqdrop_rate != 0:
                                seq_len = round((max_length - 1) * seqdrop_rate)
                            else:
                                seq_len = round((max_length - 1) * random.uniform(0, 1))
                            #dropout_rate = (max_length - 1)
                            s = set(np.random.choice(max_length - 1, seq_len, replace=False))
                            s.add(max_length - 1)
                            s.union(target_user)
                            s = list(s)
                            datapoint = datapoint[s, :]
                            X_train_seqdrop.append(datapoint)

                        X_train_seqdrop = sequence.pad_sequences(X_train_seqdrop, maxlen=max_length, truncating='pre')

                        print(
                            "----------------------------------------------dropout rate at epoch {} : {}".format(epoch,
                                                                                                                 seq_len))
                        X_train = X_train_seqdrop
                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_user_train, y_train))

                        # Prepare the training dataset.
                        train_dataset = train_dataset.batch(batch)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        for step, (x_batch_train, x_batch_user_train, y_batch_train) in enumerate(train_dataset):
                            with tf.GradientTape() as tape:
                                # Run the forward pass of the layer.
                                # The operations that the layer applies
                                # to its inputs are going to be recorded
                                # on the GradientTape.
                                logits = model([x_batch_train, x_batch_user_train], training=True)  # Logits for this minibatch

                                # Compute the loss value for this minibatch.
                                loss_value = loss_fn(y_batch_train, logits)

                            # Use the gradient tape to automatically retrieve
                            # the gradients of the trainable variables with respect to the loss.
                            grads = tape.gradient(loss_value, model.trainable_weights)

                            # Run one step of gradient descent by updating
                            # the value of the variables to minimize the loss.
                            optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        print("Training loss for epoch {} : {}".format(epoch, loss_value))

                        ###Validation loss ###
                        val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
                        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, X_user_valid, y_valid))
                        valid_dataset = valid_dataset.batch(batch)

                        for x_batch_val, x_batch_user_val, y_batch_val in valid_dataset:
                            val_output = model([x_batch_val, x_batch_user_val], training=False)
                            val_loss_metric.update_state(y_batch_val, val_output)
                        val_loss = val_loss_metric.result()
                        val_loss_metric.reset_states()

                        print("Validation loss for epoch {} : {}".format(epoch, val_loss))
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            ###save model
                            model.save('./manual_earlystopping_model_mode{}_dr{}_seqdrop_user.h5'.format(dr, seqdrop_rate))
                            from_last_min = 0
                        else:
                            from_last_min += 1
                            if from_last_min == 5:
                                print(
                                    "-------------------------------No improvement during last 5 epoch. Early Stopping")
                                model = keras.models.load_model('./manual_earlystopping_model_mode{}_dr{}_seqdrop_user.h5'.format(dr, seqdrop_rate))
                                break
                else:
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='auto', patience=5, restore_best_weights=True)

                    print("\n 5) Training the model")

                    history = model.fit([X_train, X_user_train], y_train, epochs=epochs, batch_size=batch, callbacks=[early_stopping],
                            validation_data=([X_valid,X_user_valid], y_valid))

                # To save the model trained on full sequence for the sequence dropout approach


                print("\n 6) Prediction classes with the trained model")
                predict = model.predict([X_train, X_user_train])
                for each in predict:
                    train_pred.append(1 if each > tr else 0)

                predict = model.predict([X_test, X_user_test])
                for each in predict:
                     test_pred.append(1 if each > tr else 0)

                predict_prob = model.predict([X_test, X_user_test])

            ################################################################################################################
            # Group-only
            ###############################################################################################################
            else:
                input1 = Input(shape=(max_length, X_train.shape[2]))
                #input_last = Input(shape=(X_last_train.shape[1], ))
                if opt_model =='lstm':
                    x = LSTM(hidden, recurrent_dropout=dr)(input1)
                elif opt_model == 'gru':
                    x = GRU(hidden, recurrent_dropout=dr)(input1)

                if collab:
                    input2 = Input(shape=(X_collab_train.shape[1]))
                    x2 = Dense(X_collab_train.shape[1])(input2)
                    x = Concatenate(axis=1)([x, x2])
                    
                output = Dense(1, activation='sigmoid')(x)

                if collab:
                    model = Model(inputs=[input1, input2], outputs=output)
                else:
                    model = Model(inputs=[input1], outputs=output)

                model.summary()

                ##### for seqence drop#####
                """
                Sequence dropout (or Denosing) 
                 - mode 1: random 
                 - mode 2: augment within the loop
                 - mode 3: augment dataset before training
                """

                if seqdrop:
                    optimizer = keras.optimizers.Adam()
                    loss_fn = keras.losses.BinaryCrossentropy()
                    min_val_loss = 1
                    from_last_min = 0

                    for epoch in range(epochs):
                        print("\nStart of epoch %d" % (epoch,))

                        X_train_seqdrop = []
                        #### Random Dropout without considering the current speaker#####
                        for idx, datapoint in enumerate(X_train):
                            ##### Random Dropout without considering the current speaker#####

                            target_user = set(X_user_idx[idx])

                            if seqdrop_rate != 0:
                                seq_len = round((max_length - 1) * seqdrop_rate)
                            else:
                                seq_len = round((max_length - 1) * random.uniform(0, 0.5))

                            s = set(np.random.choice(max_length - 1, seq_len, replace=False))
                            s.add(max_length - 1)
                            s.union(target_user)
                            s = list(s)
                            datapoint = datapoint[s, :]
                            X_train_seqdrop.append(datapoint)

                        X_train_seqdrop = sequence.pad_sequences(X_train_seqdrop, maxlen=max_length, truncating='pre')

                        print("----------------------------------------------dropout rate at epoch {} : {}".format(epoch, seq_len))

                        X_train = X_train_seqdrop
                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

                        # Prepare the training dataset.
                        train_dataset = train_dataset.batch(batch)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                            with tf.GradientTape() as tape:

                                # Run the forward pass of the layer.
                                # The operations that the layer applies
                                # to its inputs are going to be recorded
                                # on the GradientTape.
                                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                                # Compute the loss value for this minibatch.
                                loss_value = loss_fn(y_batch_train, logits)

                            # Use the gradient tape to automatically retrieve
                            # the gradients of the trainable variables with respect to the loss.
                            grads = tape.gradient(loss_value, model.trainable_weights)

                            # Run one step of gradient descent by updating
                            # the value of the variables to minimize the loss.
                            optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        print("Training loss for epoch {} : {}".format(epoch, loss_value))

                        ###Validation loss ###
                        val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
                        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
                        valid_dataset = valid_dataset.batch(batch)

                        for x_batch_val, y_batch_val in valid_dataset:
                            val_output = model(x_batch_val, training=False)
                            val_loss_metric.update_state(y_batch_val, val_output)
                        val_loss = val_loss_metric.result()
                        val_loss_metric.reset_states()

                        print("Validation loss for epoch {} : {}".format(epoch, val_loss))
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            ###save model
                            model.save('./manual_earlystopping_model_mode{}_dr{}_seqdrop.h5'.format(dr, seqdrop_rate))
                            from_last_min = 0
                        else:
                            from_last_min += 1
                            if from_last_min == 5:
                                print("-------------------------------No improvement during last 5 epoch. Early Stopping")
                                model = keras.models.load_model('./manual_earlystopping_model_mode{}_dr{}_seqdrop.h5'.format(dr, seqdrop_rate))
                                break

                else:
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='auto', patience=5, restore_best_weights=True)

                    print("\n 5) Training the model")

                    if collab:
                        model.fit([X_train, X_collab_train], y_train, epochs=epochs, batch_size=batch, callbacks=[early_stopping],
                                  validation_data=([X_valid, X_collab_valid], y_valid))
                    else:
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, callbacks=[early_stopping],
                            validation_data=(X_valid, y_valid))


                print("\n 6) Prediction classes with the trained model")
                #predict = np.argmax(model.predict(X_train), axis=-1)

                if collab:
                    predict = model.predict([X_train, X_collab_train])
                else:
                    predict = model.predict(X_train)
                for each in predict:
                     train_pred.append(1 if each > tr else 0)
                train_actual.extend(y_train)

                #predict = np.argmax(model.predict(X_test), axis=-1)

                if collab:
                    predict = model.predict([X_test, X_collab_test])
                else:
                    predict = model.predict(X_test)

                for each in predict:
                     test_pred.append(1 if each > tr else 0)
                #test_pred.extend(predict)

                if collab:
                    predict_prob = model.predict([X_test, X_collab_test])
                else:
                    predict_prob = model.predict(X_test)

            for each in predict_prob:
                test_pred_prob.append(each)

            test_actual.extend(y_test)

            # ''' Static Model '''
        elif opt_model in classic_ml:

            X_train = np.reshape(X_train, (X_train.shape[0], -1))
            X_test = np.reshape(X_test, (X_test.shape[0], -1))

            print("\n  Result >")
            print("   - X_train.shape = {}".format(X_train.shape))
            print("   - X_test.shape = {}".format(X_test.shape))
            print("   - y_train.shape = {}".format(y_train.shape))
            print("   - y_test.shape = {}".format(y_test.shape))

            print("\n 4) Setting-up a model")
            if opt_model == 'lr':
                model = LogisticRegression()
            elif opt_model == 'gaussian_nb':
                model = GaussianNB()
            elif opt_model == 'multinomial_nb':
                model = MultinomialNB()
            elif opt_model == 'svm':
                model = svm.SVC(probability=True)
            elif opt_model == 'rf':
                model = RF(max_depth=4, random_state=1000)


            print("\n 5) Training the model")
            model.fit(X_train, y_train)

            print("\n 6) Prediction classes with the trained model")
            predict = model.predict(X_train)
            for each in predict:
                train_pred.append(each)
            train_actual.extend(y_train)

            predict = model.predict(X_test)

            # reversefactor = dict(zip(range(3), definitions))
            test_actual = y_test
            test_pred = predict

            predict_prob = model.predict(X_test)

            for each in predict_prob:
                test_pred_prob.append(each)

        print("\n-----------------------------")
        print("Prediction result >>")
        print("-----------------------------")
        print(" - Training Accuracy: {}".format(accuracy_score(y_train, train_pred)))

        # print("accuracy, recall, f1_score, precision, auc")
        accuracy = accuracy_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f_measure = f1_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred)
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, test_pred_prob)

        precisions_all_folds.append(precisions)
        recalls_all_folds.append(recalls)
        thresholds_all_folds.append(thresholds)

        pr_auc = metrics.auc(recalls, precisions)
        auc = metrics.roc_auc_score(y_test, test_pred_prob, average='macro', multi_class='ovr')


        d_model = DummyClassifier(strategy='stratified')
        d_model.fit(X_train, y_train)
        yhat = d_model.predict_proba(X_test)
        d_pos_probs = yhat[:, 1]
        # calculate the precision-recall auc
        d_precision, d_recall, _ = metrics.precision_recall_curve(y_test, d_pos_probs)
        d_auc_score = metrics.auc(d_recall, d_precision)
        print('No Skill PR AUC: %.3f' % d_auc_score)

        print("\n - Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        print("\n - Counter(test_actual): {}".format(Counter(y_test)))
        print(" - Counter(test_pred): {}".format(Counter(test_pred)))

        performances.loc[i] = [str(fold_idx + 1), task, opt_model, input, str(context),
                               accuracy, precision, recall, f_measure, auc, pr_auc]
        print(performances)
        i += 1

    if seqdrop_rate != 0:
        performances.to_csv(
            '/home/kpark8/Desktop/dissertation/data/result/Linguistic_0529/fixed_seqdrop_rate/seed={}_dr={}_user={}_collab={}_drrate_{}_seqdroprate{}_{}_{}_{}_{}_iter{}.csv'.format(
                seed, seqdrop, user, collab, dr, seqdrop_rate, task, opt_model, input, context, count+1), index=False)
    else:
        performances.to_csv(
            '/home/kpark8/Desktop/dissertation/data/result/Linguistic_0529/seed={}_dr={}_user={}_collab={}_drrate_{}_{}_{}_{}_{}_iter{}.csv'.format(
                seed, seqdrop, user, collab, dr, task, opt_model, input, context, count+1), index=False)

    averaged = performances.mean(axis=0)

    print("Accuracy, Precision, Recall, Fmeasure, AUC, PR_AUC")

    print("{}, {}, {}, {}, {}, {}".format(averaged['Accuracy'], averaged['Precision'], averaged['Recall'], averaged['Fmeasure'], averaged['AUC'], averaged['PR_AUC']))

    # performances.to_csv(
    #     '/home/kpark8/Desktop/dissertation/data/result/{}_input={}_ctx={}_pre={}.csv'.format(
    #         opt_model,input, context, preprocessing), sep='\t')

    return averaged['Accuracy'], averaged['Precision'], averaged['Recall'], averaged['Fmeasure'], averaged['AUC'], averaged['PR_AUC']



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, help='lstm | bilstm | gru | crf | rf | lr')
    parser.add_argument('--user', default=None, help='True for user-focused, None for group only')
    parser.add_argument('--seqdrop', default=None, help='True for sequence dropout, None for without')
    parser.add_argument('--attention', default=None, help='True for user-aware attention, None for without')
    parser.add_argument('--dr', default=0, type=float, help='True for sequence dropout, None for without')
    parser.add_argument('--seqdrop_rate', default=0, type=float, help='True for sequence dropout, None for without')

    parser.add_argument('--collab', default=None, help='True for collaborative features, None for without the collab feaures')
    parser.add_argument('--input', required=True, help='bert | bow | w2v')
    parser.add_argument('--num_repeat', default = 1, type=int)

    parser.add_argument('--context', type=int, default=20)
    parser.add_argument('--gender', default=None, help='whether gender is included as feature')
    parser.add_argument('--pretest', default=None, help='whether pretest is included as feature')
    parser.add_argument('--bert_type', default='distilbert', help='select pretrained bert model')


    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--preprocessing', default='cleaned_with_meaningless')

    parser.add_argument('--pca', type=int, default=50)
    parser.add_argument('--features', default='linguistic')

    opt = parser.parse_args()

    # if opt.experiment is None:
    #     opt.experiment = 'samples'
    # os.system('mkdir {}'.format(opt.experiment))


    dic = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': [], 'PR-AUC': []}



    if opt.user == None:
        type = 'group'
    else:
        type = 'user'

    for count in range(opt.num_repeat):
        if opt.seqdrop_rate != 0:
            file_path = 'result/fixed_seqdrop_rate/seed={}_dr={}_user={}_collab={}.csv'.format(
                opt.seed, opt.seqdrop, opt.user, opt.collab)
        else:
            file_path = 'result/seed={}_dr={}_user={}_collab={}.csv'.format(
                opt.seed, opt.seqdrop, opt.user, opt.collab)

        acc_avg, prec_avg, recall_avg, f1_avg, auc_avg, prauc_avg \
            = prediction_model(opt.seed, opt.model, opt.input, opt.context,
                               opt.gender, opt.pretest, opt.hidden, opt.epoch, opt.batch,
                               opt.preprocessing, opt.pca, opt.features, opt.user, opt.seqdrop, opt.collab, count, opt.dr, opt.seqdrop_rate, opt.attention)

        dic['Accuracy'].append(acc_avg)
        dic['Precision'].append(prec_avg)
        dic['Recall'].append(recall_avg)
        dic['F1'].append(f1_avg)
        dic['AUC'].append(auc_avg)
        dic['PR-AUC'].append(prauc_avg)

        if os.path.isfile(file_path):
            df = pd.DataFrame(pd.read_csv(file_path))
        else:
            columns = ['Seed', 'User', 'SeqDropout','Collab', 'Model', 'Input', 'Context',
                       'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'PR-AUC']
            df = pd.DataFrame(pd.DataFrame(columns=columns))


        result = {'Seed': opt.seed, 'User': opt.user, 'SeqDropout': opt.seqdrop, 'collab': opt.collab, 'Model': opt.model, 'Input': opt.input,
                      'Context': opt.context,'Accuracy': acc_avg, 'Precision': prec_avg,
                      'Recall':recall_avg, 'F1': f1_avg, 'AUC': auc_avg, 'PR-AUC': prauc_avg}


        df = df.append(result, ignore_index=True)
        df.to_csv(file_path, index=False)

    print("\nFinal Result >>")
    print("User\tModel\tInput\tContext\tAccuracy\tPrecision\tRecall\tF-measure\tAUC\tPR-AUC")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(opt.user, opt.model, opt.input, opt.context,
                                                      np.average(dic['Accuracy']), np.average(dic['Precision']),
                                                      np.average(dic['Recall']),  np.average(dic['F1']),  np.average(dic['AUC']), np.average(dic['PR-AUC'])))

