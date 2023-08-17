# =========================================================================
# Feature Engineering (for BERT)
# 1. Jaccard Similarity
# 2. Sentiment
# 3. Gender
# 4. Pretest Score
# ========================================================================

import string
import datetime
import os
import pickle

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import * 
from sklearn.feature_extraction.text import CountVectorizer


class FeaturePreprocessing:

    def __init__(self, input, preprocessing, pca, features):

        self.features = features
        self.input = input
        self.preprocessing = preprocessing
        self.pca = pca

        # import pre-trained BERT model and tokenizer

        if input == 'wiki-bert':
            # Doesn't need the preprocessing step
            self.model_class, self.tokenizer_class, self.pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')

            # Load pretrained model/tokenizer
            self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
            self.model = self.model_class.from_pretrained(self.pretrained_weights)

        elif input == 'dialogue-bert':
            self.tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
            self.model = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")

        elif input == 'distilbert':
            self.model_class, self.tokenizer_class, self.pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')

            # Load pretrained model/tokenizer
            self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
            self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def bow_embedding(self, data):

        vectorizer = CountVectorizer()

        train = data.Message.values
        vectorizer.fit(train)

        vectorized_data = vectorizer.transform(data.Message.values)

        np.save('data/{}_emb_pre={}.npy'.format(self.input, self.preprocessing), vectorized_data.toarray())

        return vectorized_data.toarray()


    def averaging(self, word_list, model):
        """ Averaging the word embeddings to generate the centroid (sentence) vector"""

        # oov_tokens = set()

        # print("Averaging words")
        avg_vec = np.zeros(300)
        word_count = 0

        for word in word_list:
            if word in model:
                avg_vec += model[word]
                word_count += 1
            #else:
                # print("[{}] not exist!".format(word))
                #oov_tokens.add(word)

        if word_count:
            avg_vec = np.divide(avg_vec, word_count)

        # print(oov_tokens)
        # np.save('oov_tokens.txt')

        return avg_vec

    def w2v_embedding(self, data):
        print("\nPreparing X & y for Word2Vec")

        # >> Loading a pre-trained Word2Vec model
        from gensim.models import KeyedVectors
        w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',
                                                      binary=True)

        X = []

        for msg_lst in data.Message.tolist():
            # grp_cnt = 0
            sub_X = []
            for msg in msg_lst:
                emb = self.averaging(msg, w2v_model).tolist()
                X.append(emb)

        np.save('data/{}_emb_pre={}.npy'.format(self.input, self.preprocessing),
                np.array(X))


        return X

    def bert_embedding(self, data):

        tokenized = data.Message.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

        tokenized_word_list = data.Message.apply((lambda x: self.tokenizer.tokenize(x)))
        data["Tokens"] = tokenized_word_list
        data["Token numbers"] = tokenized

        data.to_csv('data/{}_tokens_pre={}.csv'.format(self.input, self.preprocessing))

        ## Padding the data

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        # Pad the data
        padded = np.array([i + [0] * (16 - len(i)) if len(i) < 16 else i[:16] for i in tokenized.values])

        ## Masking
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape

        ## BERT Embedding

        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # save the outputs (CLS-token only)

        features = last_hidden_states[0][:, 1:, :].numpy().mean(axis=1)
        np.save('data/{}_emb_pre={}.npy'.format(self.input, self.preprocessing),
                features)

        return features

    # Sentiment of message
    def get_sentiment(self, msg):
        ''' Generate one-hot vector for sentiment '''

        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        sentiment_result = sia.polarity_scores(msg)
        sentiment_score = sentiment_result['compound']

        # one-hot encoding for sentiment (0: positive, 1: neutral, 2: negative)
        sentiment_vector = np.zeros(3)

        if sentiment_score >= 0.05:
            sentiment_vector[0] = 1
        elif sentiment_score > -0.05:
            sentiment_vector[1] = 1
        else:
            sentiment_vector[2] = 1

        return sentiment_vector


    # Add Jaccard Similarity

    def jaccard_similarity(self, doc1, doc2):
        words_doc1 = set(doc1.lower().split())
        words_doc2 = set(doc2.lower().split())

        intersection = words_doc1.intersection(words_doc2)
        union = words_doc1.union(words_doc2)

        return float(len(intersection)) / len(union)

    def normalize_column(self, A, col):
        A[:, col] = (A[:, col] - np.min(A[:, col])) / (np.max(A[:, col]) - np.min(A[:, col]))
        return A

    def concat_data(self, data, opt_context, X_concated):
        # =========================================================================
        # Concat context window messages (context window = 5, 20)
        # For bow and w2v, this process was done before encode to a word representation,
        # For BERT: This process was done after BERT embedding
        # ========================================================================

        X = []  # concated data
        X_collab = [] # collab-related feature
        X_user = [] #concated user-specific data
        X_user_idx = []
        y = []  # labels
        y_disruptive = []
        y_topic = []
        groups = [] # groups -- for Group-level k-fold
        wizard_cnt = 0
        # data = data.reset_index()
        group_num = pd.unique(data['GroupID'])

        for group in group_num:

            data_group = data[data['GroupID'] == group]
            idx_numbers = data_group.index
            dates = pd.Series(data_group['Date']).unique()

            i = 0
            for date in dates:
                data_group_per_date = data_group[data_group['Date'] == date]
                # data_group_per_date = data_group_per_date.reset_index()
                first_row = 0
                for index, row in data_group_per_date.iterrows():
                    if first_row == 0:
                        first_row = index

                    user = row['UserID']

                    X_lst = []
                    X_user_lst = []
                    X_user_idx_lst = [] ## to retain the current user's history message
                    X_lst_collab = [] #### index 0: volume of the target user's talk 1: Equity 2: Time spent
                    user_idx = 0

                    ##length
                    upper = index + 1
                    lower = np.maximum(first_row, index - opt_context)
                    seq_len = upper-lower
                    num_msg_from_user = 0
                    user_dict = {"Jeepney":0, "Turtle":0, "Eagle":0, "Sun":0}

                    for prev_idx in range(np.maximum(first_row, index - opt_context), index + 1):

                        X_lst.append(X_concated[prev_idx])

                        user_name = data.loc[prev_idx]['UserID'][:-3]
                        if (user_name != "Wizard") and (user_name != "Helper"):
                            user_dict[user_name] += 1

                        if data.loc[prev_idx]['UserID'] == user:
                            num_msg_from_user += 1
                            X_user_lst.append(X_concated[prev_idx])
                            X_user_idx_lst.append(user_idx)

                        user_idx += 1

                    vol_of_talk = num_msg_from_user/seq_len
                    time_spent = (data.loc[index]["TimeStamp"] - data.loc[lower]["TimeStamp"]).total_seconds()
                    equity = np.var([val for val in user_dict.values()])

                    X_lst_collab.append(vol_of_talk)
                    X_lst_collab.append(time_spent)
                    X_lst_collab.append(equity)

                    X_collab.append(X_lst_collab)

                    X.append(X_lst)
                    X_user.append(X_user_lst)
                    X_user_idx.append(X_user_idx_lst)
                    y.append(row['Disruptive_Label'])
                    groups.append(row['GroupID'])
        X_collab = np.array(X_collab, dtype=float)

        X_collab = self.normalize_column(X_collab, 0)
        X_collab = self.normalize_column(X_collab, 1)
        X_collab = self.normalize_column(X_collab, 2)

        return X, X_user,X_user_idx, y, y_disruptive, y_topic, groups, X_collab

    def no_context(self, data, X_concated):
        # =========================================================================
        # Concat context window messages (context window = 5, 20)
        # For bow and w2v, this process was done before encode to a word representation,
        # For BERT: This process was done after BERT embedding
        # ========================================================================

        X = []  # concated data
        y = []  # labels
        groups = [] # groups -- for Group-level k-fold
        wizard_cnt = 0
        # data = data.reset_index()
        group_num = pd.unique(new['GroupID'])

        for group in group_num:

            data_group = data[data['GroupID'] == group]
            idx_numbers = data_group.index
            dates = pd.Series(data_group['Date']).unique()

            i = 0
            for date in dates:
                data_group_per_date = data_group[data_group['Date'] == date]
                # data_group_per_date = data_group_per_date.reset_index()
                first_row = 0
                for index, row in data_group_per_date.iterrows():
                    if first_row == 0:
                        first_row = index
                    # index = the original index number
                    # row = all data

                    # if row['Player'] == 'wizard':
                    #     wizard_cnt += 1
                    #     continue

                    user = row['UserID']

                    X_lst = []
                    X_user_lst = []

                    for prev_idx in range(np.maximum(first_row, index - opt_context), index + 1):
                        X_lst.append(X_concated[prev_idx])
                        if data.loc[prev_idx]['UserID'] == user:
                            X_user_lst.append(X_concated[prev_idx])

                    X.append(X_lst)
                    X_user.append(X_user_lst)
                    y.append(row['Disruptive_Label'])
                    groups.append(row['GroupID'])
        return X, X_user, y, groups

    def create_data(self, opt_context = None, opt_gender = None, opt_pretest = None, collab=None):

        if self.preprocessing:
            data = pd.read_csv('data/dissertation/data_combined_dissertation_action_units_linguistic_features.csv', parse_dates=["TimeStamp"])
        else:
            data = pd.read_csv('data/epistemic/chat_master.csv', parse_dates=["TimeStamp"])


        #if ('bert' in self.input) or (self.input == 'bow'):
        emb_path = 'data/{}_emb_pre={}.npy'.format(self.input, self.preprocessing)

        if os.path.isfile(emb_path):
            emb = np.load(emb_path)
        else:
            if 'bert' in self.input:
                emb = self.bert_embedding(data)
            elif self.input == 'bow':
                emb = self.bow_embedding(data)
            elif self.input == 'w2v':
                emb = self.w2v_embedding(data)
        '''
        ADD other features
        1. Jaccard
        2. Sentiment
        3. Gender
        4. Pretest
        '''

        file = open('data/GameText.txt', 'r')
        game_text = " ".join(file.read().splitlines())
        game_text = " ".join(set([w for w in word_tokenize(game_text) if w not in string.punctuation]))

        X_sentiment = []
        for message in data.Message.values:
            temp_msg = " ".join(set([w for w in word_tokenize(message) if w not in string.punctuation]))
            X_sentiment.append(self.get_sentiment(temp_msg))

        X_jaccard = []

        for message in data.Message.values:
            temp_msg = " ".join(set([w for w in word_tokenize(message) if w not in string.punctuation]))
            X_jaccard.append(self.jaccard_similarity(game_text, temp_msg))

        # Add length of each message

        X_charlen = []
        for message in data.Message.values:
            X_charlen.append(len([char for char in list(message) if char not in ' ']))

        #>> Create a one-hot vector for the speaker
        print("\nPreprocessing Scene information >")

        '''
        ADD game interaction features if collab==True
        1. Scene -- using "Scene_Label"
        2. Action_units
        '''
        if collab:
            ###Scene
            scenes = data["Scene_Label"].values
            X_scene = []

            # sp_dict = {'m': 0, 'd': 1, 'f': 2, 'j': 3, 'wizard': 4}
            for scene in scenes:
                temp = np.zeros(len(data["Scene_Label"].unique()))
                temp[scene] = 1

                X_scene.append(temp)

            ###Action units###
            action_units = ['Started', 'DisconnectedFrom', 'ConnectedTo',
       'ReceivedChatMessage', 'Opened', 'SentChatMessage', 'Closed',
       'SpokeTo', 'MovedTo', 'Ended', 'Activated', 'MovedFrom',
       'ChangedStateOf', 'Created', 'ChangedColumn', 'Moved', 'Voted',
       'Submitted', 'Deleted', 'Revealed', 'Selected']

            actions = data[action_units]

        # >> Create a one-hot vector for Gender

        print("\nPreprocessing Gender information >")

        genders = data.Gender.values
        X_gender = []

        gd_dict = {'M': 0, 'F': 1, 'W': 2}
        for gender in genders:
            temp = np.zeros(len(gd_dict))
            if not gender or pd.isnull(gender):
                temp[gd_dict['W']] = 1
            else:
                temp[gd_dict[gender]] = 1

            X_gender.append(temp)

        print(" - len(X_gender) = {}".format(len(X_gender)))

        # >> Create a one-hot vector for Pretest

        print("\nPreprocessing Pretest information >")

        pretests = data.Pretest.values
        X_pretest = []

        pt_dict = {'Low': 0, 'Medium': 1, 'High': 2, "Wizard": 3}
        for pretest in pretests:
            temp = np.zeros(len(pt_dict))
            if not pretest or pd.isnull(pretest):
                temp[pt_dict['Wizard']] = 1
            else:
                temp[pt_dict[pretest]] = 1

            X_pretest.append(temp)

        print(" - len(X_pretest) = {}".format(len(X_pretest)))

        X_concated = []
        X_user_feature = []
        for row in range(len(data.Message)):
            sub_X = []
            sub_X.extend(emb[row])
            sub_X.append(X_jaccard[row])
            sub_X.append(X_charlen[row])
            sub_X.extend(X_sentiment[row])
            sub_X.extend(X_gender[row])
            sub_X.extend(X_pretest[row])
            if self.features == 'linguistic_additional':
                print("******************Added additional linguistic features")
                sub_X.extend(data[["IsUpper", "IsElongated"]].iloc[row].values)
            if collab:
                sub_X.extend(X_scene[row])
                sub_X.extend(actions.iloc[row].values)

            X_concated.append(sub_X)

        if self.pca:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            scaler = StandardScaler()
            scaler.fit(X_concated)
            X_concated = scaler.transform(X_concated)

            pca = PCA(n_components=self.pca)
            pca.fit(X_concated)
            X_concated = pca.transform(X_concated)

        X, X_user, X_user_idx, y, y_disruptive, y_topic, groups, X_collab = self.concat_data(data, opt_context, X_concated)


        pickle.dump(X, open(
            "data/{}_X_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(self.input, opt_context, self.preprocessing, self.pca, self.features, collab), 'wb'), -1)
        pickle.dump(X_user, open(
            "data/{}_User_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(self.input, opt_context, self.preprocessing, self.pca, self.features, collab), 'wb'), -1)
        pickle.dump(X_user_idx, open(
            "data/_{}_User_idx_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(self.input,
                                                                                                       opt_context,
                                                                                                       self.preprocessing,
                                                                                                       self.pca,
                                                                                                       self.features, collab),
            'wb'), -1)

        pickle.dump(X_user_feature, open(
            "/data/{}_User_Feature_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(self.input, opt_context, self.preprocessing, self.pca, self.features, collab), 'wb'), -1)
        pickle.dump(X_collab, open(
            "/data/{}_Collab_Feature_context={}_pre={}_pca={}_{}_collab_{}.pkl".format(
                self.input, opt_context, self.preprocessing, self.pca, self.features, collab), 'wb'), -1)

        pickle.dump(y, open(
        "/data/{}_Labels_context={}_pre={}_pca={}_{}.pkl".format(self.input, opt_context, self.preprocessing, self.pca, self.features), 'wb'), -1)
        pickle.dump(groups, open(
            "/data/{}_Groups_context={}_pre={}_pca={}_{}.pkl".format(self.input, opt_context, self.preprocessing, self.pca, self.features), 'wb'), -1)


        return X, X_user,X_user_idx, y, groups, X_user_feature, X_collab
