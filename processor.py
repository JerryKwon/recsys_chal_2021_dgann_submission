"""
Name: processor.py
Description: Overall processor of Neural Network Model(DAN, ANN)
by preprocess / train / predict method
"""

import platform
import os
from copy import deepcopy
from tqdm import tqdm
import random
from datetime import datetime
import pickle

from scipy import sparse as spr
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, WeightedRandomSampler

from utils import EarlyStopping, compute_rce
from model import CustomDAN, CustomANN
from custom_dataset import DANDataset, AnnDataset

class DanProcessor:
    def __init__(self, df_train, glove, model_path, verbose=True):
        self.df_train = df_train
        self.glove = glove
        self.model_path = model_path
        self.verbose = verbose

        self.embeddings = glove["word_vectors"]
        self.token_to_tid = glove["dictionary"]

        self.embeddings_w0 = self.get_embeddings_w0()

    def seed_initialization(self, GLOBAL_SEED):
        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_embeddings_w0(self):
        return np.vstack([np.zeros(self.embeddings.shape[1]), self.embeddings])

    def token_to_tid_count(self, target_str):
        result = 0
        for token in target_str.split(" "):
            if token in self.token_to_tid.keys():
                result += 1
        return result

    def preprocess(self, batch_size, GLOBAL_SEED=None):
        if GLOBAL_SEED is not None:
            self.seed_initialization(GLOBAL_SEED)

        df_train = self.df_train

        self.np_tid_token_len = df_train["text_tokens"].apply(self.token_to_tid_count).values

        added_values = np.fromiter(self.token_to_tid.values(), dtype=np.long) + 1
        token_to_tid_add = dict(zip(list(self.token_to_tid.keys()), added_values))

        rows = list()
        cols = list()
        datas = list()

        for idx, target_str in (enumerate(tqdm(df_train["text_tokens"].values, desc="csr matrix processing")) if self.verbose else enumerate(
                df_train["text_tokens"].values)):
            mid_datas = [token_to_tid_add[token] for token in target_str.split(" ") if
                         token in self.token_to_tid.keys()]
            mid_rows = np.repeat(idx, len(mid_datas))
            mid_cols = np.arange(len(mid_datas))

            rows.append(mid_rows)
            cols.append(mid_cols)
            datas.append(mid_datas)

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(datas)

        csr_matrix = spr.csr_matrix((data, (row, col)), shape=(df_train.shape[0], self.np_tid_token_len.max()))

        np_token = csr_matrix.toarray()

        dataloader = self.make_loader(np_token, batch_size)

        return dataloader

    def make_loader(self, np_token, BATCH_SIZE):

        dan_dataset = DANDataset(np_token, self.np_tid_token_len)

        test_dataloader = torch.utils.data.DataLoader(
            dan_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        return test_dataloader


    def train(self, dataloaders, target_column, epochs, learning_rate, model_path, patience=None):

        f_name = None

        if target_column == "reply_timestamp":
            f_name = "dan_reply_over.pth"
        elif target_column == "retweet_timestamp":
            f_name = "dan_retweet_over.pth"
        elif target_column == "retweet_with_comment_timestamp":
            f_name = "dan_retweet_c_over.pth"
        elif target_column == "like_timestamp":
            f_name = "dan_like.pth"

        is_valid = len(dataloaders) == 2

        if is_valid:
            trn_dataloader, val_dataloader = dataloaders

        else:
            trn_dataloader = dataloaders

        section = 3
        batch_prints = [int(len(trn_dataloader) / section * idx) for idx in range(1, section + 1)][:-1]

        es = None
        print(patience)
        if patience > 0:
            es = EarlyStopping(model_path, patience, "max")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        dan = CustomDAN(self.embeddings_w0, DEVICE)

        optimizer = optim.SGD(dan.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        if DEVICE == "cuda":
            dan = dan.to(DEVICE)

        for epoch in (tqdm(range(epochs), desc="Dan train processing") if self.verbose else range(epochs)):

            dan.train()

            trn_loss = 0.0
            trn_rce = 0.0
            trn_ap = 0.0

            for batch_idx, (inputs, len_idx, labels) in enumerate(trn_dataloader):
                dan.zero_grad()

                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                hidden, output = dan((inputs, len_idx))

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                trn_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                trn_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                trn_loss += loss.item()

                if self.verbose:
                    if (batch_idx + 1) in batch_prints:
                        print(
                            f"BATCH:{batch_idx + 1}:{len(trn_dataloader)}; loss:{trn_loss / (batch_idx + 1):.4f}; ap:{trn_ap / (batch_idx + 1):.4f}; rce:{trn_rce / (batch_idx + 1):.4f}")

            if not is_valid:
                dan_state_dict = deepcopy(dan.state_dict())
                torch.save(dan_state_dict, os.path.join(model_path, f_name))
                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}; ap:{trn_ap / len(trn_dataloader):.4f}; rce:{trn_rce / len(trn_dataloader):.4f}")

            else:
                val_loss = 0.0
                val_rce = 0.0
                val_ap = 0.0

                dan.eval()

                with torch.no_grad():
                    for batch_idx, (inputs, len_idx, labels) in enumerate(val_dataloader):

                        if DEVICE == "cuda":
                            inputs = inputs.to(DEVICE)
                            labels = labels.to(DEVICE)

                        hidden, output = dan((inputs, len_idx))
                        loss = criterion(output, labels)

                        val_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                        val_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                        val_loss += loss.item()

                        break

                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}/{val_loss:.4f}; ap:{trn_ap / len(trn_dataloader):.4f}/{val_ap}; rce:{trn_rce / len(trn_dataloader):.4f}/{val_rce:.4f}")

                if patience > 0:
                    es((val_ap), dan, f_name)

                if es.early_stop:
                    print("early_stopping")
                    break

    def predict(self, dataloader, target_column, model_path):

        result = []

        # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DEVICE = "cpu"
        criterion = nn.CrossEntropyLoss()

        dan = CustomDAN(self.embeddings_w0, DEVICE)

        f_name = None

        if target_column == "reply":
            f_name = "dan_reply_over.pth"
        elif target_column == "retweet":
            f_name = "dan_retweet_over.pth"
        elif target_column == "retweet_c":
            f_name = "dan_retweet_c_over.pth"
        elif target_column == "like":
            f_name = "dan_like.pth"

        if DEVICE == "cuda":
            dan.load_state_dict(torch.load(os.path.join(model_path,f_name)))
        else:
            dan.load_state_dict(torch.load(os.path.join(model_path, f_name), map_location=lambda storage, loc: storage))

        dan = dan.to(DEVICE)

        dan.eval()

        # val_loss = 0.0
        # val_rce = 0.0
        # val_ap = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, len_idx) in (
                    enumerate(tqdm(dataloader, desc="Dan inference processing")) if self.verbose else enumerate(dataloader)):

                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE)

                hidden, output = dan((inputs, len_idx))

                result.append(hidden.detach().cpu().numpy())

                # loss = criterion(output, labels)
                #
                # val_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                # val_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                # val_loss += loss.item()

        # if self.verbose:
        #     print(
        #         f"loss:{val_loss / len(dataloader):.4f}; ap:{val_ap / len(dataloader):.4f}; rce:{val_rce / len(dataloader):.4f}")

        result = np.vstack(result)

        return result

class AnnProcessor:
    def __init__(self, model_path, global_seed, verbose=True):
        self.model_path = model_path
        self.verbose = verbose
        self.seed_initialization(global_seed)

    def seed_initialization(self, GLOBAL_SEED):

        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_token_feature(self, df_train, glove, target_col, batch_size):
        processor = DanProcessor(df_train, glove, self.model_path, self.verbose)
        dataloader = processor.preprocess(batch_size, self.GLOBAL_SEED)
        emb_result = processor.predict(dataloader, target_col, self.model_path)

        return emb_result

    def preprocess(self, df_train, glove, batch_size, model_path):
        target_columns = ["reply", "retweet", "retweet_c", "like"]

        # self.df_train = df_train
        # self.glove = glove
        # self.token_model = token_model

        df_indexes = df_train[["tweet_id", "engaging_user_id"]]
        df_indexes = df_indexes.rename(columns={"engaging_user_id": "user_id"})

        data = df_train.copy()

        # df_indexes = data[["tweet_id","engaging_user_id"]]

        tweet_hour = []

        for tweet_time in data['tweet_timestamp'].astype(int).values:
            tweet_hour.append(datetime.fromtimestamp(tweet_time).strftime('%H'))

        data['engaged_time > engaging_time'] = ((pd.to_numeric(
            data['engaged_with_user_account_creation']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_time > engaging_time'] = ((pd.to_numeric(data['tweet_timestamp']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_hour'] = tweet_hour
        data['tweet_hour'] = data['tweet_hour'].astype(int)

        data.loc[data["engaged_time > engaging_time"] >= 0, "bool_(engaged_time > engaging_time)"] = 1
        data.loc[data["engaged_time > engaging_time"] < 0, "bool_(engaged_time > engaging_time)"] = 0

        data['num_GIF'] = data['present_media'].str.count('GIF')
        data['num_Video'] = data['present_media'].str.count('Video')
        data['num_Photo'] = data['present_media'].str.count('Photo')

        data['num_GIF'] = data['num_GIF'].fillna(0)
        data['num_Video'] = data['num_Video'].fillna(0)
        data['num_Photo'] = data['num_Photo'].fillna(0)

        data.loc[data["present_links"] != 0, "present_links"] = 1

        data_X = data.loc[:, ['tweet_type', 'engaged_with_user_is_verified', 'engaging_user_is_verified'
                                 , 'engagee_follows_engager', 'engaged_with_user_following_count',
                              'engaged_with_user_follower_count'
                                 , 'engaging_user_follower_count', 'engaging_user_following_count', 'present_links',
                              'engaged_time > engaging_time', 'tweet_time > engaging_time', 'tweet_hour', 'language',
                              'bool_(engaged_time > engaging_time)', 'num_GIF', 'num_Video', 'num_Photo']]

        data_X.loc[data["engaged_with_user_is_verified"] == 'false', "engaged_with_user_is_verified"] = 0
        data_X.loc[data["engaged_with_user_is_verified"] == 'true', "engaged_with_user_is_verified"] = 1
        data_X.loc[data["engaging_user_is_verified"] == 'false', "engaging_user_is_verified"] = 0
        data_X.loc[data["engaging_user_is_verified"] == 'true', "engaging_user_is_verified"] = 1
        data_X.loc[data["engagee_follows_engager"] == 'false', "engagee_follows_engager"] = 0
        data_X.loc[data["engagee_follows_engager"] == 'true', "engagee_follows_engager"] = 1

        data_X.loc[data_X["language"].str.contains('488B32D24BD4BB44172EB981C1BCA6FA', na=False), "language"] = '10'
        data_X.loc[data_X["language"].str.contains('E7F038DE3EAD397AEC9193686C911677', na=False), "language"] = '9'
        data_X.loc[data_X["language"].str.contains('B0FA488F2911701DD8EC5B1EA5E322D8', na=False), "language"] = '8'
        data_X.loc[data_X["language"].str.contains('B8B04128918BBF54E2E178BFF1ABA833', na=False), "language"] = '7'
        data_X.loc[data_X["language"].str.contains('313ECD3A1E5BB07406E4249475C2D6D6', na=False), "language"] = '6'
        data_X.loc[data_X["language"].str.contains('1F73BB863A39DB62B4A55B7E558DB1E8', na=False), "language"] = '5'
        data_X.loc[data_X["language"].str.contains('9FCF19233EAD65EA6E32C2E6DC03A444', na=False), "language"] = '4'
        data_X.loc[data_X["language"].str.contains('9A78FC330083E72BE0DD1EA92656F3B5', na=False), "language"] = '3'
        data_X.loc[data_X["language"].str.contains('8729EBF694C3DAF61208A209C2A542C8', na=False), "language"] = '2'
        data_X.loc[data_X["language"].str.contains('E6936751CBF4F921F7DE1AEF33A16ED0', na=False), "language"] = '1'
        data_X.loc[data_X["language"].str.len() > 3, "language"] = '0'

        data_X['engaged_with_user_following_count'] = data_X['engaged_with_user_following_count'].astype(int)
        data_X['engaged_with_user_follower_count'] = data_X['engaged_with_user_follower_count'].astype(int)
        data_X['engaging_user_follower_count'] = data_X['engaging_user_follower_count'].astype(int)
        data_X['engaging_user_following_count'] = data_X['engaging_user_following_count'].astype(int)

        data_X['engaged_time'] = (pd.to_numeric(data['engaged_with_user_account_creation']) / 1000000)

        data_X['engaging_time'] = (pd.to_numeric(data['engaging_user_account_creation']) / 1000000)
        data_X['tweet_timestamp'] = (pd.to_numeric(data['tweet_timestamp']) / 1000000)

        # data_X[data_X['engaging_user_following_count'] == 0] = 1
        # data_X[data_X['engaged_with_user_following_count'] == 0] = 1
        #
        # data_X[data_X['engaging_user_follower_count'] == 0] = 1
        # data_X[data_X['engaged_with_user_follower_count'] == 0] = 1

        data_X["engaging_user_following_count"][data_X['engaging_user_following_count'] == 0] = 1
        data_X["engaged_with_user_following_count"][data_X['engaged_with_user_following_count'] == 0] = 1

        data_X["engaging_user_follower_count"][data_X['engaging_user_follower_count'] == 0] = 1
        data_X["engaged_with_user_follower_count"][data_X['engaged_with_user_follower_count'] == 0] = 1

        data_X['ratio_enaging_follow'] = data_X['engaging_user_follower_count'] / data_X[
            'engaging_user_following_count']
        data_X['ratio_enagaged_follow'] = data_X['engaged_with_user_follower_count'] / data_X[
            'engaged_with_user_following_count']

        with open(os.path.join(model_path, "dict_label_encoder.pkl"), "rb") as f:
            dict_label_encoder = pickle.load(f)

        data_X["tweet_type"] = data_X["tweet_type"].apply(lambda x: dict_label_encoder[x])

        df_tweet_type = pd.DataFrame()
        for idx in range(len(dict_label_encoder)):
            series = (data_X.tweet_type == idx).astype("int")
            series.name = f"tweet_type_{idx}"
            df_tweet_type = pd.concat([df_tweet_type, series], axis=1)

        data_X = pd.concat([data_X, df_tweet_type], axis=1)

        sentence_vectors = ['sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2', 'sentenc_vector_3',
                            'sentenc_vector_4', 'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7'
            , 'sentenc_vector_8', 'sentenc_vector_9', 'sentenc_vector_10', 'sentenc_vector_11'
            , 'sentenc_vector_12', 'sentenc_vector_13', 'sentenc_vector_14', 'sentenc_vector_15'
            , 'sentenc_vector_16', 'sentenc_vector_17', 'sentenc_vector_18', 'sentenc_vector_19']

        dict_data_X = dict()

        for target_column in target_columns:
            dict_data_X[target_column] = data_X.copy()
            result_emb = self.get_token_feature(data, glove, target_column, batch_size)
            dict_data_X[target_column][sentence_vectors] = result_emb

        dict_data_X_columns = dict()

        for key, df in dict_data_X.items():
            dict_data_X_columns[key] = df.columns
            dict_data_X[key] = df.astype("float").values

        with open(os.path.join(model_path, "robust_scaler.pkl"),"rb") as f:
            scaler = pickle.load(f)

        for key, df in dict_data_X.items():

            X = pd.DataFrame(df, columns=dict_data_X_columns[key])
            X_subset = scaler.transform(
                X.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                          'engaging_user_follower_count', 'engaging_user_following_count',
                          'engaged_time > engaging_time',
                          'tweet_time > engaging_time', 'tweet_hour', 'language',
                          'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                          'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset = pd.DataFrame(X_subset)
            X_last_column = X.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                      'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                      'engagee_follows_engager', 'present_links',
                                      'bool_(engaged_time > engaging_time)',
                                      'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2',
                                      'sentenc_vector_3',
                                      'sentenc_vector_4',
                                      'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7',
                                      'sentenc_vector_8',
                                      'sentenc_vector_9',
                                      'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                      'sentenc_vector_13', 'sentenc_vector_14',
                                      'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                      'sentenc_vector_18', 'sentenc_vector_19']]
            X = pd.concat((X_subset, X_last_column), axis=1)
            dict_data_X[key] = X

        return df_indexes, dict_data_X

    def train(self, datas, target_column, epochs, batch_size, learning_rate, patience, token_model, gan_ratio):

        f_name = f"ANN_{target_column.lower()}_{token_model.lower()}_gan.pth" if gan_ratio > 0.0 else f"ANN_{target_column.lower()}_{token_model.lower()}_not_gan.pth"

        X_train, y_train, X_test, y_test = datas

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ann = CustomANN()
        ann = ann.to(device)

        es = None
        if patience > 0:
            es = EarlyStopping(self.model_path, patience, ("max","max"))
            es.names = ["rce", "ap"]

        X_train_torch = torch.FloatTensor(X_train if gan_ratio > 0.0 else X_train.values)
        y_train_torch = torch.LongTensor(y_train)

        train_dataset = AnnDataset(X_train_torch, y_train_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if (X_test is not None) & (y_test is not None):
            X_test_torch = torch.FloatTensor(X_test.values)
            y_test_torch = torch.LongTensor(y_test)

            test_dataset = AnnDataset(X_test_torch, y_test_torch)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # loss 함수 정의
        criterion = torch.nn.SmoothL1Loss(reduction='sum')
        # 최적화 함수 정의
        optimizer = optim.Adam(ann.parameters(), lr=learning_rate)

        for epoch in (tqdm(range(epochs), desc="train ann processing") if self.verbose else range(epochs)):
            ann.train()
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                # [inputs, labels]의 목록인 data 로부터 입력을 받은 후;
                inputs, labels = data[0].to(device), data[1].to(device)

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = ann(inputs)
                labels = labels.type_as(outputs)
                loss = criterion(outputs, labels.reshape(-1, 1))
                loss.backward()
                optimizer.step()

                # loss 출력
                running_loss += loss.item()
                # if self.verbose:
                #     if i % 10 == 9:
                #         print('[%d, %5d] loss: %.3f' %
                #               (epoch + 1, i + 1, running_loss / 10))
                #         running_loss = 0.0

            rce = None
            ap = None

            if (X_test is not None) & (y_test is not None):
                ann.eval()
                for data in test_loader:
                    lr_probs = []
                    test_y = []
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = ann(images)
                    lr_probs = outputs.cpu().squeeze().tolist()
                    test_y = labels.cpu().squeeze().tolist()

                    rce = compute_rce(lr_probs, test_y)
                    ap = average_precision_score(test_y, lr_probs)

                    break

                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{running_loss / len(train_dataloader):.4f}; rce:{rce:.4f}; ap:{ap:.4f}")

                if patience > 0:
                    # rce = compute_rce(lr_probs, test_y)
                    # ap = average_precision_score(test_y, lr_probs)
                    es((rce,ap), ann, f_name)

                if es.early_stop:
                    print("early_stopping")
                    break
                # if self.verbose:
                #     print("rce = ", compute_rce(lr_probs, test_y), "ap = ", average_precision_score(test_y, lr_probs))

            # if epoch % 5 == 4:
                #     ann.eval()
                #     for data in test_loader:
                #         lr_probs = []
                #         test_y = []
                #         images, labels = data[0].to(device), data[1].to(device)
                #         outputs = ann(images)
                #         lr_probs = outputs.cpu().squeeze().tolist()
                #         test_y = labels.cpu().squeeze().tolist()
                #         break
                #     if self.verbose:
                #         print("rce = ", compute_rce(lr_probs, test_y), "ap = ", average_precision_score(test_y, lr_probs))


            # torch.save(ann.state_dict(), os.path.join(self.model_path, f_name))

        # if self.verbose:
        #     print('Finished Training')

    def predict(self, X, target_column, batch_size, model_path, is_gan):


        if is_gan:
            f_name = f"ANN_{target_column.lower()}_dan_gan.pth"
        else:
            f_name = f"ANN_{target_column.lower()}_dan_not_gan.pth"

        lr_probs = []
        # y_true = []

        np_X = X.values

        X = torch.FloatTensor(np_X)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

        ann = CustomANN()
        if device == "cuda":
            ann.load_state_dict(torch.load(os.path.join(model_path, f_name)))
        else:
            ann.load_state_dict(torch.load(os.path.join(model_path, f_name), map_location=lambda storage, loc: storage))

        ann = ann.to(device)

        test_dataset = AnnDataset(X)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            ann.eval()
            for data in (tqdm(test_loader, desc="inference ann processing") if self.verbose else test_loader):
                # images, labels = data[0].to(device), data[1].to(device)
                images = data.to(device)
                outputs = ann(images)
                lr_probs.append(outputs.cpu().tolist())
                # y_true.append(labels.cpu().tolist())

        lr_probs = np.concatenate(lr_probs)
        # y_true = np.hstack(y_true)

        return lr_probs
