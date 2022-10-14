from random import betavariate
import pandas as pd
from tqdm.notebook import tqdm as tqdm
import requests
from bs4 import BeautifulSoup
import re
import time
import urllib.request
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt
from graphviz import *
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from scipy.special import comb
from itertools import combinations
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from scipy.special import comb
from itertools import permutations
import pickle
# import fasttext as ft

place_dict = {
    '札幌':'01',  '函館':'02',  '福島':'03',  '新潟':'04',  '東京':'05', 
    '中山':'06',  '中京':'07',  '京都':'08',  '阪神':'09',  '小倉':'10'
}

race_type_dict = {
    '芝': '芝', 'ダ': 'ダート', '障': '障害'
}



def split_data(df, test_size=0.2, rank_learning=True):
    """
    データを学習データと, 訓練データに分ける関数
    """
    df_ = df.copy()
    if not rank_learning:
        df_['rank'] = df_['rank'].map(lambda x:1 if x<4 else 0)
    sorted_id_list = df_.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df_.loc[train_id_list]#.drop(['date'], axis=1)
    test = df_.loc[test_id_list]#.drop(['date'], axis=1)
    return train, test

def rus_data(df, test_size=0.2):
    train, test = split_data(df,test_size=test_size)
    x_train = train.drop(['rank', 'date','単勝'], axis=1)
    y_train = train['rank']
    x_test = test.drop(['rank', 'date','単勝'], axis=1)
    y_test = test['rank']
    
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled, x_test, y_test

def load_csv(load_path):
    df = pd.read_csv(load_path, index_col=0)
    return df


def plot(g,label=''):
    plt.fill_between(g.index,y1 = g['return_rate'] - g['std'],y2=g['return_rate']+g['std'],alpha=0.3)
    plt.plot(g.index,g['return_rate'],label=label)
    plt.grid(True)
    
def update_data(old, new):
    """
    Parameters:
    ----------
    old : pandas.DataFrame
        古いデータ
    new : pandas.DataFrame
        新しいデータ
    """

    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])


def plot_importances(xgb_model, x_test):
    importances = pd.DataFrame(
    {'features' : x_test.columns, 'importances' : xgb_model.feature_importances_})
    print(importances.sort_values('importances', ascending=False)[:20])

def make_data(data_,test_rate=0.8,is_rus=True):
    data_ = data_.sort_values('date')
    x_ = data_.drop(['rank','date','単勝'],axis=1)
    y_ = data_['rank']

    test_rate = int(test_rate*len(x_))
    x_train, x_test = x_.iloc[:test_rate],x_.iloc[test_rate:]
    y_train, y_test = y_.iloc[:test_rate],y_.iloc[test_rate:]
    if is_rus:
        rus = RandomUnderSampler(random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
        return x_resampled, y_resampled, x_test, y_test
    else:
        return x_train,y_train,x_test,y_test

def make_race_id_list():

    race_id_list = []

    for place in range(1, 11, 1):
        for kai in range(1, 6, 1):
            for day in range(1, 13, 1):
                for r in range(1, 13, 1):
                    race_id = "2022" + str(place).zfill(2) + str(kai).zfill(2) +\
            str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
    
    return race_id_list


def save_pickle(save_path,object_):
    with open(save_path, mode="wb") as f:
        pickle.dump(object_, f)

def load_pickle(save_path):
    with open(save_path, mode="rb") as f:
        object_ = pickle.load(f)
    return object_

class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[['日付', '着順', '賞金', '着差', '通過',
                                            '開催', '距離']]
        self.preprocessing()
        
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list])
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        #horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in tqdm(horse_id_list):
#         for horse_id in horse_id_list:
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる        
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df
    
    
    #省略
        
    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(['日付'], axis=1, inplace=True)
        
        #賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)
        
        #1着の着差を0にする
        df['着差'] = df['着差'].map(lambda x: 0 if x<0 else x)
        
        #レース展開データ
        #n=1: 最初のコーナー位置, n=4: 最終コーナー位置
        def corner(x, n):
            if type(x) != str:
                return x
            elif n==4:
                return int(re.findall(r'\d+', x)[-1])
            elif n==1:
                return int(re.findall(r'\d+', x)[0])
        df['first_corner'] = df['通過'].map(lambda x: corner(x, 1))
        df['final_corner'] = df['通過'].map(lambda x: corner(x, 4))
        
        df['final_to_rank'] = df['final_corner'] - df['着順']
        df['first_to_rank'] = df['first_corner'] - df['着順']
        df['first_to_final'] = df['first_corner'] - df['final_corner']
        
        place_dict = {
            '札幌':'01',  '函館':'02',  '福島':'03',  '新潟':'04',  '東京':'05', 
            '中山':'06',  '中京':'07',  '京都':'08',  '阪神':'09',  '小倉':'10'
        }

        race_type_dict = {
            '芝': '芝', 'ダ': 'ダート', '障': '障害'
        }


        #開催場所
        df['開催'] = df['開催'].str.extract(r'(\D+)')[0].map(place_dict).fillna('11')
        #race_type
        df['race_type'] = df['距離'].str.extract(r'(\D+)')[0].map(race_type_dict)
        #距離
        df['course_len'] = df['距離'].str.extract(r'(\d+)').astype(int) // 100
        df.drop(['距離'], axis=1, inplace=True)
        
        #インデックス名を与える
        df.index.name = 'horse_id'
    
        self.horse_results = df
        self.target_list = ['着順', '賞金', '着差', 'first_corner',
                            'first_to_rank', 'first_to_final','final_to_rank']
        
        
    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.query('index in @horse_id_list')
        
        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')
          
        self.average_dict = {}
        self.average_dict['non_category'] = filtered_df.groupby(level=0)[self.target_list]\
            .mean().add_suffix('_{}R'.format(n_samples))
        for column in ['course_len', 'race_type', '開催']:
            self.average_dict[column] = filtered_df.groupby(['horse_id', column])\
                [self.target_list].mean().add_suffix('_{}_{}R'.format(column, n_samples)).fillna(0)

    
    def merge(self, results, date, n_samples='all'):
        df = results[results['date']==date]
        horse_id_list = df['horse_id']
        self.average(horse_id_list, date, n_samples)
        merged_df = df.merge(self.average_dict['non_category'], left_on='horse_id',
                             right_index=True, how='left')
        for column in ['course_len','race_type', '開催']:
            merged_df = merged_df.merge(self.average_dict[column], 
                                        left_on=['horse_id', column],
                                        right_index=True, how='left').fillna(0)
        return merged_df
    
    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat(
            [self.merge(results, date, n_samples) for date in tqdm(date_list)]
        )
        return merged_df

class Return:

    def __init__(self, return_tables):
        self.return_tables = return_tables
    
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list])
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        """
        払い戻し表データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = {}
        for race_id in tqdm(race_id_list):
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urllib.request.urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df
    
    

    @property
    def fukusho(self):
        fukusho = self.return_tables[self.return_tables[0]=='複勝'][[1,2]]
        wins = fukusho[1].str.split('br', expand=True)[[0,1,2]]
        
        wins.columns = ['win_0', 'win_1', 'win_2']
        returns = fukusho[2].str.split('br', expand=True)[[0,1,2]]
        returns.columns = ['return_0', 'return_1', 'return_2']
        
        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(',', '')
        return df.fillna(0).astype(int)
    
    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0]=='単勝'][[1,2]]
        tansho.columns = ['win', 'return']
        
        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors='coerce')
            
        return tansho
    
    @property
    def umaren(self):
        umaren = self.return_tables[self.return_tables[0]=='馬連'][[1,2]]
        wins = umaren[1].str.split('-', expand=True)[[0,1]].add_prefix('win_')
        return_ = umaren[2].rename('return')  
        df = pd.concat([wins, return_], axis=1)        
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    @property
    def umatan(self):
        umatan = self.return_tables[self.return_tables[0]=='馬単'][[1,2]]
        wins = umatan[1].str.split('→', expand=True)[[0,1]].add_prefix('win_')
        return_ = umatan[2].rename('return')  
        df = pd.concat([wins, return_], axis=1)        
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    @property
    def wide(self):
        wide = self.return_tables[self.return_tables[0]=='ワイド'][[1,2]]
        wins = wide[1].str.split('br', expand=True)[[0,1,2]]
        wins = wins.stack().str.split('-', expand=True).add_prefix('win_')
        return_ = wide[2].str.split('br', expand=True)[[0,1,2]]
        return_ = return_.stack().rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x.str.replace(',',''), errors='coerce'))
    
    @property
    def sanrentan(self):
        rentan = self.return_tables[self.return_tables[0]=='三連単'][[1,2]]
        wins = rentan[1].str.split('→', expand=True)[[0,1,2]].add_prefix('win_')
        return_ = rentan[2].rename('return')
        df = pd.concat([wins, return_], axis=1) 
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    @property
    def sanrenpuku(self):
        renpuku = self.return_tables[self.return_tables[0]=='三連複'][[1,2]]
        wins = renpuku[1].str.split('-', expand=True)[[0,1,2]].add_prefix('win_')
        return_ = renpuku[2].rename('return')
        df = pd.concat([wins, return_], axis=1) 
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
class ModelEvaluator:

    
    def __init__(self, model, return_tables):
        self.model = model
        self.rt = Return(return_tables)
        self.fukusho = self.rt.fukusho
        self.tansho = self.rt.tansho
        self.umaren = self.rt.umaren
        self.umatan = self.rt.umatan
        self.wide = self.rt.wide
        self.sanrenpuku = self.rt.sanrenpuku
        self.sanrentan = self.rt.sanrentan

    
    #3着以内に入る確率を予測
    def predict_proba(self, X, train=True, std=True, minmax=False):
        if train:
            proba = pd.Series(
                self.model.predict_proba(X.drop(['単勝'], axis=1))[:, 1], index=X.index
            )
        else:
            proba = pd.Series(
                self.model.predict_proba(X, axis=1)[:, 1], index=X.index
            )
        if std:
            #レース内で標準化して、相対評価する。「レース内偏差値」みたいなもの。
            standard_scaler = lambda x: (x - x.mean()) / x.std()
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            #データ全体を0~1にする
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba
    
    #0か1かを予測
    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        self.proba = y_pred
        return [0 if p<threshold else 1 for p in y_pred]
    
    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))
    
    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame({"features": X.columns, 
                                    "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]
    
    def pred_table(self, X, threshold=0.5, bet_only=True):
        pred_table = X.copy()[['馬番', '単勝']]
        pred_table['pred'] = self.predict(X, threshold)
        pred_table['score'] = self.proba
        if bet_only:
            return pred_table[pred_table['pred']==1][['馬番', '単勝', 'score','pred']]
        else:
            return pred_table[['馬番', '単勝', 'score', 'pred']]
        
    def bet(self, race_id, kind, umaban, amount):
        if kind == 'fukusho':
            rt_1R = self.fukusho.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1', 'win_2']]==umaban).values * \
                rt_1R[['return_0', 'return_1', 'return_2']].values * amount/100
            return_ = np.sum(return_)
        if kind == 'tansho':
            rt_1R = self.tansho.loc[race_id]
            return_ = (rt_1R['win']==umaban) * rt_1R['return'] * amount/100
        if kind == 'umaren':
            rt_1R = self.umaren.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if kind == 'umatan':
            rt_1R = self.umatan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1']]) == list(umaban))\
                * rt_1R['return']/100 * amount
        if kind == 'wide':
            rt_1R = self.wide.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1']].\
                           apply(lambda x: set(x)==set(umaban), axis=1)) \
                * rt_1R['return']/100 * amount
            return_ = return_.sum()
        if kind == 'sanrentan':
            rt_1R = self.sanrentan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1', 'win_2']]) == list(umaban)) * \
                rt_1R['return']/100 * amount
        if kind == 'sanrenpuku':
            rt_1R = self.sanrenpuku.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1', 'win_2']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if not (return_ >= 0):
                return_ = amount
        return return_
        
    def fukusho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(np.sum([
                self.bet(race_id, 'fukusho', umaban, 1) for umaban in preds['馬番']
            ]))
        return_rate = np.sum(return_list) / n_bets
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x>0 for x in return_list])
        return n_bets, return_rate, n_hits, std
    
    def tansho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        self.sample = pred_table
        n_bets = len(pred_table)
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum([self.bet(race_id, 'tansho', umaban, 1) for umaban in preds['馬番']])
            )
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def tansho_return_proper(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum(preds.apply(lambda x: self.bet(
                    race_id, 'tansho', x['馬番'], 1/x['単勝']), axis=1)))
        
        bet_money = (1 / pred_table['単勝']).sum()
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / bet_money
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / bet_money
        return n_bets, return_rate, n_hits, std
    
    def umaren_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umaren', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def umatan_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue   
            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umatan', umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def wide_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'wide', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std  
        
    def sanrentan_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            if len(preds)<3:
                continue
            else:
                for umaban in permutations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrentan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def sanrenpuku_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            if len(preds)<3:
                continue
            else:
                for umaban in combinations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrenpuku', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def umaren_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'umaren', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += n_aite
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umaren', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def umatan_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'umatan', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += n_aite
                
            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umatan', umaban, 1)
                    n_bets += 1
            return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def wide_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'wide', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'wide', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
    def sanrentan_nagashi(self, X, threshold = 1.5, n_aite=7):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) == 2:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[2:(n_aite+2)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'sanrentan',
                        np.append(preds_jiku['馬番'].values, x),
                        1
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 3:
                return_ = 0
                for umaban in permutations(preds_jiku['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrentan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
        
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        
        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std
    
class DataProcessor:
    
    def __init__(self):
        self.data = pd.DataFrame() #raw data
        self.data_p = pd.DataFrame() #after preprocessing
        self.data_h = pd.DataFrame() #after merging horse_results
        self.data_pe = pd.DataFrame() #after merging peds
        self.data_c = pd.DataFrame() #after processing categorical features
        
    #馬の過去成績データの追加
    def merge_horse_results(self, hr, n_samples_list=[5, 9, 'all']):
        self.data_h = self.data_p.copy()
        for n_samples in n_samples_list:
            self.data_h = hr.merge_all(self.data_h, n_samples=n_samples)
        self.data_h.drop(['開催'], axis=1, inplace=True)
            
    #血統データ追加
    def merge_peds(self, peds):
        self.data_pe = self.data_h.merge(peds, left_on='horse_id', right_index=True,how='left')
#         重複データを削除
        self.data_pe = self.data_pe[~self.data_pe.duplicated()]
        self.no_peds = self.data_pe[self.data_pe['peds_0'].isnull()]['horse_id'].unique()
#         print("type :",type(self.no_peds)) ndarray
#         Peds.scrape()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')
            print('no peds list',self.no_peds)
            
        #カテゴリ変数の処理
    def process_categorical(self, le_horse, le_jockey,results_m):
        df = self.data_pe.copy()
        
        #ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
        mask_horse = df['horse_id'].isin(le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df['horse_id'] = le_horse.transform(df['horse_id'])
        
        mask_jockey = df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        df['jockey_id'] = le_jockey.transform(df['jockey_id'])
#         pedsデータのラベルエンコーディング

#         for column in p.peds_e.columns:
# #             self.le_peds_dict[column] = LabelEncoder().fit_transform(df[column].fillna('Na'))
# #             mask_peds = df[column].isin(p.le_peds[column].classes_)
#             new_peds_id = df[column].dropna().unique()
# #             p.le_peds[column].classes_ = np.concatenate([p.le_peds[column].classes_, new_peds_id])
#             df[column] = p.le_peds[column].transform(df[column])
        
        
        #horse_id, jockey_idをpandasのcategory型に変換
        df['horse_id'] = df['horse_id'].astype('category')
        df['jockey_id'] = df['jockey_id'].astype('category')
        
        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        weathers = results_m['weather'].unique()
        race_types = results_m['race_type'].unique()
        ground_states = results_m['ground_state'].unique()
        sexes = results_m['性'].unique()
        df['weather'] = pd.Categorical(df['weather'], weathers)
        df['race_type'] = pd.Categorical(df['race_type'], race_types)
        df['ground_state'] = pd.Categorical(df['ground_state'], ground_states)
        df['性'] = pd.Categorical(df['性'], sexes)
        df = pd.get_dummies(df, columns=['weather', 'race_type', 'ground_state', '性'])
        
        self.data_c = df    
    
class ShutubaTable(DataProcessor):
    
    
    def __init__(self, shutuba_tables):
        super(ShutubaTable, self).__init__()
        self.data = shutuba_tables
    
    @classmethod
    def scrape(cls, race_id_list, date):
        data = pd.DataFrame()
        for race_id in tqdm(race_id_list):
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id
            df = pd.read_html(url)[0]
            df = df.T.reset_index(level=0, drop=True).T

            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")

            texts = soup.find('div', attrs={'class': 'RaceData01'}).text
            texts = re.findall(r'\w+', texts)
            for text in texts:
                if 'm' in text:
                    df['course_len'] = [int(re.findall(r'\d+', text)[0])] * len(df)
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    df["weather"] = [text] * len(df)
                if text in ["良", "稍重", "重","稍"]:
                    df["ground_state"] = [text] * len(df)
                if '不' in text:
                    df["ground_state"] = ['不良'] * len(df)
                if '芝' in text:
                    df['race_type'] = ['芝'] * len(df)
                if '障' in text:
                    df['race_type'] = ['障害'] * len(df)
                if 'ダ' in text:
                    df['race_type'] = ['ダート'] * len(df)
            df['date'] = [date] * len(df)

            # horse_id
            horse_id_list = []
            horse_td_list = soup.find_all("td", attrs={'class': 'HorseInfo'})
            for td in horse_td_list:
                horse_id = re.findall(r'\d+', td.find('a')['href'])[0]
                horse_id_list.append(horse_id)
            # jockey_id
            jockey_id_list = []
            jockey_td_list = soup.find_all("td", attrs={'class': 'Jockey'})
            for td in jockey_td_list:
                jockey_id = re.findall(r'\d+', td.find('a')['href'])[0]
                jockey_id_list.append(jockey_id)
            df['horse_id'] = list(map(lambda x: int(x),horse_id_list)) 
            df['jockey_id'] = jockey_id_list

            df.index = [race_id] * len(df)
#             win 環境だとなぜかintに直せない.floatならつかえる
#               int -> np.int64 とすることでエラー解消
            df.index = df.index.astype(np.int64)
            data = data.append(df)

            
        return data
                
    def preprocessing(self):
        df = self.data.copy()
        
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

#         体重変化をデータから消した
        # 馬体重を体重と体重変化に分ける
        df = df[df["馬体重(増減)"] != '--']
        df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1].replace('前計不',0).astype(int)


        
        df["date"] = pd.to_datetime(df["date"])
        
        df['枠'] = df['枠'].astype(int)
        df['馬番'] = df['馬番'].astype(int)
        df['斤量'] = df['斤量'].astype(int)
        df['開催'] = df.index.map(lambda x:str(x)[4:6])
        df['n_horse'] = df.index.map(lambda x: len(df.loc[x]))

        # 不要な列を削除
        df = df[['枠', '馬番', '斤量', 'course_len', 'weather','race_type',
        'ground_state', 'date', 'horse_id', 'jockey_id', '性', '年齢','開催','n_horse','体重','体重変化']]
        
        self.data_p = df.rename(columns={'枠': '枠番'})
        
class Results(DataProcessor):
    def __init__(self, results):
        super(Results, self).__init__()
        self.data = results
        self.le_peds = None
        
        
    @staticmethod
    def scrape(race_id_list):
        #race_idをkeyにしてDataFrame型を格納
        race_results = {}
        for race_id in tqdm(race_id_list):
            time.sleep(0.5)
            try:
                url = "https://db.netkeiba.com/race/" + race_id
                #メインとなるテーブルデータを取得
                df = pd.read_html(url)[0]
                html = requests.get(url)
                html.encoding = "EUC-JP"
                soup = BeautifulSoup(html.text, "html.parser")

                #天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[0])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                #馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list

                #インデックスをrace_idにする
                df.index = [race_id] * len(df)

                race_results[race_id] = df
            #存在しないrace_idを飛ばす
            except IndexError:
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break
            #Jupyterで停止ボタンを押した時の対処
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key] for key in race_results])

        return race_results_df
        
    #前処理    
    def preprocessing(self):
        df = self.data.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
#         rank学習の場合はそのまま
#         df['rank'] = df['着順'].map(lambda x:1 if x<4 else 0)
        df['rank'] = df['着順']

        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        # データをint, floatに変換
        df["単勝"] = df["単勝"].astype(float)
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除
        df.drop(["タイム", "着差", "調教師", "性齢", "馬体重", '馬名', '騎手', '人気', '着順'],
                axis=1, inplace=True)

        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")
        
        #開催場所
        df['開催'] = df.index.map(lambda x:str(x)[4:6])
        df['n_horse'] = df.index.map(lambda x: len(df.loc[x]))
        
        self.data_p = df
    
    #カテゴリ変数の処理
    def process_categorical(self):
        self.le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        self.le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
#         self.le_peds = p.le_peds_dict
        super().process_categorical(self.le_horse, self.le_jockey,self.data_pe)
        
class Peds:

    def __init__(self, peds):
        self.peds = peds
        self.peds_cat = pd.DataFrame() #after label encoding and transforming into category
        self.peds_re = pd.DataFrame()
        self.peds_vec = pd.DataFrame()
    
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list])
        return cls(df)
    
    @staticmethod
    def scrape(horse_id_list):
        peds_dict = {}
        for horse_id in tqdm(horse_id_list):
#         for horse_id in horse_id_list:
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
            
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)

                peds_dict[horse_id] = ped.reset_index(drop=True)
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict],
                            axis=1).T.add_prefix('peds_')
        peds_df.index =peds_df.index.astype(int)

        return peds_df
    
    
#     血統データが正規化されたいないデータに対して, 正規化する関数
    def regularize_peds(self):
        peds = self.peds.copy()
        error_idx_list = []
        for idx in tqdm(peds.index):
            for col in peds.columns:
            #     漢字 : 一-龥
                code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％一-龥\d]')
                try:
                    cleaned_text = code_regex.sub('', peds[col].loc[idx])
                    one_word = "".join(cleaned_text.split())
                    p_alphabet = re.compile('[a-zA-Z]+')
                    p_katakana = re.compile(r'[ァ-ヶー]+')

                    peds[col].loc[idx] = one_word
                    if (not p_alphabet.fullmatch(one_word)) and not (p_katakana.fullmatch(one_word)):
                        peds[col].loc[idx] = re.sub('[a-zA-Z]+', '', one_word)
                except:
                    error_idx_list.append(idx)
        self.error_idx_list_r = error_idx_list
        self.peds_re = peds

    
    def categorize(self):
        df = self.peds.copy()
        self.le_peds_dict = {}
        
        
        for column in df.columns:
            
            self.le_peds_dict[column] = LabelEncoder()
            df[column] = self.le_peds_dict[column].fit_transform(df[column].fillna('Na'))
#             df[column] = self.le_peds_dict[column]
        self.peds_cat = df.astype('category')
        self.le_peds = self.le_peds_dict
        
        
#         血統データをベクトル化する関数
# peds_re は 正規化済み血統データを仮定
# model_ft : fasttextモデル
    def vectorize(self,peds_re,model_ft):
        df = peds_re.copy()
        error_idx_list = []
        for idx in tqdm(df.index):
            text = ','.join(df.loc[idx].tolist())
            df.loc[idx] = model_ft[text]
#             except:
#                 error_idx_list.append(idx)
        self.error_idx_list_v = error_idx_list
        self.peds_vec = df.astype('float')


class Simulater():
    
    def __init__(self, model):
        self.model = model
        self.return_tables = None
        self.pred_df = None
        self.is_long = True
    

    #     当日のデータでシミュレートするとあかん
    def return_table(self, race_id_list):
        return_tables = Return.scrape(race_id_list)
        return_tables.rename(columns={'0':0,'1':1,'2':2,'3':3},inplace=True)
        self.return_tables = return_tables
    
    
    def return_table_today(self,race_id_list):
        return_tables = {}
        for race_id in tqdm(race_id_list):
            try:
                url = 'https://race.netkeiba.com/race/result.html?race_id='+race_id+'&amp;rf=race_submenu'
                dfs = pd.read_html(url)
                df = pd.concat([dfs[1], dfs[2]])
                df.index = [race_id] * len(df)
                return_tables[race_id] = df
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break
            #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        # return_tables_df.index = return_tables_df.index.astype(int)
        self.return_tables = return_tables_df
    
    
    def return_pred_table(self,data_c,is_long=False):
        # is_long って何？
        #予測
        if not is_long:
            scores = pd.Series(self.model.predict(data_c.drop(['date'],axis=1)),index=data_c.index)
        else:
            scores = pd.Series(self.model.predict(data_c.drop(['date','rank','単勝'],axis=1)),index=data_c.index)
        pred = data_c[['馬番']].copy()
        pred['scores'] = scores
        pred = pred.sort_values('scores',ascending=False)
        return pred

#     odds以上の馬券しか買わない
    def get_result_df(self, data_c, return_tables, is_long=True, odds=2.0, bet = 100):
        race_id_list = list(set(data_c.index))
        race_dict = {}

        for race_id in race_id_list:
            pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list = self.return_race_result(data_c,race_id,return_tables)
            row_list = [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list]
            race_dict[int(race_id)] = row_list
        all_result = pd.DataFrame(race_dict).T
        all_result.rename(columns={
            0:'pred_list',
            1:'actual_rank_list',
            2:'tansho_odds',
            3:'fukusho_odds',
            4:'umaren_odds',
            5:'wide_odds',
            6:'umatan_odds',
            7:'sanrenpuku_odds',
            8:'sanrentan_odds',
            9:'wide_comb',
            10:'odds_list'
            },inplace=True)
        return all_result
        

    def return_race_result(self, data_c ,race_id, return_tables):
        race_id = int(race_id)
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=self.is_long)
        pred_df = pred_df.loc[race_id]
        pred_df = pred_df.sort_values('scores',ascending=False)
        dc = data_c.loc[race_id]
        return_table  = return_tables.loc[race_id]
        
        
        pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

        score_1 = pred_df['scores'].iloc[0]
        score_2 = pred_df['scores'].iloc[1]
        is_same_score = False

        try:
        
            tansho_row = return_table[return_table[0]=='単勝']
            fukusho_row = return_table[return_table[0]=='複勝']
            umaren_row =  return_table[return_table[0]=='馬連']
            umatan_row =  return_table[return_table[0]=='馬単']
            wide_row =  return_table[return_table[0]=='ワイド']
            sanrentan_row =  return_table[return_table[0]=='三連単']
            sanrenpuku_row =  return_table[return_table[0]=='三連複']
            
            # odds 順番は予測した順
            odds_list = []
            for ub in pred_df['馬番'].tolist():
                odds_list.append(dc[dc['馬番']==ub]['単勝'].values[0])
        
            if score_1 == score_2:
                is_same_score =True
                
            # １着が同着    
            if int(tansho_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds_list = tansho_row[2].str.split('br').values[0][0:3]
                tansho_odds_list = [i for i in tansho_odds_list if i!='']
                tansho_odds = list(map(lambda x: int(x.replace(',',''))/100 ,tansho_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
        
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
            # S2
            elif int(umaren_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                umaren_odds_list = umaren_row[2].str.split('br').values[0][0:3]
                umaren_odds_list = [i for i in umaren_odds_list if i!='']
                umaren_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umaren_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            
            # S3
            elif int(sanrenpuku_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:4]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                umaren_odds = int(umaren_row[2])/100
                umatan_odds = int(umatan_row[2])/100
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:5]))
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:5]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                sanrenpuku_odds_list = sanrenpuku_row[2].str.split('br').values[0][0:3]
                sanrenpuku_odds_list = [i for i in sanrenpuku_odds_list if i!='']
                sanrenpuku_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrenpuku_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            else:
                actual_rank_list = list(map(int,sanrentan_row[1].str.split('→').values[0]))
                
                tansho_odds = int(tansho_row[2])/100
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                umatan_odds = int(umatan_row[2])/100
                
                sanrentan_odds = int(sanrentan_row[2])/100
                
        except Exception as e:
            print(e)
            print(race_id)
            return
        
        
        return  pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list


class TodaySimulater(Simulater):

    def __init__(self,model):
        super(TodaySimulater,self).__init__(model)
        self.is_long = False

    def return_race_result(self,data_c,race_id,return_tables):
        race_id = int(race_id)
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=self.is_long)
        return_table  = return_tables.loc[str(race_id)]
        pred_df = pred_df.loc[race_id]
        pred_df = pred_df.sort_values('scores',ascending=False)
        pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

        score_1 = pred_df['scores'].iloc[0]
        score_2 = pred_df['scores'].iloc[1]
        is_same_score = False

        
        tansho_row = return_table[return_table[0]=='単勝']
        fukusho_row = return_table[return_table[0]=='複勝']
        umaren_row =  return_table[return_table[0]=='馬連']
        umatan_row =  return_table[return_table[0]=='馬単']
        wide_row =  return_table[return_table[0]=='ワイド']
        sanrentan_row =  return_table[return_table[0]=='3連単']
        sanrenpuku_row =  return_table[return_table[0]=='3連複']
        
        actual_rank_list = list(map(int,sanrentan_row[1].str.split(' ').values[0]))
        
        odds_tmp =umaren_row[2].str.replace(',','')
        umaren_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = tansho_row[2].str.replace(',','')
        tansho_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = umatan_row[2].str.replace(',','')
        umatan_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = sanrenpuku_row[2].str.replace(',','')
        sanrenpuku_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = sanrentan_row[2].str.replace(',','')
        sanrentan_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        fukusho_odds_list = fukusho_row[2].str.split('円').values[0][0:3]
        fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
        fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
        wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('円').values[0][0:3]))
        
        tmp_list = list(map(int,wide_row[1].str.split(' ').tolist()[0]))
        wide_comb = []
        for i in range(0,len(tmp_list),2):
            wide_comb.append(tmp_list[i:i+2])
        
        if score_1 == score_2:
            is_same_score =True
        
        
        return  pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb

    def get_result_df(self, data_c, return_tables, race_id_list, kaime='tansho', odds=2, bet=100):
        race_dict = {}

        if kaime=='tansho':
            for race_id in race_id_list:
                profit,is_atari,is_buy,actual_rank,not_buy_reason,pred_odds = self.calc_tansho(data_c,race_id,odds,bet,return_tables)
                row_list = [profit,is_atari,is_buy,actual_rank,not_buy_reason,pred_odds]
                race_dict[int(race_id)] = row_list
            tansho_result = pd.DataFrame(race_dict).T
            tansho_result.rename(columns={0:'profit',1:'is_atari',2:'is_buy',3:'actual_rank',4:'not_buy_reason',5:'real_odds'},inplace=True)
            return tansho_result
        
        if kaime=='all':
            for race_id in race_id_list:
                pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb = self.return_race_result(data_c,race_id,return_tables)
                row_list = [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb]
                race_dict[int(race_id)] = row_list
            all_result = pd.DataFrame(race_dict).T
            all_result.rename(columns={
                0:'pred_list',
                1:'actual_rank_list',
                2:'tansho_odds',
                3:'fukusho_odds',
                4:'umaren_odds',
                5:'wide_odds',
                6:'umatan_odds',
                7:'sanrenpuku_odds',
                8:'sanrentan_odds',
                9:'wide_comb'
                },inplace=True)
            return all_result


class LearnLGBM():
    
    
    def __init__(self,peds,results,horse_results):
        self.model = None
        self.model_ft = None
        self.date = '2022/12/31'
        self.pe = None
        self.r = None
        self.horse_results = None
        self.peds = peds
        self.results = results
        self.horse_results = horse_results
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.path_ft = '/Users/Owner/Desktop/Horse/horse/peds_ft.txt'

        self.lgbm_params = {
                'metric': 'ndcg',
                'objective': 'lambdarank',
                'ndcg_eval_at': [1,2,3],
                'boosting_type': 'gbdt',
                'random_state': 777,
                'lambdarank_truncation_level': 10,
                'learning_rate': 0.02273417953255777,
                'n_estimators': 97,
                'num_leaves': 42,
                'force_col_wise':True
            }



    def learn_model_ft(self,minn=2,maxn=14):
        path_ft = self.path_ft
        model_ft = ft.train_unsupervised(path_ft,dim=62,minn=minn,maxn=maxn)
        self.model_ft = model_ft


    def get_model_ft(self):
        return self.model_ft
    

    def process_pe(self,peds):
        pe = Peds(peds)
        pe.regularize_peds()
        # 血統データ　カテゴリ変数処理
        pe.categorize()
        # pe.vectorize(pe.peds_re,self.model_ft)
        self.pe = pe
        print("pe finish")
        print("pe regularizrd")


    def process_hr(self,results,horse_results):
        r = Results(results)
        r.preprocessing()
        #馬の過去成績データ追加
        hr = HorseResults(horse_results)
        self.hr = hr
        r.merge_horse_results(hr)
        r.merge_peds(self.pe.peds_cat)
        # r.merge_peds(pe.peds_cat)
        #カテゴリ変数の処理
        # pedsは既にカテゴリ化したdataをconcatしているので, ここでカテゴリ化せずとも良い
        r.process_categorical()
        self.r = r


    def process_data(self):
        peds = self.peds.copy()
        results = self.results.copy()
        horse_results = self.horse_results.copy()
        # self.learn_model_ft()
        self.process_pe(peds)
        self.process_hr(results,horse_results)
        
        
    def get_train_data(self,test_size=0.2):
        self.process_data()
        train, test = split_data(self.r.data_c.fillna(0),test_size=test_size,rank_learning=False)
        x_train = train.drop(['rank', 'date','単勝'], axis=1)
        y_train = train['rank']
        x_test = test.drop(['rank', 'date','単勝'], axis=1)
        y_test = test['rank']
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return train, test

    
    def get_train_data2(self):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        return train, test


    def learn_lgb(self,lgbm_params=None,test_size=0.2):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        train, test = self.get_train_data(test_size=test_size)
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank

# train data を与えて学習させるのが learn_lgb2
    def learn_lgb2(self,train,lgbm_params=None):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank
    

class Predictor(LearnLGBM):


    def __init__(self,peds,results,horse_results,race_id_list):
        super(Predictor, self).__init__(peds,results,horse_results)
        self.race_id_list = race_id_list
        self.nopeds_id_list = []
        

    def process_hr(self,results,horse_results):
        r = Results(results)
        r.preprocessing()
        horse_id_list = self.data['horse_id'].astype(str).unique()
        horse_results_tmp = HorseResults.scrape(horse_id_list)
        new_horse_results = update_data(horse_results,horse_results_tmp)
        self.hores_results = new_horse_results.copy()
        hr = HorseResults(new_horse_results)
        self.hr = hr
        print("hr finish")
        r.merge_horse_results(hr)
        r.merge_peds(self.pe.peds_cat)
        r.process_categorical()  
        self.r = r
        

    def process_data(self):
        race_id_list = self.race_id_list.copy()
        data =  ShutubaTable.scrape(race_id_list, self.date)
        self.data = data
        peds = self.peds.copy()
        results = self.results.copy()
        horse_results = self.horse_results.copy()
        nopeds_id_list = []


        for ind in data['horse_id'].astype(int).unique():
            if ind not in peds.index:
                nopeds_id_list.append(str(ind))

        if len(nopeds_id_list)!=0:
            peds_tmp = Peds.scrape(nopeds_id_list)
            pe_tmp = Peds(peds_tmp)
            pe_tmp.regularize_peds() 
            new_peds = update_data(peds, pe_tmp.peds_re)
        else:
            new_peds = peds.copy()
        
        self.nopeds_id_list = nopeds_id_list
        self.peds = new_peds.copy()
        path_ft =  self.path_ft 
        
        new_peds.to_csv(path_ft,header=False,index=False,sep=',')
        self.learn_model_ft()
        self.process_pe(new_peds)
        self.process_hr(results,horse_results)

    
    def predict(self, race_id):
        data =  ShutubaTable.scrape([str(race_id)], self.date)
        st = ShutubaTable(data)
        st.preprocessing()
        st.merge_horse_results(self.hr)
        st.merge_peds(self.pe.peds_cat)
        st.process_categorical(self.r.le_horse, self.r.le_jockey, self.r.data_pe)
        self.st = st
        sl = Simulater(self.model)
        pred_table = sl.return_pred_table(st.data_c)
        self.sl = sl
        print(pred_table)
            

    def show_results_today(self):
        data =  ShutubaTable.scrape(self.race_id_list, self.date)
        self.data = data.copy()
        st = ShutubaTable(self.data)
        st.preprocessing()
        st.merge_horse_results(self.hr)
        st.merge_peds(self.pe.peds_cat)
        st.process_categorical(self.r.le_horse, self.r.le_jockey, self.r.data_pe)
        sl = RankSimulater(self.model)
        sl.return_table_today(self.race_id_list)
        sl.show_results_today(st ,self.race_id_list)


import lightgbm as lgb



class Simulater():
    
    def __init__(self, model):
        
        self.model = model
        self.return_tables = None
        self.pred_df = None
        self.is_long = True
    

    #     当日のデータでシミュレートするとあかん
    def return_table(self, race_id_list):
        return_tables = Return.scrape(race_id_list)
        return_tables.rename(columns={'0':0,'1':1,'2':2,'3':3},inplace=True)
        self.return_tables = return_tables
        return
    
    
    def return_table_today(self,race_id_list):
        return_tables = {}
        for race_id in tqdm(race_id_list):
            try:
                url = 'https://race.netkeiba.com/race/result.html?race_id='+race_id+'&amp;rf=race_submenu'
                dfs = pd.read_html(url)
                df = pd.concat([dfs[1], dfs[2]])
                df.index = [race_id] * len(df)
                return_tables[race_id] = df
                time.sleep(0.5)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break
            #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        # return_tables_df.index = return_tables_df.index.astype(int)
        self.return_tables = return_tables_df
    
    
    def return_pred_table(self,data_c,is_long=False):
        # is_long って何？
        #予測
        if not is_long:
            scores = pd.Series(self.model.predict(data_c.drop(['date'],axis=1)),index=data_c.index)
        else:
            scores = pd.Series(self.model.predict(data_c.drop(['date','rank','単勝'],axis=1)),index=data_c.index)
        pred = data_c[['馬番']].copy()
        pred['scores'] = scores
        pred = pred.sort_values('scores',ascending=False)
        return pred

#     odds以上の馬券しか買わない
    def get_result_df(self, data_c, return_tables, is_long=True, odds=2.0, bet = 100):
        race_id_list = list(set(data_c.index))
        race_dict = {}

        for race_id in race_id_list:
            pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list = self.return_race_result(data_c,race_id,return_tables)
            row_list = [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list]
            race_dict[int(race_id)] = row_list
        all_result = pd.DataFrame(race_dict).T
        all_result.rename(columns={
            0:'pred_list',
            1:'actual_rank_list',
            2:'tansho_odds',
            3:'fukusho_odds',
            4:'umaren_odds',
            5:'wide_odds',
            6:'umatan_odds',
            7:'sanrenpuku_odds',
            8:'sanrentan_odds',
            9:'wide_comb',
            10:'odds_list'
            },inplace=True)
        return all_result
        

    def return_race_result(self, data_c ,race_id, return_tables):
        race_id = race_id
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=self.is_long)
        pred_df = pred_df.loc[race_id]
        pred_df = pred_df.sort_values('scores',ascending=False)
        dc = data_c.loc[race_id]
        return_table  = return_tables.loc[race_id]
        
        
        pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

        score_1 = pred_df['scores'].iloc[0]
        score_2 = pred_df['scores'].iloc[1]
        is_same_score = False

        try:
        
            tansho_row = return_table[return_table[0]=='単勝']
            fukusho_row = return_table[return_table[0]=='複勝']
            umaren_row =  return_table[return_table[0]=='馬連']
            umatan_row =  return_table[return_table[0]=='馬単']
            wide_row =  return_table[return_table[0]=='ワイド']
            sanrentan_row =  return_table[return_table[0]=='三連単']
            sanrenpuku_row =  return_table[return_table[0]=='三連複']
            
            # odds 順番は予測した順
            odds_list = []
            for ub in pred_df['馬番'].tolist():
                odds_list.append(dc[dc['馬番']==ub]['単勝'].values[0])
        
            if score_1 == score_2:
                is_same_score =True
                
            # １着が同着    
            if int(tansho_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds_list = tansho_row[2].str.split('br').values[0][0:3]
                tansho_odds_list = [i for i in tansho_odds_list if i!='']
                tansho_odds = list(map(lambda x: int(x.replace(',',''))/100 ,tansho_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
        
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
            # S2
            elif int(umaren_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                umaren_odds_list = umaren_row[2].str.split('br').values[0][0:3]
                umaren_odds_list = [i for i in umaren_odds_list if i!='']
                umaren_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umaren_odds_list))
                
                umatan_odds_list = umatan_row[2].str.split('br').values[0][0:3]
                umatan_odds_list = [i for i in umatan_odds_list if i!='']
                umatan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,umatan_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                    
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            
            # S3
            elif int(sanrenpuku_row[1].str.count('br'))==1:
                actual_tmp0 = sanrentan_row[1].str.split('br').values[0][0]
                actual_tmp1 = sanrentan_row[1].str.split('br').values[0][1]
                actual_rank_list0 = list(map(int,actual_tmp0.split('→')))
                actual_rank_list1 = list(map(int,actual_tmp1.split('→')))
                actual_rank_list = [actual_rank_list0,actual_rank_list1]
                
                tansho_odds = int(tansho_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:4]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                umaren_odds = int(umaren_row[2])/100
                umatan_odds = int(umatan_row[2])/100
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:5]))
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:5]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                sanrenpuku_odds_list = sanrenpuku_row[2].str.split('br').values[0][0:3]
                sanrenpuku_odds_list = [i for i in sanrenpuku_odds_list if i!='']
                sanrenpuku_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrenpuku_odds_list))
                
                sanrentan_odds_list = sanrentan_row[2].str.split('br').values[0][0:3]
                sanrentan_odds_list = [i for i in sanrentan_odds_list if i!='']
                sanrentan_odds = list(map(lambda x: int(x.replace(',',''))/100 ,sanrentan_odds_list))
            else:
                actual_rank_list = list(map(int,sanrentan_row[1].str.split('→').values[0]))
                
                tansho_odds = int(tansho_row[2])/100
                umaren_odds = int(umaren_row[2])/100
                sanrenpuku_odds = int(sanrenpuku_row[2])/100
                fukusho_odds_list = fukusho_row[2].str.split('br').values[0][0:3]
                fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
                fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
                
                wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('br').values[0][0:3]))
                
                tmp_list = list(map(lambda x:x.replace(' - ',' '),wide_row[1].str.split('br').values[0][0:3]))
                wide_comb = []
                for tl in tmp_list:
                    pair_list = list(map(lambda x: int(x),tl.split(' ')))
                    wide_comb.append(pair_list)
                
                umatan_odds = int(umatan_row[2])/100
                
                sanrentan_odds = int(sanrentan_row[2])/100
                
        except Exception as e:
            print(e)
            print(race_id)
            return
        
        
        return  pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb,odds_list


class TodaySimulater(Simulater):

    def __init__(self,model):
        super(TodaySimulater,self).__init__(model)
        self.is_long = False

    def return_race_result(self,data_c,race_id,return_tables):
        race_id = int(race_id)
        pred_df = self.return_pred_table(data_c.loc[race_id],is_long=self.is_long)
        return_table  = return_tables.loc[str(race_id)]
        pred_df = pred_df.loc[race_id]
        pred_df = pred_df.sort_values('scores',ascending=False)
        pred_list = [int(pred_df['馬番'].iloc[i]) for i in range(len(pred_df))]

        score_1 = pred_df['scores'].iloc[0]
        score_2 = pred_df['scores'].iloc[1]
        is_same_score = False

        
        tansho_row = return_table[return_table[0]=='単勝']
        fukusho_row = return_table[return_table[0]=='複勝']
        umaren_row =  return_table[return_table[0]=='馬連']
        umatan_row =  return_table[return_table[0]=='馬単']
        wide_row =  return_table[return_table[0]=='ワイド']
        sanrentan_row =  return_table[return_table[0]=='3連単']
        sanrenpuku_row =  return_table[return_table[0]=='3連複']
        
        actual_rank_list = list(map(int,sanrentan_row[1].str.split(' ').values[0]))
        
        odds_tmp =umaren_row[2].str.replace(',','')
        umaren_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = tansho_row[2].str.replace(',','')
        tansho_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = umatan_row[2].str.replace(',','')
        umatan_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = sanrenpuku_row[2].str.replace(',','')
        sanrenpuku_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        odds_tmp = sanrentan_row[2].str.replace(',','')
        sanrentan_odds = int(odds_tmp.str.replace('円','').values[0])/100
        
        fukusho_odds_list = fukusho_row[2].str.split('円').values[0][0:3]
        fukusho_odds_list = [i for i in fukusho_odds_list if i!='']
        fukusho_odds = list(map(lambda x: int(x.replace(',',''))/100 , fukusho_odds_list))
        wide_odds = list(map(lambda x: int(x.replace(',',''))/100 , wide_row[2].str.split('円').values[0][0:3]))
        
        tmp_list = list(map(int,wide_row[1].str.split(' ').tolist()[0]))
        wide_comb = []
        for i in range(0,len(tmp_list),2):
            wide_comb.append(tmp_list[i:i+2])
        
        if score_1 == score_2:
            is_same_score =True
        
        
        return  pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb

    def get_result_df(self, data_c, return_tables, race_id_list, kaime='tansho', odds=2, bet=100):
        race_dict = {}

        if kaime=='tansho':
            for race_id in race_id_list:
                profit,is_atari,is_buy,actual_rank,not_buy_reason,pred_odds = self.calc_tansho(data_c,race_id,odds,bet,return_tables)
                row_list = [profit,is_atari,is_buy,actual_rank,not_buy_reason,pred_odds]
                race_dict[int(race_id)] = row_list
            tansho_result = pd.DataFrame(race_dict).T
            tansho_result.rename(columns={0:'profit',1:'is_atari',2:'is_buy',3:'actual_rank',4:'not_buy_reason',5:'real_odds'},inplace=True)
            return tansho_result
        
        if kaime=='all':
            for race_id in race_id_list:
                pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb = self.return_race_result(data_c,race_id,return_tables)
                row_list = [pred_list,actual_rank_list,tansho_odds,fukusho_odds,umaren_odds,wide_odds,umatan_odds,sanrenpuku_odds,sanrentan_odds,wide_comb]
                race_dict[int(race_id)] = row_list
            all_result = pd.DataFrame(race_dict).T
            all_result.rename(columns={
                0:'pred_list',
                1:'actual_rank_list',
                2:'tansho_odds',
                3:'fukusho_odds',
                4:'umaren_odds',
                5:'wide_odds',
                6:'umatan_odds',
                7:'sanrenpuku_odds',
                8:'sanrentan_odds',
                9:'wide_comb'
                },inplace=True)
            return all_result


class LearnLGBM():
    

    def __init__(self,peds,results,horse_results):
        self.model = None
        self.model_ft = None
        self.date = '2022/12/31'
        self.pe = None
        self.r = None
        self.horse_results = None
        self.peds = peds
        self.results = results
        self.horse_results = horse_results
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.path_ft = '/Users/Owner/Desktop/Horse/horse/peds_ft.txt'

        self.lgbm_params = {
                'metric': 'ndcg',
                'objective': 'lambdarank',
                'ndcg_eval_at': [1,2,3],
                'boosting_type': 'gbdt',
                'random_state': 777,
                'lambdarank_truncation_level': 10,
                'learning_rate': 0.02273417953255777,
                'n_estimators': 97,
                'num_leaves': 42,
                'force_col_wise':True
            }



    def learn_model_ft(self,minn=2,maxn=14):
        path_ft = self.path_ft
        model_ft = ft.train_unsupervised(path_ft,dim=62,minn=minn,maxn=maxn)
        self.model_ft = model_ft


    def get_model_ft(self):
        return self.model_ft
    

    def process_pe(self,peds):
        pe = Peds(peds)
        pe.regularize_peds()
        # 血統データ　カテゴリ変数処理
        pe.categorize()
        # pe.vectorize(pe.peds_re,self.model_ft)
        self.pe = pe
        print("pe finish")
        print("pe regularizrd")


    def process_hr(self,results,horse_results):
        r = Results(results)
        r.preprocessing()
        #馬の過去成績データ追加
        hr = HorseResults(horse_results)
        self.hr = hr
        r.merge_horse_results(hr)
        r.merge_peds(self.pe.peds_cat)
        # r.merge_peds(pe.peds_cat)
        #カテゴリ変数の処理
        # pedsは既にカテゴリ化したdataをconcatしているので, ここでカテゴリ化せずとも良い
        r.process_categorical()
        self.r = r


    def process_data(self):
        peds = self.peds.copy()
        results = self.results.copy()
        horse_results = self.horse_results.copy()
        # self.learn_model_ft()
        self.process_pe(peds)
        self.process_hr(results,horse_results)
        
        
    def get_train_data(self,test_size=0.2):
        self.process_data()
        train, test = split_data(self.r.data_c.fillna(0),test_size=test_size,rank_learning=False)
        x_train = train.drop(['rank', 'date','単勝'], axis=1)
        y_train = train['rank']
        x_test = test.drop(['rank', 'date','単勝'], axis=1)
        y_test = test['rank']
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return train, test

    
    def get_train_data2(self):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        return train, test


    def get_train_data3(featured_data,test_rate=1.0,is_rus=False):
        train,test = split_data(feature_enginnering.featured_data)
        x_train,y_train,_,_ = make_data(train,test_rate=test_rate,is_rus=False)
        x_test,y_test,_,_ = make_data(test,test_rate=test_rate,is_rus=False)

        train_query = x_train.groupby(x_train.index).size()
        test_query = x_test.groupby(x_test.index).size()
        train = lgb.Dataset(x_train, y_train, group=train_query)
        test = lgb.Dataset(x_test, y_test, reference=train, group=test_query)
        return train, test


    def learn_lgb(self,lgbm_params=None,test_size=0.2):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        train, test = self.get_train_data(test_size=test_size)
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank

# train data を与えて学習させるのが learn_lgb2
    def learn_lgb2(self,train,lgbm_params=None):
        if lgbm_params==None:
            lgbm_params = self.lgbm_params
        
        lgb_rank = lgb.train(
                lgbm_params,
                train,
                # valid_sets=test,
                num_boost_round=100,
                valid_names=['train'],
                # early_stopping_rounds=20,
            )

        self.model = lgb_rank
    
def split_data(df, test_size=0.2, rank_learning=True):
    """
    データを学習データと, 訓練データに分ける関数
    """
    df_ = df.copy()
    if not rank_learning:
        df_['rank'] = df_['rank'].map(lambda x:1 if x<4 else 0)
    sorted_id_list = df_.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df_.loc[train_id_list]#.drop(['date'], axis=1)
    test = df_.loc[test_id_list]#.drop(['date'], axis=1)
    return train, test

def make_data(data_,test_rate=0.8,is_rus=True):
    x_ = data_.drop(['rank','date','単勝'],axis=1)
    y_ = data_['rank']

    test_rate = int(test_rate*len(x_))
    x_train, x_test = x_.iloc[:test_rate],x_.iloc[test_rate:]
    y_train, y_test = y_.iloc[:test_rate],y_.iloc[test_rate:]
    if is_rus:
        rus = RandomUnderSampler(random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
        return x_resampled, y_resampled, x_test, y_test
    else:
        return x_train,y_train,x_test,y_test

def calc_tansho(all_results,odds_alpha=2,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}

# 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:

#**************
        is_particular_race = True
        # if str(race_id)[-2:]=='01' or str(race_id)[-2:]=='02' or str(race_id)[-2:]=='03':is_particular_race = True
#***************
        if is_particular_race:
            is_buy = False
            ar = all_results.loc[race_id]
            pred_list = ar['pred_list']
            actual_list = ar['actual_rank_list']
            tansho_odds = ar['tansho_odds']
            pred_odds = ar['odds_list'][0]

            if not is_today:
                
                if pred_odds>=odds_alpha:
                    profit -= bet
                else:
                    is_buy=True
            else:
                profit -= bet*length
                
            if pred_list[0]==actual_list[0] and is_buy:
                tekichu+=1
                profit += bet*tansho_odds
                race_hit_dist[str(race_id)[-2:]] += 1
    # -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)
    
def calc_fukusho(all_results,odds_alpha=2,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    # 何点買うか決める変数
    num_kai = 3
    fukusho_ken = 3

# 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:

#**************
        is_particular_race = False
        if str(race_id)[-2:]=='01':is_particular_race = True
            # or str(race_id)[-2:]=='02' or str(race_id)[-2:]=='03':is_particular_race = True
#***************
        if is_particular_race:
            is_buy = False
            ar = all_results.loc[race_id]
            pred_list = ar['pred_list']
            actual_list = ar['actual_rank_list']
            tansho_odds = ar['tansho_odds']
            fukusho_odds = ar['fukusho_odds']

            if not is_today:
                pred_odds = ar['odds_list'][0]
            
            if not is_today:
                if pred_odds>=odds_alpha:
                    profit -= bet
                    is_buy = True
            else:
                profit -= bet
                is_buy= True
            

            if len(fukusho_odds)==2:
                fukusho_ken = 2

            if pred_list[0] in actual_list[0:fukusho_ken] and is_buy:

                tekichu_index = actual_list[0:fukusho_ken].index(pred_list[0])
                tekichu+=1
                profit += bet*fukusho_odds[tekichu_index]
                race_hit_dist[str(race_id)[-2:]] += 1

    # -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)
    

def calc_wide(all_results,odds_alpha=2.0,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    
    # 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:
        is_buy = True
        ar = all_results.loc[race_id]
        pred_list = ar['pred_list']
        wide_comb = ar['wide_comb']
        actual_list = ar['actual_rank_list']
        wide_odds = ar['wide_odds']
        pred_odds = ar['odds_list'][0]
        
        if not is_today:
            if pred_odds>=odds_alpha:
                profit -= bet
            else:
                is_buy=False

        else:
            profit -= bet*length
            
        if sorted(pred_list[0:2]) in wide_comb and is_buy:
            tekichu_index = wide_comb.index(sorted(pred_list[0:2]))
            tekichu+=1
            profit += bet*wide_odds[tekichu_index]
            race_hit_dist[str(race_id)[-2:]] += 1

# -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)

def calc_umaren(all_results,odds_alpha=2.0,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    
    # 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:
        is_buy = True
        ar = all_results.loc[race_id]
        pred_list = sorted(ar['pred_list'][0:2])
        actual_list = sorted(ar['actual_rank_list'][0:2])
        umaren_odds = ar['umaren_odds']
        pred_odds = ar['odds_list'][0]
        
        if not is_today:
            if pred_odds>=odds_alpha:
                profit -= bet
            else:
                is_buy=False

        else:
            profit -= bet*length
            
        if  pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*umaren_odds
            race_hit_dist[str(race_id)[-2:]] += 1

# -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)

def calc_umatan(all_results,odds_alpha=2.0,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    
    # 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:
        is_buy = True
        ar = all_results.loc[race_id]
        pred_list = ar['pred_list'][0:2]
        actual_list = ar['actual_rank_list'][0:2]
        umatan_odds = ar['umatan_odds']
        pred_odds = ar['odds_list'][0]
        
        if not is_today:
            if pred_odds>=odds_alpha:
                profit -= bet
            else:
                is_buy=False

        else:
            profit -= bet*length
            
        if  pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*umatan_odds
            race_hit_dist[str(race_id)[-2:]] += 1

# -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)


def calc_sanrenpuku(all_results,odds_alpha=2.0,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    
    # 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:
        is_buy = True
        ar = all_results.loc[race_id]
        pred_list = sorted(ar['pred_list'][0:3])
        actual_list = sorted(ar['actual_rank_list'][0:3])
        sanrenpuku_odds = ar['sanrenpuku_odds']
        pred_odds = ar['odds_list'][0]
        
        if not is_today:
            if pred_odds>=odds_alpha:
                profit -= bet
            else:
                is_buy=False

        else:
            profit -= bet*length
            
        if pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*sanrenpuku_odds
            race_hit_dist[str(race_id)[-2:]] += 1

# -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)


def calc_sanrentan(all_results,odds_alpha=2.0,bet=100,is_today=False):
    length = len(all_results)
    tekichu = 0
    profit = 0
    bet = 100
    race_hit_dist = {'{}'.format(str(i).zfill(2)):0 for i in range(1,13)}
    
    # 将来的にここだけ関数化
# ----------------------------
    for race_id in all_results.index:
        is_buy = True
        ar = all_results.loc[race_id]
        pred_list = ar['pred_list'][0:3]
        actual_list = ar['actual_rank_list'][0:3]
        sanrentan_odds = ar['sanrentan_odds']
        pred_odds = ar['odds_list'][0]
        
        if not is_today:
            if pred_odds>=odds_alpha:
                profit -= bet
            else:
                is_buy=False

        else:
            profit -= bet*length
            
        if pred_list == actual_list and is_buy:
            tekichu+=1
            profit += bet*sanrentan_odds
            race_hit_dist[str(race_id)[-2:]] += 1

# -----------------------------------

    print('的中率 {0}'.format(tekichu/length))
    print("収益   {0} 円".format(profit))
    print('race dist',race_hit_dist)

