import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self):
        self.columns = []

    def read_file(self, TRAIN_PATH, TEST_PATH=None, sep=','):
        train_df = pd.read_csv(TRAIN_PATH, delimiter=sep)
        self.columns = [*train_df.columns]

        if TEST_PATH is not None:
            test_df = pd.read_csv(TEST_PATH, delimiter=sep)
            return train_df, test_df

        return train_df

    def verbose(self, df, columns=[]):
        title      = 'Missing Data Info'
        columns    = df.columns if len(columns) == 0 else columns
        nb_of_null = [[col, len(df[df[col].isna()]), df[col].dtype] for col in columns]
        null_df    = pd.DataFrame(nb_of_null)
        null_df.columns = ['Columns', 'Count', 'DataType']

        print('{} {} {}'.format('=' * 14, 'Data Info', '=' * 14))
        print('Number of rows: {}'.format(len(df)))
        print('Number of cols: {}'.format(len(columns)))

        print('{} {} {}'.format('=' * 10, title, '=' * 10))
        print(null_df)
        print('=' * (22 + len(title)))

    def plot(self, df, columns=[]):
        columns = self.columns if len(columns) == 0 else columns

        for i, c in enumerate(columns):
            k, v = np.unique(df[c], return_counts=True)

            plt.figure()
            plt.xlabel(c)
            plt.ylabel('Counts')
            plt.title('Unique value counts (Features: {}) '.format(c))
            plt.bar(k, v, label=c)

    def clean(self, df, verbose=True, auto_fillna=True):
        assert type(df) == pd.DataFrame, 'Make sure df is a pandas dataframe'

        # display the missing information
        if verbose: self.verbose(df, columns=df.columns)

        # add in the survived column if missing
        if 'Survived' not in df.columns: df['Survived'] = np.zeros(df.shape[0])

        # convert the decimal age to proper age value
        df.loc[df.Age < 1, 'Age'] = df.loc[df.Age < 1, 'Age'] * 100

        # fill the missing Nan
        if auto_fillna:
            df = df.drop(columns=['Cabin'])
            df['Age'] = df['Age'].fillna(value=int(df['Age'].median()))
            df['Fare'] = df['Fare'].fillna(value=int(df['Fare'].median()))
            df['Embarked'] = df['Embarked'].fillna(value='S')
            self.columns = [*df.columns]

        # features engineering
        df = self.features_engineer(df)

        # one hot encode columns
        df = self.one_hot_encode_col(df, 'Sex')
        df = self.one_hot_encode_col(df, 'Embarked')
        # df = self.one_hot_encode_col(df, 'Title')

        return df

    @staticmethod
    def features_engineer(df):
        df['FamilySize'] = df.SibSp + df.Parch + 1

        df['IsAlone'] = np.ones(df.shape[0])
        df.loc[df.FamilySize > 1, 'IsAlone'] = 0

        df['HighClassFemale'] = np.zeros(df.shape[0])
        df.loc[(df.Pclass == 1) & (df.Sex == 'female'), 'HighClassFemale'] = 1

        # df['HighClassMale'] = np.zeros(df.shape[0])
        # df.loc[(df.Embarked == 'C') & (df.Fare >= 50), 'HighClassMale'] = 1

        df['HighClassFamily'] = np.zeros(df.shape[0])
        df.loc[(df.Pclass == 1) & (df.FamilySize <= 4) & (df.IsAlone == 0), 'HighClassFamily'] = 1

        df['Title'] = df['Name'].str.split(", ", expand=True)[1] \
            .str.split(".", expand=True)[0]

        df['Title_Mrs'] = np.zeros(df.shape[0])
        df.loc[df.Title == 'Mrs', 'Title_Mrs'] = 1

        df['Title_Miss'] = np.zeros(df.shape[0])
        df.loc[df.Title == 'Miss', 'Title_Miss'] = 1

        df = df.drop(columns=['Title'])

        return df

    @staticmethod
    def one_hot_encode_col(df, column_name):
        dummies = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, dummies], axis=1)

        return df.drop([column_name], axis=1)

    @staticmethod
    def plot_correlation(df):
        assert type(df) == pd.DataFrame, 'Please ensure df is a dataframe type'

        corr = df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        plt.figure()
        plt.title('Pearson Correlation between features')
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r')

    @staticmethod
    def train_valid_split(df, split=0.8):
        train_idx = np.random.rand(len(df)) < split
        train_df  = df[train_idx]
        valid_df  = df[~train_idx]

        return train_df, valid_df
