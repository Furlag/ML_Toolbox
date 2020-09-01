#  region --------------------------------------------- Imports --------------------------------------------------------
# data analysis and wrangling
import pandas as pd
import scipy.stats as stats
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style

# Machine learning regresors
import xgboost as xgb
import catboost as catboost
import lightgbm as lightgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

# Gridsearch CV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Others
from math import sqrt
from IPython.display import display
from sklearn.base import clone

plt.style.use('default')
# plt.style.use('dark_background')
#  endregion


#######################################################################################################################
#  region --------------------------------- Data Analysis and visualizations ------------------------------------------

def test_normality(y):
    ''' Typical info needed to test normality on y'''
    def shapiro_test(y):
        result = stats.shapiro(y)
        print("Normality test on Shapiro-Wilk:", result)

    def kolmogorov_test(y):
        result = stats.kstest(y, 'norm', args=(np.mean(y), np.std(y)))
        print("Normality test on Anderson-Darling:", result)

    def anderson_test(y):
        result = stats.anderson(y, dist='norm')
        print("Normality test on Anderson-Darling:", result)

    shapiro_test(y)
    kolmogorov_test(y)
    anderson_test(y)

    # skewness and kurtosis
    print("Skewness:", y.skew(), "/ Kurtosis:", y.kurt())

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            style.use('fivethirtyeight')

            # Creating a customized chart. and giving in figsize and everything
            fig = plt.figure(constrained_layout=True, figsize=(8, 12))
            # creating a grid of 3 cols and 3 rows
            grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

            ax1 = fig.add_subplot(grid[0, :2])
            ax1.set_title('Johnson SU')
            sns.distplot(y, kde=False, fit=stats.johnsonsu, ax=ax1)

            ax2 = fig.add_subplot(grid[1, :2])
            ax2.set_title('Normal')
            sns.distplot(y, kde=False, fit=stats.norm, ax=ax2)

            ax3 = fig.add_subplot(grid[2, :2])
            ax3.set_title('Log Normal')
            sns.distplot(y, kde=False, fit=stats.lognorm, ax=ax3)

            fig.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            None
        else:
            # Other type (?)
            None

    except NameError:
        # Probably standard Python interpreter
        None


def visualize_Bifeature_scatterplot(df, vector_x, vector_y):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            style.use('fivethirtyeight')

            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            grid = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
            ax1 = fig.add_subplot(grid[0, 0])

            # Set the title
            ax1.set_title('ScatterPlot')

            # plot the scatterplot
            sns.scatterplot(data=df, x=vector_x, y=vector_y, ax=ax1)

            display(grid)

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print("Error: Plase, run in Jupyter cell")
        else:
            # Other type (?)
            print("Error: Plase, run in Jupyter cell")

    except NameError:
        # Probably standard Python interpreter
        print("Error: Plase, run in Jupyter cell")


def visualize_scatterplot_Grid(df, vector_y):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            numeric_X = df.select_dtypes(exclude=['object'])

            # Visualization with boxplots:
            fig, ax = plt.subplots(figsize=(15, 45))

            index = 1
            for col in numeric_X:
                plt.subplot(15, 4, index)
                sns.scatterplot(data=numeric_X.dropna(), x=col, y=vector_y)
                index = index + 1

            fig.tight_layout(pad=1.0)
            plt.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print("Error: Plase, run in Jupyter cell")
        else:
            # Other type (?)
            print("Error: Plase, run in Jupyter cell")

    except NameError:
        # Probably standard Python interpreter
        print("Error: Plase, run in Jupyter cell")


def visualize_boxplot_Grid(df):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            numeric_X = df.select_dtypes(exclude=['object'])

            # Visualization with boxplots:
            fig, ax = plt.subplots(figsize=(15, 45))

            index = 1
            for col in numeric_X:
                plt.subplot(15, 4, index)
                sns.boxplot(y=col, data=numeric_X.dropna())
                index = index + 1

            fig.tight_layout(pad=1.0)
            plt.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print("Error: Plase, run in Jupyter cell")
        else:
            # Other type (?)
            print("Error: Plase, run in Jupyter cell")

    except NameError:
        # Probably standard Python interpreter
        print("Error: Plase, run in Jupyter cell")


def visualize_feature_info(feature):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            style.use('fivethirtyeight')

            # Creating a customized chart. and giving in figsize and everything
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            # creating a grid of 3 cols and 3 rows
            grid = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

            # Customizing the histogram grid
            ax1 = fig.add_subplot(grid[0, :2])
            # Set the title
            ax1.set_title('Histogram')
            # plot the histogram
            sns.distplot(feature, norm_hist=True, ax=ax1)

            # customizing the QQ_plot
            ax2 = fig.add_subplot(grid[1, :2])
            # Set the title
            ax2.set_title('QQ_plot')
            # Plotting the QQ_Plot
            stats.probplot(feature, plot=ax2)

            # Customizing the Box Plot
            ax3 = fig.add_subplot(grid[:, 2])
            # Set title
            ax3.set_title('Box Plot')
            # Plotting the box plot
            sns.boxplot(feature, orient='v', ax=ax3)

            fig.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print("Error: Plase, run in Jupyter cell")
        else:
            # Other type (?)
            print("Error: Plase, run in Jupyter cell")

    except NameError:
        # Probably standard Python interpreter
        print("Error: Plase, run in Jupyter cell")


def show_feature_correlation(df, dependant_feature_name):
    numeric_X = df.select_dtypes(exclude=['object'])
    corr = numeric_X.corr()

    info = corr[[dependant_feature_name]].sort_values([dependant_feature_name], ascending=False)

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            plt.figure(figsize=(5, 15))
            sns.heatmap(
                corr[[dependant_feature_name]].sort_values(by=['SalePrice'], ascending=False),
                annot_kws={"size": 16}, vmin=-1, cmap='PiYG', annot=True)
            plt.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print(info)
        else:
            # Other type (?)
            print(info)

    except NameError:
        # Probably standard Python interpreter
        print(info)


def visualize_correlation_heatmap(df, method='pearson', mask_threshold=-1, figsize=(20, 15)):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            numeric_X = df.select_dtypes(exclude=['object'])

            plt.figure(1)
            plt.subplots(figsize=figsize)
            corr = numeric_X.corr(method=method)

            # colormap = sns.diverging_palette(220, 10, as_cmap=True)
            colormap = 'PiYG'
            sns.heatmap(corr, xticklabels=True, yticklabels=True, center=0,
                        annot=True, annot_kws={'size': 7}, square=True, linewidths=0.1, linecolor='black',
                        linewidth=0.5, cmap=colormap, mask=corr < mask_threshold)
            plt.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print("Error: Plase, run in Jupyter cell")
        else:
            # Other type (?)
            print("Error: Plase, run in Jupyter cell")

    except NameError:
        # Probably standard Python interpreter
        print("Error: Plase, run in Jupyter cell")


def show_missing_values_info(df):
    """This function takes a DataFrame(df) as input and returns two columns,
        total missing values and total missing values percentage"""

    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100, 2)[
              round(df.isnull().sum().sort_values(ascending=False)/len(df)*100, 2) != 0
              ]

    info = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            display(info)

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print(info)
        else:
            # Other type (?)
            print(info)

    except NameError:
        # Probably standard Python interpreter
        print(info)


def show_feature_importance(X, y, model=RandomForestRegressor(random_state=42, n_jobs=4)):
    model.fit(X, y)

    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    feature_importances = feature_importances.reset_index(drop=True)

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole

            # display(feature_importances)

            plt.figure(figsize=(20, 10))
            sns.barplot(x="Importance", y="Feature",
                        data=feature_importances.sort_values(by="Importance", ascending=False).iloc[:50])
            plt.title('Feature Impportances')
            plt.tight_layout()
            plt.show()

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print(feature_importances)
        else:
            # Other type (?)
            print(feature_importances)

    except NameError:
        # Probably standard Python interpreter
        print(feature_importances)

#  endregion


#######################################################################################################################
#  region ----------------------------------- Data preparation and wragling -------------------------------------------

class Apply_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_imputer_type='mean', cat_imputer_type='most_frequent', cat_candidates=[], verbose=False):
        self.num_imputer_type = num_imputer_type
        self.cat_imputer_type = cat_imputer_type
        self.cat_candidates = cat_candidates
        self.verbose = verbose

    def transform(self, X):
        X_copy = X.copy()

        ''' Imputation for numerical columns '''
        # Impute cat_candidates (Numerical columns that have been identified to work like categorical ones):
        if len(self.num_cat_cols) > 0:
            cat_like_imputer = SimpleImputer(strategy=self.cat_imputer_type, fill_value=0)
            X_copy[self.num_cat_cols] = cat_like_imputer.fit_transform(X_copy[self.num_cat_cols])

        # Impute Numerical columns:
        if len(self.numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy=self.num_imputer_type, fill_value=0)
            X_copy[self.numerical_cols] = num_imputer.fit_transform(X_copy[self.numerical_cols])
        # _________________________________________________________________________________________________

        ''' Imputation for categorical columns'''
        # Impute Categorical data:
        if len(self.categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy=self.cat_imputer_type, fill_value='missing')
            X_copy[self.categorical_cols] = cat_imputer.fit_transform(X_copy[self.categorical_cols])

        # _________________________________________________________________________________________________

        # Change dtypes to original ones
        for feat in X_copy.columns:
            X_copy[feat] = X_copy[feat].astype(X[feat].dtype.name)

        # Print info
        print("Numerical imputer applied ->", self.num_imputer_type, "to features:",
              (self.numerical_cols + self.num_cat_cols),
              "\nCategorical imputer applied ->", self.cat_imputer_type) if self.verbose is True else None

        return X_copy

    def fit(self, X, y=None):

        # Identify numerical columns
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        self.num_cat_cols = []
        if len(self.cat_candidates) > 0:
            for feat in self.cat_candidates:
                if feat in self.numerical_cols:
                    self.numerical_cols.remove(feat)
                    self.num_cat_cols.append(feat)

        # Identify categorical columns
        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

        return self


class Apply_Power_Transformer(BaseEstimator, TransformerMixin):
    '''
        Apply skewness reductor to features with skewness above thresold.
        Valid nethods are: 'box-cox' and 'yeo-johnson'.
    '''

    def __init__(self, skew_threshold=0.5, method='yeo-johnson', ignore_features=[], verbose=False):
        self.skew_threshold = skew_threshold
        self.method = method
        self.ignore_features = ignore_features
        self.verbose = verbose

    def transform(self, X):
        transformer = PowerTransformer(method=self.method)
        X_copy = X.copy()

        # Apply transformer
        X_copy[self.high_skewed_columns] = transformer.fit_transform(X_copy[self.high_skewed_columns])

        print("Power transformation applied to columns:\n", self.high_skewed_columns) if self.verbose is True else None

        return X_copy

    def fit(self, X, y=None):
        # Select numerical cols
        numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        for feat in self.ignore_features:
            numerical_cols.remove(feat) if feat in numerical_cols else None

        # Identify columns to which transformer will be applied
        skewColumns = X[numerical_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewed_cols_info = skewColumns[abs(skewColumns) > self.skew_threshold]

        print("The following features have been identified as highly skewed:\n",
              skewed_cols_info) if self.verbose is True else None

        self.high_skewed_columns = list(skewed_cols_info.index)

        return self


class Apply_Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, chosen_Scaler='RobustScaler', ignore_features=[], verbose=False):
        self.chosen_Scaler = chosen_Scaler
        self.ignore_features = ignore_features
        self.verbose = verbose

    def transform(self, X):
        X_copy = X.copy()

        if self.chosen_Scaler == 'StandardScaler':
            scaler = StandardScaler().fit(X_copy[self.numerical_cols])

        elif self.chosen_Scaler == 'RobustScaler':
            scaler = RobustScaler().fit(X_copy[self.numerical_cols])

        elif self.chosen_Scaler == 'MinMaxScaler':
            scaler = MinMaxScaler().fit(X_copy[self.numerical_cols])

        elif self.chosen_Scaler == 'MaxAbsScaler':
            scaler = MaxAbsScaler().fit(X_copy[self.numerical_cols])

        else:
            print("Need valid Scaler. Valid types are 'StandardScaler', 'RobustScaler' and 'MinMaxScaler' ")
            return EOFError

        # Apply the Scaler to numerical columns
        X_copy[self.numerical_cols] = scaler.transform(X_copy[self.numerical_cols])

        print("Scaler: {} applied to columns {}".format(self.chosen_Scaler, self.numerical_cols)
              ) if self.verbose is True else None

        return X_copy

    def fit(self, X, y=None):
        # Identify numerical columns
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        for feat in self.ignore_features:
            self.numerical_cols.remove(feat) if feat in self.numerical_cols else None

        return self


class Apply_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_type='Label', verbose=False):
        self.encoder_type = encoder_type
        self.verbose = verbose

    def transform(self, X):
        # Get numerical data
        num_X = X[self.numerical_cols].copy()

        # Get categorical data
        cat_X = X[self.categorical_cols].copy()

        # Encode categorical data:
        if len(self.categorical_cols) > 0:
            if self.encoder_type == 'dummie_OH':
                encoded_cat_X = pd.get_dummies(cat_X)

            elif self.encoder_type == 'OH':
                encoded_cat_X = pd.DataFrame(self.encoder.transform(cat_X))
                encoded_cat_X.index = cat_X.index

            elif self.encoder_type == 'Label':
                encoded_cat_X = cat_X.copy()

                i = 0
                for col in self.categorical_cols:
                    label_encoder = self.label_encoders_list[i]

                    encoded_cat_X[col] = label_encoder.transform(encoded_cat_X[col])

                    i = i + 1

            else:
                print("Error: Need valid encoder. Valid types are 'dummie_OH', 'OH' and 'Label'.")
                return EOFError

        if len(self.numerical_cols) > 0 and len(self.categorical_cols) > 0:
            encoded_X = pd.concat([num_X, encoded_cat_X], axis=1)
        elif len(self.numerical_cols) > 0 and len(self.categorical_cols) == 0:
            encoded_X = num_X.copy()
        elif len(self.numerical_cols) == 0 and len(self.categorical_cols) > 0:
            encoded_X = encoded_cat_X.copy()
        else:
            print("Error: No Numerical or Categorical cols have been found on dataframe")
            return EOFError

        encoded_X.index = X.index.copy()

        print("Data preprocessor completed: Encoder applied ->", self.encoder_type) if self.verbose is True else None

        return encoded_X

    def fit(self, X, y=None):
        # Get numerical columns
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        # Get categorical columns
        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

        # Get categorical data to fit on encoders
        cat_X = X[self.categorical_cols].copy()

        # Encode categorical data:
        if len(self.categorical_cols) > 0:
            if self.encoder_type == 'dummie_OH':
                None

            elif self.encoder_type == 'OH':
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                self.encoder.fit(cat_X)

            elif self.encoder_type == 'Label':
                self.label_encoders_list = []

                for col in self.categorical_cols:
                    label_encoder = LabelEncoder()
                    label_encoder.fit(cat_X[col])
                    self.label_encoders_list.append(label_encoder)

            else:
                print("Error: Need valid encoder. Valid types are 'dummie_OH', 'OH' and 'Label'.")
                return EOFError

        return self


class Create_unique_counts(BaseEstimator, TransformerMixin):
    def __init__(self, chosen_columns=[], verbose=False):
        self.chosen_columns = chosen_columns
        self.verbose = verbose

    def transform(self, X):
        X_copy = X.copy()

        for column in self.chosen_columns:
            counts_df = X_copy[column].value_counts().to_frame()

            counts_df = pd.merge(X[column], counts_df, how='left', left_on=column, right_index=True,
                                 suffixes=('_value', '_unique_count'))

            # Create new column name
            new_column_name = str(column) + '_unique_count'

            # Add new column to dataframe
            X_copy[new_column_name] = counts_df[new_column_name]

        return X_copy

    def fit(self, X, y=None):
        # Identify categorical columns if no columns were passed
        if len(self.chosen_columns) == 0:
            self.chosen_columns = [cname for cname in X.columns if X[cname].dtype == "object"]

        return self


class Equalize_features(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, remodel):
        remodel_copy = remodel.copy()

        for column in self.X_columns:
            if column not in remodel_copy:
                remodel_copy[column] = 0
        remodel_copy = remodel_copy[self.X_columns]

        return remodel_copy

    def fit(self, X, y=None):
        self.X_columns = X.columns

        return self


class Remove_features_with_mostly_one_value(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.96, verbose=False):
        self.threshold = threshold
        self.verbose = verbose

    def transform(self, X):
        ''' Features that have more that threshold of just one value will be dropped. '''
        X_copy = X.copy()

        X_copy = X_copy.drop(self.overfit, axis=1)

        print("Features with >", self.threshold, "% of the same value:", self.overfit,
              "were removed") if self.verbose is True else None
        return X_copy

    def fit(self, X, y=None):
        self.overfit = []
        for i in X.columns:
            counts = X[i].value_counts()
            zeros = counts.iloc[0]
            if zeros / len(X) > self.threshold:
                self.overfit.append(i)

        self.overfit = list(self.overfit)

        return self


class Assign_dtype_to_features(BaseEstimator, TransformerMixin):
    def __init__(self, features=[], change_to_type='object'):
        self.change_to_type = change_to_type
        self.features = features

    def transform(self, X):
        X_copy = X.copy()

        for feat in self.features:
            if feat in X_copy.columns:
                X_copy[feat] = X_copy[feat].astype(self.change_to_type)

        return X_copy

    def fit(self, X, y=None):

        self.features = X.columns if len(self.features) == 0 else None

        return self


class Get_columns_names_by_unique_counts(BaseEstimator, TransformerMixin):
    def __init__(self, numinque_threshold=20):
        self.numinque_threshold = numinque_threshold

    def transform(self, X):
        valid_columns = []

        for col in self.X_columns:
            if X[col].numinque() > self.numinque_threshold:
                valid_columns.append(col)

        return valid_columns

    def fit(self, X, y=None):
        self.X_columns = X.columns

        return self

#  endregion


# region ------------------------------------------- Pycaret classes ---------------------------------------------------

class Replace_New_Catagorical_Levels_in_TestData(BaseEstimator, TransformerMixin):
    '''
        -This treats if a new level appears in the test dataset catagorical's feature
            (i.e a level on whihc model was not trained previously)
        -It simply replaces the new level in test data set with the most frequent or least frequent level
            in the same feature in the training data set
        -It is recommended to run the Zroe_NearZero_Variance and Define_dataTypes first
        -Ignores target variable
        Args:
            target: string , name of the target variable
            replacement_strategy:string , 'least frequent' or 'most frequent' (default 'most frequent' )
    '''

    def __init__(self, target, replacement_strategy='most frequent'):
        self.target = target
        self.replacement_strategy = replacement_strategy

    def fit(self, data, y=None):
        # need to make a place holder that keep records of all the levels,
        # and in case a new level appears in test we will change it to others
        self.ph_train_level = pd.DataFrame(
            columns=data.drop(self.target, axis=1).select_dtypes(include="object").columns
            )
        for i in self.ph_train_level.columns:
            if self.replacement_strategy == "least frequent":
                self.ph_train_level.loc[0, i] = list(data[i].value_counts().sort_values().index)
            else:
                self.ph_train_level.loc[0, i] = list(data[i].value_counts().index)

    def transform(self, data, y=None):
        # transorm
        # we need to learn the same for test data , and then we will compare to check what levels are new in there
        self.ph_test_level = pd.DataFrame(
            columns=data.drop(self.target, axis=1, errors='ignore').select_dtypes(include="object").columns
            )
        for i in self.ph_test_level.columns:
            self.ph_test_level.loc[0, i] = list(data[i].value_counts().sort_values().index)

        # new we have levels for both test and train, we will start comparing and replacing levels in test set
        # (Only if test set has new levels)
        for i in self.ph_test_level.columns:
            new = list((set(self.ph_test_level.loc[0, i]) - set(self.ph_train_level.loc[0, i])))
        # now if there is a difference , only then replace it
        if len(new) > 0:
            data[i].replace(new, self.ph_train_level.loc[0, i][0], inplace=True)

        return(data)

    def fit_transform(self, data, y=None):  # There is no transformation happening in training data set
        self.fit(data)
        return(data)

# endregion


# region ----------------------------------------- Feature Selection --------------------------------------------------

class Apply_ImportanceBased_feature_selection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=50, model=RandomForestRegressor(random_state=42, n_jobs=4)):
        self.n_components = n_components
        self.model = model

    def transform(self, X, y):
        self.n_components = len(X) if self.n_components > len(X) else None
        self.model.fit(X, y)

        # Get feature importances and sort features by it
        feature_importances_pd = pd.DataFrame({'feature': X.columns, 'importance': self.model.feature_importances_})
        feature_importances_pd = feature_importances_pd.sort_values('importance', ascending=False)
        feature_importances_pd = feature_importances_pd.reset_index(drop=True)

        # Get top n_components features and parse them as a list
        feature_importances_pd = feature_importances_pd.head(self.n_components)
        features_names_by_importance = list(feature_importances_pd['feature'])

        return X[features_names_by_importance]

    def fit(self, X, y):
        return self


class Apply_PCA_feature_selection(BaseEstimator, TransformerMixin):
    def __init__(self, PCA_model=PCA(n_components=50), verbose=False):
        self.PCA_model = PCA_model
        self.verbose = verbose

    def transform(self, X):
        # Other option is to chose a % n_components to keep ej: PCA(n_components=0.95)
        self.PCA_model.fit(X)

        # Info
        n_pcs = self.PCA_model.components_.shape[0]

        if self.verbose:
            print("Total features selected by PCA =", n_pcs)
            print("Noise variance =", self.PCA_model.noise_variance_)
            try:
                shell = get_ipython().__class__.__name__  # noqa
                if shell == 'ZMQInteractiveShell':
                    # Jupyter notebook or qtconsole

                    # Plot explained variance progression of PCA
                    plt.figure(constrained_layout=True, figsize=(8, 6))
                    plt.title("PCA variance explanation")
                    plt.plot(np.cumsum(PCA_model.explained_variance_ratio_))
                    plt.xlabel('number of components')
                    plt.ylabel('cumulative explained variance')
                    plt.show()

                elif shell == 'TerminalInteractiveShell':
                    # Terminal running IPython
                    None
                else:
                    # Other type (?)
                    None
            except NameError:
                # Probably standard Python interpreter
                None

        # Get best features
        initial_feature_names = X.columns
        most_important_features = [np.abs(self.PCA_model.components_[i]).argmax() for i in range(n_pcs)]
        most_important_feature_names = [initial_feature_names[most_important_features[i]] for i in range(n_pcs)]

        # X_PCA = PCA_model.transform(X)

        print("PCA selected features:", most_important_feature_names) if self.verbose else None
        return X[most_important_feature_names]

    def fit(self, X, y=None):
        return self

#  endregion


#######################################################################################################################
#  region ---------------------------------------------- Other --------------------------------------------------------

def evaluate_model(X, y, model, score_method='neg_mean_absolute_error', n_jobs=4):
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                  scoring=score_method, n_jobs=n_jobs)

    print("Scores:\n", scores)
    print("Average " + str(score_method) + " score (across experiments):" + str(scores.mean()))

#  endregion


#######################################################################################################################
#  region ------------------------------ Cross validated search of best model ------------------------------------------

def find_best_CatBoost_model_CV(X, y, model=catboost.CatBoostRegressor(verbose=False, random_state=42),
                                cat_candidates=[],
                                pipeline=None,
                                score_method='neg_mean_absolute_error',
                                splits=KFold(n_splits=4, shuffle=False, random_state=42),
                                early_stopping_rounds=400,
                                randomized_search=False, n_iter=500,
                                n_jobs=1):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    # Get categorical cols to fit on model -----------------------------------------------------------
    categorical_features = [cname for cname in X.columns if X[cname].dtype == 'object']
    if len(cat_candidates) > 0:
        for feat in cat_candidates:
            if feat in X.columns and feat not in categorical_features:
                categorical_features.append(feat)
    # -------------------------------------------------------------------------------------------------

    # Save a processed dataset for fitting the model later
    processed_X = production_pipeline.fit_transform(X)

    # Append model to pipeline
    production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__learning_rate': [0.1, 0.05, 0.01],
            'Model__n_estimators': range(500, 10000, 250),
            'Model__depth': range(1, 12, 1),
            'Model__l2_leaf_reg': [1, 3, 5, 7, 9],
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__bagging_temperature': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
                                           0.4, 0.7, 1, 4, 7, 10, 40, 70, 100]
            }
        random_search = GridSearchCV(estimator=production_pipeline,
                                     param_grid=param_test_random_search, scoring=score_method,
                                     n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        random_search.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 1
        param_test1 = {
            'Model__n_estimators': range(10, 110, 10)
            }
        search1 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test1, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search1.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(search1.best_params_))
        print("Best score achived: " + str(search1.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search1.best_estimator_

        # Param search 2
        param_test2 = {
            'Model__depth': range(4, 10, 1)
            }
        search2 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test2, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search2.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(search2.best_params_))
        print("Best score achived: " + str(search2.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search2.best_estimator_

        '''
        # Param search 3 | only available when training on CPU
        param_test3 = {
            'Model__l2_leaf_reg': [None, 0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
                                   0.4, 0.7, 1, 4, 7, 10],
            'Model__random_strength': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
                                       0.4, 0.7, 1, 4, 7, 10]
            }
        search3 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test3, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search3.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(search3.best_params_))
        print("Best score achived: " + str(search3.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search3.best_estimator_

        # Param search 4
        param_test4 = {
            'Model__bagging_temperature': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
                                           0.4, 0.7, 1, 4, 7, 10]
            }
        search4 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test4, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search4.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(search4.best_params_))
        print("Best score achived: " + str(search4.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search4.best_estimator_
        '''
        # Param search 5 - round optimization and learning rate reduction
        param_test5 = {
            'Model__learning_rate': [0.01],
            'Model__n_estimators': range(1000, 9500, 500)
            }
        search5 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test5, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search5.fit(X, y, Model__cat_features=categorical_features)
        print("Best params encountered: " + str(search5.best_params_))
        print("Best score achived: " + str(search5.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search5.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y, cat_features=categorical_features)

    return model


def find_best_LightGBM_model_CV(X, y, model=lightgbm.LGBMRegressor(
                                    objective='regression', boosting_type='gbdt', learning_rate=0.1, random_state=42),
                                pipeline=None,
                                score_method='neg_mean_absolute_error',
                                splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__boosting': ['gbdt', 'rf', 'dart', 'goss'],
            'Model__learning_rate': [0.01, 0.05],
            'Model__n_estimators': range(500, 9000, 500),
            'Model__max_depth': range(1, 15, 1),
            'Model__num_leaves': range(2, 256, 2),
            'Model__min_data_in_leaf': range(2, 256, 2),
            'Model__min_gain_to_split': [
                0, 0.00001, 0.00004, 0.00007, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007,
                0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10],
            'Model__lambda_l1': range(2, 1000, 2),
            'Model__lambda_l2': range(2, 1000, 2),
            'Model__feature_fraction': [
                0, 0.00001, 0.00004, 0.00007, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
                0.4, 0.7, 1]
            }

        # Used only in binary classification
        '''
        'Model__bagging_fraction': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
        'Model__bagging_freq': range(1, 100, 1),
        '''

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__n_estimators': range(20, 110, 10)
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Param search 1
        param_test1 = {
            'Model__max_depth': range(-1, 11, 1)
            }
        search1 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test1, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search1.fit(X, y)
        print("Best params encountered: " + str(search1.best_params_))
        print("Best score achived: " + str(search1.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search1.best_estimator_

        # Param search 2
        param_test2 = {
            'Model__num_leaves': range(2, 30, 2),
            'Model__min_data_in_leaf': range(2, 30, 2)
            }
        search2 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test2, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search2.fit(X, y)
        print("Best params encountered: " + str(search2.best_params_))
        print("Best score achived: " + str(search2.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search2.best_estimator_

        # Param search 3
        param_test3 = {
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__colsample_bytree': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            }
        search3 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test3, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search3.fit(X, y)
        print("Best params encountered: " + str(search3.best_params_))
        print("Best score achived: " + str(search3.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search3.best_estimator_

        # Param search 3
        '''
        param_test3 = {
            'Model__feature_fraction': [0.01, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                                        0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__feature_fraction_seed': [42]
            }
        search3 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test3, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search3.fit(X, y)
        print("Best params encountered: " + str(search3.best_params_))
        print("Best score achived: " + str(search3.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search3.best_estimator_
        '''
        # Param search 4
        param_test4 = {
            'Model__bagging_fraction': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__bagging_freq': range(1, 30, 2)
            }
        search4 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test4, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search4.fit(X, y)
        print("Best params encountered: " + str(search4.best_params_))
        print("Best score achived: " + str(search4.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search4.best_estimator_

        # Param search 5 - Round optimization
        param_test5 = {
            'Model__learning_rate': [0.01],
            'Model__n_estimators': range(1000, 9500, 500)
            }
        search5 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test5, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search5.fit(X, y)
        print("Best params encountered: " + str(search5.best_params_))
        print("Best score achived: " + str(search5.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search5.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_GradientBoosting_model_CV(X, y, model=GradientBoostingRegressor(
                                            random_state=42, subsample=0.8, learning_rate=0.1),
                                        pipeline=None,
                                        score_method='neg_mean_absolute_error',
                                        splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                        randomized_search=False, n_iter=500, n_jobs=4):
    model.set_params(min_samples_split=round(0.0075*len(X.index)))

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__learning_rate': [0.01, 0.05],
            'Model__n_estimators': range(1000, 9000, 500),
            'Model__max_depth': range(1, 11, 1),
            'Model__min_samples_split': range(0, round(0.02*len(X.index)), round(0.002*len(X.index))),
            'Model__min_samples_leaf': range(1, 120, 2),
            'Model__max_features':
                [None, 'auto', 'sqrt', 'log2',
                 round(sqrt(len(X.columns))), round(0.1*len(X.columns)), round(0.15*len(X.columns)),
                 round(0.2*len(X.columns)), round(0.25*len(X.columns)),
                 round(0.3*len(X.columns)), round(0.35*len(X.columns)), round(0.4*len(X.columns))
                 ],
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            }
        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__n_estimators': range(100, 1010, 100)
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Param search 1
        param_test1 = {
            'Model__max_depth': range(2, 11, 1)
            }
        search1 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test1, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search1.fit(X, y)
        print("Best params encountered: " + str(search1.best_params_))
        print("Best score achived: " + str(search1.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search1.best_estimator_

        # Param search 2
        param_test2 = {
            'Model__max_features':
                [None, 'auto', 'sqrt', 'log2',
                 round(sqrt(len(X.columns))), round(0.1*len(X.columns)), round(0.15*len(X.columns)),
                 round(0.2*len(X.columns)), round(0.25*len(X.columns)),
                 round(0.3*len(X.columns)), round(0.35*len(X.columns)), round(0.4*len(X.columns))
                 ],
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            }
        search2 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test2, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search2.fit(X, y)
        print("Best params encountered: " + str(search2.best_params_))
        print("Best score achived: " + str(search2.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search2.best_estimator_

        # Param search 3
        param_test3 = {
            'Model__min_samples_split': range(2, round(0.02*len(X.index)), round(0.002*len(X.index))),
            'Model__alpha': [0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.5,
                             0.6, 0.07, 0.9, 0.1, 0.4, 0.7, 1]
            }
        search3 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test3, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search3.fit(X, y)
        print("Best params encountered: " + str(search3.best_params_))
        print("Best score achived: " + str(search3.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search3.best_estimator_

        # Param search 4 - Round optimization
        param_test4 = {
            'Model__learning_rate': [0.01],
            'Model__n_estimators': range(1000, 9500, 500)
            }
        search4 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test4, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search4.fit(X, y)
        print("Best params encountered: " + str(search4.best_params_))
        print("Best score achived: " + str(search4.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search4.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_XGboost_model_CV(X, y, model=xgb.XGBRegressor(
                                    random_state=42, objective='reg:squarederror',
                                    subsample=0.8, colsample_bytree=0.8),
                               pipeline=None,
                               score_method='neg_mean_absolute_error',
                               splits=KFold(n_splits=5, shuffle=True, random_state=42),
                               randomized_search=False, n_iter=500,
                               n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__learning_rate': [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.2, 0.3],
            'Model__n_estimators': range(1000, 9000, 500),
            'Model__max_depth': range(1, 12, 1),
            'Model__gamma': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7,
                             1, 4, 7, 10, 40, 70, 100],
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__colsample_bytree': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__reg_alpha': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7,
                                 1, 4, 7, 10, 40, 70, 100]
            }
        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__n_estimators': range(20, 110, 10)
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Param search 1
        param_test1 = {
            'Model__max_depth': range(2, 10, 1),
            'Model__min_child_weight': range(1, 10, 1)
            }
        search1 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test1, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search1.fit(X, y)
        print("Best params encountered: " + str(search1.best_params_))
        print("Best score achived: " + str(search1.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search1.best_estimator_

        # Param search 2
        param_test2 = {
            'Model__subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'Model__colsample_bytree': [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            }
        search2 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test2, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search2.fit(X, y)
        print("Best params encountered: " + str(search2.best_params_))
        print("Best score achived: " + str(search2.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search2.best_estimator_

        # Param search 3
        param_test3 = {
            'Model__gamma': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7,
                             1, 4, 7, 10, 40, 70, 100]
            }
        search3 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test3, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search3.fit(X, y)
        print("Best params encountered: " + str(search3.best_params_))
        print("Best score achived: " + str(search3.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search3.best_estimator_

        # Param search 4
        param_test4 = {
            'Model__reg_alpha': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1,
                                 3, 5, 7, 9, 10],
            'Model__reg_lambda': [0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1,
                                  3, 5, 7, 9, 10]
            }
        search4 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test4, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search4.fit(X, y)
        print("Best params encountered: " + str(search4.best_params_))
        print("Best score achived: " + str(search4.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search4.best_estimator_

        # Round optimization
        param_test5 = {
            'Model__learning_rate': [0.01],
            'Model__n_estimators': range(1000, 9500, 500)
            }
        search5 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test5, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search5.fit(X, y)
        print("Best params encountered: " + str(search5.best_params_))
        print("Best score achived: " + str(search5.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search5.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit the model for later use
    model.fit(processed_X, y)

    return model


def find_best_Ridge_model_CV(X, y, model=Ridge(random_state=42, max_iter=None),
                             pipeline=None,
                             score_method='neg_mean_absolute_error',
                             splits=KFold(n_splits=5, shuffle=True, random_state=42),
                             randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__alpha': [-3, -2, -1, 1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1, 1.5, 2, 3, 4, 5,
                             10, 20, 30, 40]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__alpha': [-3, -2, -1, 1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1, 1.5, 2, 3, 4, 5,
                             10, 20, 30, 40]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_KernelRidge_model_CV(X, y, model=KernelRidge(),
                                   pipeline=None,
                                   score_method='neg_mean_absolute_error',
                                   splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                   randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__alpha': [2.0, 2.2, 2.4, 2.6],
            'Model__gamma': [0.0001, 0.001, 0.01, 0.1],
            'Model__degree': [1, 2, 3, 4, 5, 6],
            'Model__coef0': [0.1, 0.3, 0.5, 1.0, 2.0]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__alpha': [2.0, 2.2, 2.4, 2.6],
            'Model__gamma': [0.0001, 0.001, 0.01, 0.1],
            'Model__degree': [1, 2, 3, 4, 5, 6],
            'Model__coef0': [0.1, 0.3, 0.5, 1.0, 2.0]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_ElasticNet_model_CV(X, y, model=ElasticNet(),
                                  pipeline=None,
                                  score_method='neg_mean_absolute_error',
                                  splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                  randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                             0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 50, 100],
            'Model__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1],
            'Model__fit_intercept': [True],  # ,False
            'Model__normalize': [False],  # True,
            'Model__max_iter': range(50, 500, 50),
            'Model__selection': ['random'],  # 'cyclic',
            'Model__random_state': [None]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                             0.001, 0.01, 0.1, 0.5, 1],
            'Model__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1],
            'Model__fit_intercept': [True],  # ,False
            'Model__normalize': [False],  # True,
            'Model__max_iter': range(50, 500, 50),
            'Model__selection': ['random'],  # 'cyclic',
            'Model__random_state': [None]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_Lasso_model_CV(X, y, model=Lasso(random_state=42),
                             pipeline=None,
                             score_method='neg_mean_absolute_error',
                             splits=KFold(n_splits=5, shuffle=True, random_state=42),
                             randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__alpha': [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                             0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9,
                             1, 2, 3, 4, 5, 7, 9, 10, 20, 30],
            'Model__max_iter': [200, 500, 1000, 10000],
            'Model__tol': [0.005]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__alpha': [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                             0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9,
                             1, 2, 3, 4, 5, 7, 9, 10, 20, 30],
            'Model__max_iter': [200, 500, 1000, 10000],
            'Model__tol': [0.005]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_RandomForest_model_CV(X, y, model=RandomForestRegressor(random_state=42, n_jobs=4),
                                    pipeline=None,
                                    score_method='neg_mean_absolute_error',
                                    splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                    randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__bootstrap': [True, False],
            'Model__max_depth': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'Model__max_features': ['auto', 'sqrt'],
            'Model__min_samples_leaf': [1, 2, 4],
            'Model__min_samples_split': [2, 5, 10],
            'Model__n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__max_depth': range(1, 20, 2),
            'Model__n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Param search 1
        param_test1 = {
            'Model__max_features': [None, 'auto', 'sqrt'],
            'Model__min_samples_leaf': range(5, 15, 5),
            'Model__min_samples_split': range(10, 30, 10),
            'Model__criterion': ['mse', 'mae']
            }
        search1 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test1, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search1.fit(X, y)
        print("Best params encountered: " + str(search1.best_params_))
        print("Best score achived: " + str(search1.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search1.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model


def find_best_SVRegressorMachines_model_CV(X, y, model=SVR(gamma='auto', kernel='rbf', verbose=False),
                                           pipeline=None,
                                           score_method='neg_mean_absolute_error',
                                           splits=KFold(n_splits=5, shuffle=True, random_state=42),
                                           randomized_search=False, n_iter=500, n_jobs=4):

    if pipeline is None:
        production_pipeline = Pipeline(steps=[('Model', model)])

        # Nothing to process, just copy X
        processed_X = X.copy()
    else:
        production_pipeline = clone(pipeline)

        # Save a processed dataset for fitting the model later
        processed_X = production_pipeline.fit_transform(X)

        # Append model to pipeline
        production_pipeline.steps.append(['Model', model])

    if randomized_search:
        # Param search random
        param_test_random_search = {
            'Model__epsilon': [0.001, 0.01, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1, 10, 30, 70, 100],
            'Model__gamma': ['auto', 'scale', 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.0007,
                             0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1]
            }

        random_search = RandomizedSearchCV(estimator=production_pipeline, n_iter=n_iter,
                                           param_distributions=param_test_random_search, scoring=score_method,
                                           cv=splits, verbose=True, random_state=42, n_jobs=n_jobs)
        random_search.fit(X, y)
        print("Best params encountered: " + str(random_search.best_params_))
        print("Best score achived: " + str(random_search.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = random_search.best_estimator_

        model = production_pipeline.named_steps['Model']

    else:
        # Param search 0
        param_test0 = {
            'Model__epsilon': [0.001, 0.01, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1, 10, 30, 70, 100],
            'Model__gamma': ['auto', 'scale', 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.0007,
                             0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1]
            }
        search0 = GridSearchCV(estimator=production_pipeline,
                               param_grid=param_test0, scoring=score_method,
                               n_jobs=n_jobs, iid=False, cv=splits, verbose=True)
        search0.fit(X, y)
        print("Best params encountered: " + str(search0.best_params_))
        print("Best score achived: " + str(search0.best_score_))
        print("------------------------------------------------------------------------")
        production_pipeline = search0.best_estimator_

        # Get model from pipeline
        model = production_pipeline.named_steps['Model']

    # Fit model for later use
    model.fit(processed_X, y)

    return model

# endregion


#  region ---------------------------------------------- Tests --------------------------------------------------------


#  endregion
