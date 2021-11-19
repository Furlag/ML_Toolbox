#  region --------------------------------------------- Imports --------------------------------------------------------
# data analysis and wrangling
import pandas as pd
import scipy.stats as stats
import numpy as np
import math

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa explicitly require this experimental feature
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif, f_regression  # noqa
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style

# Machine learning regresors and classifiers
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Gridsearch CV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Others
from IPython.display import display
import sys
from sklearn.base import clone

# fuzz is used to compare TWO strings
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

plt.style.use('default')
# plt.style.use('dark_background')

#  endregion


#######################################################################################################################
#  region --------------------------------- Data Analysis and visualizations ------------------------------------------

def test_normality(serie, visualize=True):
    '''
        Typical info needed to test normality on y
    '''

    def shapiro_test(serie):
        result = stats.shapiro(serie)
        print("Normality test on Shapiro-Wilk:", result)

    def kolmogorov_test(serie):
        result = stats.kstest(serie, 'norm', args=(np.mean(serie), np.std(serie)))
        print("Normality test on Anderson-Darling:", result)

    def anderson_test(serie):
        result = stats.anderson(serie, dist='norm')
        print("Normality test on Anderson-Darling:", result)

    shapiro_test(serie)
    kolmogorov_test(serie)
    anderson_test(serie)

    # skewness and kurtosis
    print("Skewness:", serie.skew(), "/ Kurtosis:", serie.kurt())

    if visualize:
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
                sns.distplot(serie, kde=False, fit=stats.johnsonsu, ax=ax1)

                ax2 = fig.add_subplot(grid[1, :2])
                ax2.set_title('Normal')
                sns.distplot(serie, kde=False, fit=stats.norm, ax=ax2)

                ax3 = fig.add_subplot(grid[2, :2])
                ax3.set_title('Log Normal')
                sns.distplot(serie, kde=False, fit=stats.lognorm, ax=ax3)

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


def visualize_feature_info(serie):
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
            sns.distplot(serie, norm_hist=True, ax=ax1)

            # customizing the QQ_plot
            ax2 = fig.add_subplot(grid[1, :2])
            # Set the title
            ax2.set_title('QQ_plot')
            # Plotting the QQ_Plot
            stats.probplot(serie, plot=ax2)

            # Customizing the Box Plot
            ax3 = fig.add_subplot(grid[:, 2])
            # Set title
            ax3.set_title('Box Plot')
            # Plotting the box plot
            sns.boxplot(serie, orient='v', ax=ax3)

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


def show_feature_correlation(df, feature, method='pearson'):
    num_df = df.select_dtypes(exclude=['object'])
    corr = num_df.corr(method=method)

    info = corr[[feature]].sort_values([feature], ascending=False)

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole

            width = 5
            height = len(num_df.columns) // 3

            plt.figure(figsize=(width, height))
            plt.title("Correlations", fontsize=15)
            sns.heatmap(
                corr[[feature]].sort_values(by=[feature], ascending=False),
                annot_kws={"size": 14}, vmin=-1, cmap='PiYG', annot=True)
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

    return info


def visualize_correlation_heatmap(df, method='pearson', mask_threshold=-1):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole

            num_df = df.select_dtypes(exclude=['object'])

            width = len(num_df.columns) // 0.7
            height = len(num_df.columns) // 1.5

            plt.figure(1)
            plt.subplots(figsize=(width, height))
            corr = num_df.corr(method=method)

            # colormap = sns.diverging_palette(220, 10, as_cmap=True)
            colormap = 'PiYG'
            sns.heatmap(corr, xticklabels=True, yticklabels=True, center=0,
                        annot=True, annot_kws={'size': 10}, square=True, linewidths=0.1, linecolor='black',
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


def visualize_boxplot_Grid(df):
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            numerical_cols = X.select_dtypes(include=np.number).columns
            num_df = df[numerical_cols]

            width = 20
            height = len(numerical_cols) // 1.1

            # Visualization with boxplots:
            fig, ax = plt.subplots(figsize=(width, height))

            plot_rows = math.ceil(len(numerical_cols) / 5)

            index = 1
            for col in num_df:
                plt.subplot(plot_rows, 5, index)
                sns.boxplot(y=col, data=num_df.dropna())
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


def show_missing_values_info(df):
    """This function takes a DataFrame(df) as input and returns two columns,
        total missing values and total missing values percentage"""

    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100, 2)[
              round(df.isnull().sum().sort_values(ascending=False)/len(df)*100, 2) != 0
              ]

    info = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    if len(info) > 0:
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

    else:
        print('No missing values found on this dataset')


def show_top_multicollinearity(df, threshold=0.9, method='pearson', features=[], n=50):

    numerical_cols = df.select_dtypes(include=np.number).columns

    # Get X correlations
    df_corr = df[numerical_cols].corr().abs()
    X_corr = (df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
              .stack().sort_values(ascending=False))

    X_corr = pd.DataFrame({'feat_1': X_corr.index.get_level_values(0),
                           'feat_2': X_corr.index.get_level_values(1),
                           'corr': X_corr.values
                           })

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            display(X_corr.head(n))

        else:
            # Other type (?)
            print(X_corr.head(n))

    except NameError:
        # Probably standard Python interpreter
        print(X_corr.head(n))

    return X_corr


def show_feature_importance(
    X, y, model=RandomForestClassifier(random_state=42, n_jobs=4),
    save_as=None
):
    try:
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        feature_importances = feature_importances.reset_index(drop=True)
    except:  # noqa
        model.fit(X, y)

        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        feature_importances = feature_importances.reset_index(drop=True)

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole

            # display(feature_importances)

            width = 18
            height = len(X.columns) // 2

            plt.figure(figsize=(width, height))
            sns.barplot(x="Importance", y="Feature",
                        data=feature_importances.sort_values(by="Importance", ascending=False))
            plt.title('Feature Impportances')
            plt.tight_layout()
            plt.show()

            if save_as is not None:
                plt.savefig(save_as)

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print(feature_importances)
        else:
            # Other type (?)
            print(feature_importances)

    except NameError:
        # Probably standard Python interpreter
        print(feature_importances)


def show_permutation_feature_importance(
    X, y, model=RandomForestClassifier(random_state=42, n_jobs=4),
    n_jobs=1, save_as=None
):
    try:
        result = permutation_importance(model, X, y, n_repeats=10,
                                        random_state=42, n_jobs=n_jobs)
    except:  # noqa
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=10,
                                        random_state=42, n_jobs=n_jobs)

    sorted_idx = result.importances_mean.argsort()

    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            width = 10
            height = len(X.columns) // 2.5

            fig, ax = plt.subplots(figsize=(width, height))
            ax.boxplot(result.importances[sorted_idx].T,
                       vert=False, labels=X.columns[sorted_idx])
            ax.set_title("Permutation Importances")
            fig.tight_layout()
            plt.show()

            if save_as is not None:
                plt.savefig(save_as)

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            print(result)
        else:
            # Other type (?)
            print(result)

    except NameError:
        # Probably standard Python interpreter
        print(result)


def kbest_analysis(X, y, score_func=f_classif):
    skbest = SelectKBest(score_func, k="all").fit(X, y)

    info = pd.DataFrame(np.array([X.columns.values, skbest.scores_, skbest.pvalues_]).transpose(),
                        columns=['Feature', 'score', 'pvalue'])
    return info

#  endregion


#######################################################################################################################
#  region ----------------------------------- Data preparation and wragling -------------------------------------------

class Apply_transformer_to_features(BaseEstimator, TransformerMixin):
    def __init__(self, class_instance=None, features=None, warnings=True):
        self.class_instance = class_instance
        self.features = features
        self.warnings = warnings

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_copy = X.copy()

        # Transform
        X_copy[self.features] = self.class_instance.transform(X_copy[self.features])

        return X_copy

    def fit(self, X, y=None):
        # Select features to imput
        if isinstance(self.features, str):
            if self.features == 'Numerical':
                self.features = X.select_dtypes(include=np.number).columns
            elif self.features == 'Categorical':
                self.features = X.select_dtypes(include="object").columns
            else:
                sys.exit("Error: Feature type not valid.")

        elif isinstance(self.features, list):
            for feat in self.features:
                sys.exit("Error: Feature '" + str(feat) + "' not found on dataframe.") \
                    if feat not in X.columns else None  # Continuation of before line

        elif self.features is None:
            print("Warning: Since no features were passed all will be selected. \
                  ") if self.warnings else None
            self.features = X.columns
        # _________________________________________________________________________________________________

        # Fit
        self.class_instance.fit(X[self.features])

        return self


class Apply_Imputer(BaseEstimator, TransformerMixin):

    '''
        SimpleImputer:
        Params = {
            "strategy": 'mean' | 'median' | 'most_frequent' | 'constant',
            "fill_value": 'missing' | 0
        }
    '''

    def __init__(self, features=None, tecnique='SimpleImputer', params={}, warnings=True):
        self.features = features
        self.tecnique = tecnique
        self.params = params
        self.warnings = warnings

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_copy = X.copy()

        # Transform
        X_copy[self.features] = self.imputer.transform(X_copy[self.features])

        # Make sure that dtypes are still the original ones
        for feat in X_copy.columns:
            if X_copy[feat].dtype.name != X[feat].dtype.name:
                X_copy[feat] = X_copy[feat].astype(X[feat].dtype.name)

        return X_copy

    def fit(self, X, y=None):
        # Select features to imput
        if isinstance(self.features, str):
            if self.features == 'Numerical':
                self.features = X.select_dtypes(include=np.number).columns
            elif self.features == 'Categorical':
                self.features = X.select_dtypes(include="object").columns
            else:
                sys.exit("Error: Feature type not valid.")

        elif isinstance(self.features, list):
            for feat in self.features:
                if feat not in X.columns:
                    sys.exit("Error: Feature '" + str(feat) + "' not found on dataframe.")

        elif self.features is None:
            print("Warning: Since no features were passed all will be selected for imputing. \
                  ") if self.warnings else None
            self.features = X.columns
        # _________________________________________________________________________________________________

        # Fit imputer
        if self.tecnique == 'SimpleImputer':
            self.imputer = SimpleImputer(**self.params)
            self.imputer.fit(X[self.features])

        elif self.tecnique == 'IterativeImputer':
            self.imputer = IterativeImputer(**self.params)
            self.imputer.fit(X[self.features])

        else:
            sys.exit("Error: Tecnique not supported. No valid imputer reference was created.")
        # _________________________________________________________________________________________________

        return self


class Apply_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, tecnique='OrdinalEncoder'):
        self.tecnique = tecnique

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_copy = X.copy()

        # Encode categorical data:
        if len(self.cat_features) > 0:
            if self.tecnique == 'get_dummies':
                encoded_cat_X = pd.get_dummies(X_copy[self.cat_features])

                # Drop old columns and append new ones
                X_copy = X_copy.drop(self.cat_features, axis=1)
                X_copy[encoded_cat_X.columns] = encoded_cat_X

            elif self.tecnique == 'OneHotEncoder':
                new_feature_names = self.encoder.get_feature_names(self.cat_features)
                encoded_cat_X = pd.DataFrame(data=self.encoder.transform(X_copy[self.cat_features]),
                                             index=X_copy.index, columns=new_feature_names)

                # Drop old columns and append new ones
                X_copy = X_copy.drop(self.cat_features, axis=1)
                X_copy[new_feature_names] = encoded_cat_X

            elif self.tecnique == 'OrdinalEncoder':
                X_copy[self.cat_features] = self.encoder.transform(X_copy[self.cat_features].copy())
            else:
                sys.exit("Error: Need valid encoder. Valid types are 'dummie_OH', 'OH' and 'Label'.")

        return X_copy

    def fit(self, X, y=None):
        # Get numerical features
        self.num_features = X.select_dtypes(include=np.number).columns

        # Get categorical features
        self.cat_features = X.select_dtypes(include="object").columns

        # Fit encoders:
        if len(self.cat_features) > 0:
            if self.tecnique == 'get_dummies':
                None  # No encoder to fit

            elif self.tecnique == 'OneHotEncoder':
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                self.encoder.fit(X[self.cat_features])

            elif self.tecnique == 'OrdinalEncoder':
                self.encoder = OrdinalEncoder()
                self.encoder.fit(X[self.cat_features])

            else:
                sys.exit("Error: Need valid encoder. Valid types are 'get_dummies', 'OH' and 'Ordinal'.")
        else:
            print("Warning: No categorical features have been found. An unchanged dataframe will be returned.")

        return self


class Apply_Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, tecnique='MinMaxScaler', features=[]):
        self.tecnique = tecnique
        self.features = features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_copy = X.copy()

        # Apply the Scaler to numerical columns
        X_copy[self.features] = self.scaler.transform(X_copy[self.features])

        return X_copy

    def fit(self, X, y=None):
        # Select numerical features if no features were selected
        if len(self.features) == 0:
            self.features = X.select_dtypes(include=np.number).columns

        # Initialize Scaler
        if self.tecnique == 'StandardScaler':
            self.scaler = StandardScaler(copy=True)

        elif self.tecnique == 'RobustScaler':
            self.scaler = RobustScaler()

        elif self.tecnique == 'MinMaxScaler':
            self.scaler = MinMaxScaler()

        elif self.tecnique == 'MaxAbsScaler':
            self.scaler = MaxAbsScaler()

        else:
            sys.exit('''Error: Need valid Scaler.
                     \nValid types are 'StandardScaler', 'RobustScaler',
                     'MinMaxScaler' and 'MaxAbsScaler'.''')

        # Fit Scaler
        self.scaler.fit(X[self.features])

        return self


class Apply_PowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tecnique='yeo-johnson', features=[], skew_threshold=0.0):
        self.tecnique = tecnique
        self.features = features
        self.skew_threshold = skew_threshold

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_copy = X.copy()

        # Apply the transformer to columns
        if self.tecnique == 'log1p':
            X_copy[self.features] = np.log1p(X_copy[self.features])
        elif self.tecnique == 'sqrt':
            X_copy[self.features] = np.sqrt(X_copy[self.features])
        else:
            X_copy[self.features] = self.transformer.transform(X_copy[self.features])

        return X_copy

    def fit(self, X, y=None):
        # Select numerical features if no features were selected
        if len(self.features) == 0:
            self.features = X.select_dtypes(include=np.number).columns

        if self.skew_threshold > 0:
            # Identify columns to which transformer will be applied
            skewColumns = X[self.features].apply(lambda x: skew(x)).sort_values(ascending=False)
            skewed_cols_info = skewColumns[abs(skewColumns) > self.skew_threshold]
            self.features = list(skewed_cols_info.index)

        # Initialize transformer
        if self.tecnique == 'log1p':
            pass
        elif self.tecnique == 'sqrt':
            pass
        elif self.tecnique == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson')

        elif self.tecnique == 'box-cox':
            self.transformer = PowerTransformer(method='box-cox')

        elif self.tecnique == 'QuantileTansformer':
            self.transformer = QuantileTransformer(random_state=42)

        else:
            sys.exit('''Error: Need valid transformer.
                     \nValid types are 'log1p', 'sqrt',
                     'yeo-johnson', 'box-cox', 'QuantileTansformer'.''')

        # Fit transformer
        self.transformer.fit(X[self.features])

        return self


class Create_Categorical_Unique_Counts(BaseEstimator, TransformerMixin):
    '''
        Creates new features with unique count for selected features.
        If no features are given then all categorical features are selected automatically.
    '''

    def __init__(self, features=[]):
        self.features = features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X, y=None):
        X_copy = X.copy()

        for column in self.features:
            # Create unique counts
            counts_df = X_copy[column].value_counts().to_frame()

            # Merge
            X_copy = pd.merge(X_copy, counts_df, how='left', left_on=column, right_index=True,
                              suffixes=('', '_unique_count'))

        return X_copy

    def fit(self, X, y=None):
        # Identify categorical columns if no columns were passed
        if len(self.features) == 0:
            self.features = X.select_dtypes(include="object").columns

        return self


def Create_PolynomialFeatures(X, degree=2):
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)

    return pd.DataFrame(data=poly.transform(X), columns=poly.get_feature_names())


def Fix_multicollinearity(X, y, threshold=0.95, method='pearson', features=[], verbose=True):

    if len(features) == 0:
        # Identify numerical features
        features = X.select_dtypes(include=np.number).columns

    # Get y correlations
    y_corr = abs(X[features].corrwith(y, method=method))
    y_corr = pd.DataFrame({'feat': y_corr.index.values, 'corr': y_corr.values})

    # Get X correlations
    X_corr = X.corr().abs()
    X_corr = (X_corr.where(np.triu(np.ones(X_corr.shape), k=1).astype(np.bool))
              .stack().sort_values(ascending=False))

    X_corr = pd.DataFrame({'feat_1': X_corr.index.get_level_values(0),
                           'feat_2': X_corr.index.get_level_values(1),
                           'corr': X_corr.values
                           })

    # Select features to drop
    drop_features = []
    for index, row in X_corr.iterrows():
        if row['corr'] > threshold:
            # Get correlation of features with y
            feat_1_y_corr = y_corr['corr'].loc[y_corr['feat'] == row['feat_1']].values
            feat_2_y_corr = y_corr['corr'].loc[y_corr['feat'] == row['feat_2']].values

            # Compare to drop the feature with less correlation with y
            if feat_1_y_corr >= feat_2_y_corr:
                drop_features.append(row['feat_2'])

            elif feat_1_y_corr < feat_2_y_corr:
                drop_features.append(row['feat_1'])
        else:
            break  # No need to continue since X_corr is sorted

    if verbose:
        if len(drop_features) > 0:
            print('Dropped features: ' + str(drop_features))
        else:
            print('No features were dropped.')

    return X.drop(drop_features, axis=1)

# endregion


#  region -------------------------------------- Pycaret adapted classes ----------------------------------------------

class Replace_New_Catagorical_Levels(BaseEstimator, TransformerMixin):
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

    def __init__(self, target=None, replacement_strategy='most frequent'):
        self.target = target
        self.replacement_strategy = replacement_strategy

    def fit(self, data, y=None):
        # need to make a place holder that keep records of all the levels,
        # and in case a new level appears in test we will change it to others

        if self.target is None:
            self.ph_train_level = pd.DataFrame(
                columns=data.select_dtypes(include="object").columns
                )
        else:
            self.ph_train_level = pd.DataFrame(
                columns=data.drop(self.target, axis=1).select_dtypes(include="object").columns
                )

        for i in self.ph_train_level.columns:
            if self.replacement_strategy == "least frequent":
                self.ph_train_level.loc[0, i] = list(data[i].value_counts().sort_values().index)
            else:
                self.ph_train_level.loc[0, i] = list(data[i].value_counts().index)

    def transform(self, data, y=None):
        # we need to learn the same for test data , and then we will compare to check what levels are new in there
        if self.target is None:
            self.ph_test_level = pd.DataFrame(
                columns=data.select_dtypes(include="object").columns
                )
        else:
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

def ImportanceBased_feature_selection(X, y, model=RandomForestRegressor(random_state=42, n_jobs=4),
                                      n_components=50, verbose=True):

    model.fit(X, y)

    # Get feature importances and sort features by it
    feature_importances_pd = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances_pd = feature_importances_pd.sort_values('importance', ascending=False)
    feature_importances_pd = feature_importances_pd.reset_index(drop=True)

    # Get top n_components features and parse them as a list
    feature_importances_pd = feature_importances_pd.head(n_components)

    if verbose:
        print("Top " + str(n_components) + " selected features by model:")
        try:
            shell = get_ipython().__class__.__name__  # noqa
            if shell == 'ZMQInteractiveShell':
                # Jupyter notebook or qtconsole
                display(feature_importances_pd)

        except NameError:
            # Probably standard Python interpreter
            None

    features_names_by_importance = list(feature_importances_pd['feature'])

    return X[features_names_by_importance]


def kbest_feature_selection(X, y, score_func=f_classif, threshold=0.05):
    skbest = SelectKBest(f_classif, k="all").fit(X, y)

    info = pd.DataFrame(np.array([X.columns.values, skbest.scores_, skbest.pvalues_]).transpose(),
                        columns=['Feature', 'score', 'pvalue'])

    selected_features = (info['pvalue' <= threshold])['Feature']
    return selected_features


def PCA_feature_selection(X, n_components=None, normalize=False, verbose=True):
    '''
        n_components: can be a number to choose the number of features to keep,
        or a percentaje to keep a percentaje of features (ej: n_components=0.95).

        returns array with the features chosen by PCA
    '''

    if normalize:
        X = Apply_Scaler(tecnique='StandardScaler').fit_transform(X, y=None)

    PCA_model = PCA(n_components=n_components, random_state=42)
    PCA_model.fit(X)

    # Info
    n_pcs = PCA_model.components_.shape[0]

    if verbose:
        print("Total features selected by PCA=", n_pcs, "/", X.shape[1])
        print("Noise variance =", PCA_model.noise_variance_)
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

        except NameError:
            # Probably standard Python interpreter
            None

    # Get best features
    initial_feature_names = X.columns
    most_important_features = [np.abs(PCA_model.components_[i]).argmax() for i in range(n_pcs)]
    most_important_feature_names = [initial_feature_names[most_important_features[i]] for i in range(n_pcs)]
    most_important_feature_names = list(dict.fromkeys(most_important_feature_names))

    print("PCA selected features:", most_important_feature_names) if verbose else None

    # return most_important_feature_names
    # return X[most_important_feature_names]
    return PCA_model.transform(X)

#  endregion


# region ------------------------------------------------ Utils --------------------------------------------------------

class Remove_features_with_mostly_one_value(BaseEstimator, TransformerMixin):
    '''
        Features that are above threshold in count of just one value will be dropped.
    '''

    def __init__(self, threshold=0.96, verbose=False):
        self.threshold = threshold
        self.verbose = verbose

    def transform(self, X):
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


def Get_columns_names_by_unique_counts(X, numinque_threshold=50, sign='<'):
    ''' Valid signs are ('>' and '<') '''

    X_columns = X.columns

    valid_columns = []

    if sign == '<':
        for col in X_columns:
            if X[col].numinque() < numinque_threshold:
                valid_columns.append(col)

    elif sign == '>':
        for col in X_columns:
            if X[col].numinque() > numinque_threshold:
                valid_columns.append(col)

    else:
        print('Please provide valid sign')

    return valid_columns


def Evaluate_model(X, y, model=None, pipeline=None, score_method='accuracy',
                   splits=KFold(n_splits=5, shuffle=True, random_state=42),
                   n_jobs=4):

    if model is not None and pipeline is None:
        testing_model = model

    elif model is None and pipeline is not None:
        testing_model = clone(pipeline)

    elif model is not None and pipeline is not None:
        testing_model = clone(pipeline)
        testing_model.steps.append(('Model', clone(model)))

    else:
        sys.exit("Error: Needs at least model or pipeline with model in it.")

    scores = cross_val_score(testing_model, X, y, cv=splits, scoring=score_method, n_jobs=n_jobs)

    print("Scores:\n", scores)
    print("Average " + str(score_method) + " score (across experiments):" + str(scores.mean()))

# endregion


#######################################################################################################################
#  region ---------------------------------------------- Other --------------------------------------------------------

def FuzzyWuzzy_matching(left_table, right_table, left_on, right_on, threshold=75):

    # Parse searched data as string to avoid errors
    left_table[left_on] = left_table[left_on].astype(str)
    right_table[right_on] = right_table[right_on].astype(str)

    # Extract arrays from tables
    left_array = left_table[left_on]
    right_array = right_table[right_on]

    best_matchs_list = []
    match_scores = []

    for value in left_array:
        matchs = process.extract(value, right_array, limit=1, scorer=fuzz.token_set_ratio)

        print("Best match scored " + str(matchs[0][1]) + " for (" + str(value)
              + ") == (" + str(matchs[0][0]) + ")")

        if matchs[0][1] > threshold:
            best_matchs_list.append(matchs[0][0])
        else:
            best_matchs_list.append('NaN')
        match_scores.append(matchs[0][1])

    matchs = pd.DataFrame({'original_values': left_array,
                           'best_match': best_matchs_list,
                           'match_scores': match_scores})

    matched_table = pd.merge(left_table, matchs, how='left', left_on=left_on, right_on='original_values')
    matched_table = matched_table.drop_duplicates(subset=left_table.columns, keep='first')

    matched_table = pd.merge(matched_table, right_table, how='left', left_on='best_match', right_on=right_on)
    matched_table = matched_table.drop_duplicates(subset=None, keep='first')

    matched_table = matched_table.drop(['best_match', 'original_values'], axis=1)

    return matched_table

#  endregion
