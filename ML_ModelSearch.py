#  region --------------------------------------------- Imports --------------------------------------------------------
# Optuna
import optuna

# Machine learning regresors and classifiers
import xgboost as xgboost
import lightgbm as lightgbm
import catboost as catboost

# Gridsearch CV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold

# Data
# import pandas as pd
import numpy as np

# Others
import sys
import joblib

#  endregion


#######################################################################################################################
#  region ----------------------------- Cross validated search of best model ------------------------------------------

def Model_Search_LightGBM_cv(X, y, model='binary', folds=5,
                             sklearn_metric=None, lightgbm_metric=None,
                             step_wise_start_at=0, final_learning_rate=0.01,
                             use_optuna=False, n_trials=50, load_study_from=None, save_study_as=None,
                             n_jobs=4):

    # If model type is specified by string
    if isinstance(model, str):
        if model == 'binary':
            model = lightgbm.LGBMClassifier(objective='binary', metric='binary_logloss',
                                            random_state=42, feature_fraction_seed=42,
                                            n_jobs=n_jobs)
        elif model == 'multiclass':
            n_classes = len(np.unique(y))
            model = lightgbm.LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=n_classes,
                                            random_state=42, feature_fraction_seed=42,
                                            n_jobs=n_jobs)
        elif model == 'regression':
            model = lightgbm.LGBMRegressor(objective='regression', metric='rmse',
                                           random_state=42, feature_fraction_seed=42,
                                           n_jobs=n_jobs)
        elif model == 'ranking':
            model = lightgbm.LGBMRanker(objective='lambdarank', metric='average_precision',
                                        random_state=42, feature_fraction_seed=42,
                                        n_jobs=n_jobs)
        else:
            sys.exit('Error: Unkown model type.')

    # Score Metrics
    if sklearn_metric is None:  # https://scikit-learn.org/stable/modules/model_evaluation.html
        if isinstance(model, lightgbm.LGBMClassifier):
            sklearn_metric = 'neg_log_loss'
        elif isinstance(model, lightgbm.LGBMRegressor):
            sklearn_metric = 'neg_root_mean_squared_error'
        elif isinstance(model, lightgbm.LGBMRanker):
            sklearn_metric = 'average_precision_score'
        else:
            sys.exit('Error: Sklearn score metric needs to be provided.')

    if lightgbm_metric is not None:  # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        model.set_params(metric=lightgbm_metric)

    # Folds
    if isinstance(folds, int):
        if isinstance(model, lightgbm.LGBMClassifier):
            folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            folds = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Set fixed params
    fixed_params = {
        "verbosity": -1,

        "feature_pre_filter": False,

        "n_estimators": 10000,
        "first_metric_only": True,
    }
    model.set_params(**fixed_params)

    # Create dataset for .cv
    d_train = lightgbm.Dataset(X, label=y)

    if use_optuna:
        print("Searching for a Lightgbm model with optuna \n")

        params = model.get_params()

        if load_study_from is not None:
            study = joblib.load(load_study_from)
        else:
            study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())

        def objetive(trial):

            params.update({
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.008, 0.1),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                })

            cv_results = lightgbm.cv(params, d_train, num_boost_round=10000, early_stopping_rounds=50,
                                     folds=folds, metrics=None, show_stdv=False, verbose_eval=None)

            rmetric_name = list(cv_results.keys())[0]
            score = cv_results[rmetric_name][-1]  # np.min(cv_results[rmetric_name])

            print("Num_boost_round: " + str(len(cv_results[rmetric_name])))

            if save_study_as is not None:
                joblib.dump(study, save_study_as)

            return score

        study.optimize(objetive, n_trials=n_trials, n_jobs=1)

        print("------------------------------------------------------------------------")
        print("Best parameters found: " + str(study.best_params))
        print("Best score achived: " + str(study.best_value))
        print("------------------------------------------------------------------------")

        model.set_params(**study.best_params)

        # num_boost_round optimization
        cv_results = lightgbm.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                 folds=folds, metrics=None, show_stdv=False, verbose_eval=None)

        rmetric_name = list(cv_results.keys())[0]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(n_estimators=best_boost_round)

    else:
        print("Searching for a Lightgbm model with the step wise method \n")

        if step_wise_start_at <= 0:
            # num_boost_round optimization
            cv_results = lightgbm.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                     folds=folds, metrics=None, show_stdv=False, verbose_eval=None)

            rmetric_name = list(cv_results.keys())[0]
            best_boost_round = len(cv_results[rmetric_name])
            best_score_achived = cv_results[rmetric_name][-1]
            print("------------------------------------------------------------------------")
            print("Best num_boost_round: " + str(best_boost_round))
            print("Best score achived: " + str(best_score_achived))
            print("------------------------------------------------------------------------")
            model.set_params(n_estimators=best_boost_round)

        # Param search
        if step_wise_start_at <= 1:
            param_test = {
                'model__max_depth': range(2, 13, 1)
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 2:
            param_test = {
                'num_leaves': [2, 4, 8, 16, 31, 64, 96, 128, 256, 384, 512, 640, 768, 896, 1024]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 3:
            param_test = {
                'subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 4:
            param_test = {
                'colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 5:
            param_test = {
                'lambda_l1': [0, 0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 40, 70, 100, 200]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 6:
            param_test = {
                'lambda_l2': [0, 0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 40, 70, 100, 200]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        # Get model from pipeline
        model = pipe_model.named_steps['model']

        # Set final learning rate
        model.set_params(learning_rate=final_learning_rate)

        # num_boost_round optimization
        model.set_params(n_estimators=10000)
        cv_results = lightgbm.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                 folds=folds, metrics=None, show_stdv=False, verbose_eval=None)

        rmetric_name = list(cv_results.keys())[0]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(n_estimators=best_boost_round)

    return model


def Model_Search_XGboost_cv(X, y, model='binary', folds=5,
                            sklearn_metric=None, xgboost_metric=None,
                            step_wise_start_at=0, final_learning_rate=0.01,
                            use_optuna=False, n_trials=50, load_study_from=None, save_study_as=None,
                            n_jobs=4):

    # If model type is specified by string
    if isinstance(model, str):
        if model == 'binary':
            model = xgboost.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                          random_state=42, seed=42, feature_fraction_seed=42,
                                          use_label_encoder=False, nthread=n_jobs)
        elif model == 'multiclass':
            n_classes = len(np.unique(y))
            model = xgboost.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', num_class=n_classes,
                                          random_state=42, seed=42, feature_fraction_seed=42,
                                          use_label_encoder=False, nthread=n_jobs)
        elif model == 'regression':
            model = xgboost.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                         random_state=42, seed=42, feature_fraction_seed=42,
                                         use_label_encoder=False, nthread=n_jobs)
        elif model == 'ranking':
            model = xgboost.XGBRanker(objective='rank:map', eval_metric='map',
                                      random_state=42, seed=42, feature_fraction_seed=42,
                                      use_label_encoder=False, nthread=n_jobs)
        else:
            sys.exit('Error: Unkown model type.')

    # Score Metrics
    if sklearn_metric is None:  # https://scikit-learn.org/stable/modules/model_evaluation.html
        if isinstance(model, xgboost.XGBClassifier):
            sklearn_metric = 'neg_log_loss'
        elif isinstance(model, xgboost.XGBRegressor):
            sklearn_metric = 'neg_root_mean_squared_error'
        elif isinstance(model, xgboost.XGBRanker):
            sklearn_metric = 'average_precision_score'
        else:
            sys.exit('Error: Sklearn score metric needs to be provided.')

    if xgboost_metric is None:  # https://xgboost.readthedocs.io/en/latest/parameter.html
        if isinstance(model, xgboost.XGBClassifier):
            n_classes = len(np.unique(y))
            if n_classes > 2:
                xgboost_metric = 'mlogloss'
            else:
                xgboost_metric = 'logloss'
        elif isinstance(model, xgboost.XGBRegressor):
            xgboost_metric = 'rmse'
        elif isinstance(model, xgboost.XGBRanker):
            xgboost_metric = 'map'
        else:
            sys.exit('Error: Xgboost score metric needs to be provided.')

    # Folds
    if isinstance(folds, int):
        if isinstance(model, xgboost.XGBClassifier):
            folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            folds = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Set fixed params
    fixed_params = {
        'verbosity': 0,
        'silent': 1,

        # 'num_iterations': 10000,
        'n_estimators': 10000,
    }
    model.set_params(**fixed_params)

    # dataset for .cv
    d_train = xgboost.DMatrix(X, label=y)

    if use_optuna:
        print("Searching for a Xgboost model with optuna \n")

        params = model.get_params()

        if load_study_from is not None:
            study = joblib.load(load_study_from)
        else:
            study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())

        def objetive(trial):

            params.update({
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.008, 0.1),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 500),
                "gamma": trial.suggest_loguniform("gamma", 1e-4, 1e4),
                "subsample": trial.suggest_loguniform("subsample", 0.4, 1),
                "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.4, 1),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e4),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1e4),
                })

            cv_results = xgboost.cv(params, d_train, num_boost_round=10000, early_stopping_rounds=50,
                                    folds=folds, metrics=xgboost_metric, show_stdv=False, verbose_eval=None,
                                    as_pandas=False)

            rmetric_name = list(cv_results.keys())[2]
            score = cv_results[rmetric_name][-1]  # np.min(cv_results[rmetric_name])

            print("Num_boost_round: " + str(len(cv_results[rmetric_name])))

            if save_study_as is not None:
                joblib.dump(study, save_study_as)

            return score

        study.optimize(objetive, n_trials=n_trials, n_jobs=1)

        print("------------------------------------------------------------------------")
        print("Best parameters found: " + str(study.best_params))
        print("Best score achived: " + str(study.best_value))
        print("------------------------------------------------------------------------")

        model.set_params(**study.best_params)

        # num_boost_round optimization
        cv_results = xgboost.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                folds=folds, metrics=xgboost_metric, show_stdv=False, verbose_eval=None,
                                as_pandas=False)

        rmetric_name = list(cv_results.keys())[2]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(n_estimators=best_boost_round)

    else:
        print("Searching for a Xgboost model with the step wise method \n")

        if step_wise_start_at <= 0:
            # num_boost_round optimization
            cv_results = xgboost.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                    folds=folds, metrics=xgboost_metric, show_stdv=False, verbose_eval=None,
                                    as_pandas=False)

            rmetric_name = list(cv_results.keys())[2]
            best_boost_round = len(cv_results[rmetric_name])
            best_score_achived = cv_results[rmetric_name][-1]
            print("------------------------------------------------------------------------")
            print("Best num_boost_round: " + str(best_boost_round))
            print("Best score achived: " + str(best_score_achived))
            print("------------------------------------------------------------------------")
            model.set_params(n_estimators=best_boost_round)

        # Param search
        if step_wise_start_at <= 1:
            param_test = {
                'max_depth': range(2, 11, 1)
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 2:
            param_test = {
                'min_child_weight': [0, 1, 2, 3, 5, 7, 10, 12, 15, 25, 50, 75, 100]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 3:
            param_test = {
                'gamma': [0, 0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 40, 70, 100]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 4:
            param_test = {
                'subsample': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 5:
            param_test = {
                'colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 6:
            param_test = {
                'alpha': [0, 0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 40, 70, 100, 200]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 7:
            param_test = {
                'lambda': [0, 0.0001, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 40, 70, 100, 200]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=1, cv=folds, verbose=True)
            search.fit(X, y)
            print("Best parameters found: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        # Get model from pipeline
        model = pipe_model.named_steps['model']

        # Set final learning rate
        model.set_params(learning_rate=final_learning_rate)

        # num_boost_round optimization
        model.set_params(num_iterations=10000)
        cv_results = xgboost.cv(model.get_params(), d_train, num_boost_round=10000, early_stopping_rounds=50,
                                folds=folds, metrics=xgboost_metric, show_stdv=False, verbose_eval=None,
                                as_pandas=False)

        rmetric_name = list(cv_results.keys())[2]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("------------------------------------------------------------------------")
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(n_estimators=best_boost_round)

    return model


def Model_Search_Catboost_cv(X, y, model='binary', cat_features=None, folds=5,
                             sklearn_metric=None, catboost_metric=None,
                             step_wise_start_at=0, final_learning_rate=0.01,
                             use_optuna=False, n_trials=20, load_study_from=None, save_study_as=None,
                             n_jobs=4):

    # If model type is specified by string
    if isinstance(model, str):
        if model == 'binary':
            model = catboost.CatBoostClassifier(loss_function='Logloss', thread_count=n_jobs)
        elif model == 'multiclass':
            model = catboost.CatBoostClassifier(loss_function='MultiClass', thread_count=n_jobs)
        elif model == 'regression':
            model = catboost.CatBoostRegressor(loss_function='RMSE', thread_count=n_jobs)
        else:
            sys.exit('Error: Unkown model type.')

    # Score Metrics
    if sklearn_metric is None:  # https://scikit-learn.org/stable/modules/model_evaluation.html
        if isinstance(model, catboost.CatBoostClassifier):
            sklearn_metric = 'neg_log_loss'
        elif isinstance(model, catboost.CatBoostRegressor):
            sklearn_metric = 'neg_root_mean_squared_error'
        else:
            sys.exit('Error: Sklearn score metric needs to be provided.')

    if catboost_metric is not None:  # https://catboost.ai/docs/concepts/loss-functions.html
        model.set_params(loss_function=catboost_metric)

    # Folds
    if isinstance(folds, int):
        if isinstance(model, lightgbm.LGBMClassifier):
            folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            folds = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Set fixed params
    fixed_params = {
        "verbose": False,

        "random_state": 42,

        "iterations": 10000,
    }
    model.set_params(**fixed_params)

    # Set dataset for .cv
    d_train = catboost.Pool(X, label=y, cat_features=cat_features)

    if use_optuna:
        print("Searching for a Catboost model with optuna \n")

        params = model.get_params()

        if load_study_from is not None:
            study = joblib.load(load_study_from)
        else:
            study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())

        def objetive(trial):

            params.update({
                "boosting_type": trial.suggest_categorical("boosting_type", ['Ordered', 'Plain']),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.008, 0.1),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-4, 100),
                "border_count": trial.suggest_int('border_count', 1, 255),
                "random_strength": trial.suggest_loguniform("random_strength", 1e-4, 1e4),
                "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 1e-4, 1e4),
                })

            cv_results = catboost.cv(params=params, pool=d_train, iterations=10000, early_stopping_rounds=50,
                                     folds=folds, verbose_eval=None, as_pandas=False)

            rmetric_name = list(cv_results.keys())[1]
            score = cv_results[rmetric_name][-1]  # np.min(cv_results[rmetric_name])

            print("Num_boost_round: " + str(len(cv_results[rmetric_name])))

            if save_study_as is not None:
                joblib.dump(study, save_study_as)

            return score

        study.optimize(objetive, n_trials=n_trials, n_jobs=1)

        print("------------------------------------------------------------------------")
        print("Best parameters found: " + str(study.best_params))
        print("Best score achived: " + str(study.best_value))
        print("------------------------------------------------------------------------")

        model.set_params(**study.best_params)

        # num_boost_round optimization
        cv_results = catboost.cv(params=model.get_params(), pool=d_train, iterations=10000, early_stopping_rounds=50,
                                 folds=folds, verbose_eval=None, as_pandas=False)

        rmetric_name = list(cv_results.keys())[1]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(iterations=best_boost_round)

    else:
        print("Searching for a Catboost model with the step wise method \n")

        if step_wise_start_at <= 0:
            # num_boost_round optimization
            cv_results = catboost.cv(params=model.get_params(), pool=d_train, iterations=10000,
                                     early_stopping_rounds=50,
                                     folds=folds, verbose_eval=None, as_pandas=False)

            rmetric_name = list(cv_results.keys())[1]
            best_boost_round = len(cv_results[rmetric_name])
            best_score_achived = cv_results[rmetric_name][-1]
            print("------------------------------------------------------------------------")
            print("Best num_boost_round: " + str(best_boost_round))
            print("Best score achived: " + str(best_score_achived))
            print("------------------------------------------------------------------------")
            model.set_params(iterations=best_boost_round)

        # Param search
        if step_wise_start_at <= 1:
            param_test = {
                'max_depth': range(2, 11, 1)
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 2:
            param_test = {
                'l2_leaf_reg': [0, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 1, 2, 3, 4, 7, 10]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 3:
            param_test = {
                'random_strength': [0, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 4:
            param_test = {
                'bagging_temperature': [0, 0.001, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 4, 7, 10, 30, 60, 90, 120]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 5:
            param_test = {
                'border_count': [32, 5, 10, 20, 50, 100, 150, 200, 255]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        if step_wise_start_at <= 6:
            param_test = {
                'ctr_border_count': [50, 5, 10, 20, 100, 150, 200, 255]
                }
            search = GridSearchCV(estimator=model,
                                  param_grid=param_test, scoring=sklearn_metric,
                                  n_jobs=n_jobs, iid=False, cv=folds, verbose=True)
            search.fit(X, y, cat_features=cat_features)
            print("Best params encountered: " + str(search.best_params_))
            print("Best score achived: " + str(search.best_score_))
            print("------------------------------------------------------------------------")
            pipe_model = search.best_estimator_

        # Get model from pipeline
        model = pipe_model.named_steps['model']

        # Set final learning rate
        model.set_params(learning_rate=final_learning_rate)

        # num_boost_round optimization
        cv_results = catboost.cv(params=model.get_params(), pool=d_train, iterations=10000, early_stopping_rounds=50,
                                 folds=folds, verbose_eval=None, as_pandas=False)

        rmetric_name = list(cv_results.keys())[1]
        best_boost_round = len(cv_results[rmetric_name])
        best_score_achived = cv_results[rmetric_name][-1]
        print("Best num_boost_round: " + str(best_boost_round))
        print("Best score achived: " + str(best_score_achived))
        print("------------------------------------------------------------------------")
        model.set_params(iterations=best_boost_round)

    return model

#  endregion
