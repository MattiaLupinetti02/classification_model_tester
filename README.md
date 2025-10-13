```mermaid

classDiagram
    %% =========================
    %% CLASSI PRINCIPALI
    %% =========================
    
    class ModelTester {
        +Dict modelList
        +Dict metrics
        +pd.DataFrame original_data
        +DataHandler data_handler
        +Visualizer visualizer
        +Dict performances
        +List ensambleModelList
        +Dict ensamble_hyperpar
        +str target
        +Dict resampled_data
        +__init__(modelList, metrics, data, target, ensambleModelList=None, resamplingMethods=None)
        +initialize_performance(resampling_methods=None)
        +get_soft_voting_ready_models(models)
        +scale_data(to_scale)
        +make_dataframe_performances()
        +best_param_calculator_by_label(cv=10, avg='macro', searcher_class=GridSearchCV)
        +best_param_calculator_ensamble_by_label(cv=10, avg='macro', searcher_class=GridSearchCV)
        +best_param_calculator_ensamble(avg='macro', cv=10, searcher_class=GridSearchCV)
        +best_param_calculator_ensamble_from_augmented_data(...)
        +best_param_calculator_from_augmented_data(...)
        +best_param_calculator_ensamble_from_augmented_data_by_label(...)
        +best_param_calculator_from_augmented_data_by_label(...)
    }

    class CustomBestParamCalculator {
        +Dict models
        +Dict metrics
        +Dict label_mapping
        +int cv
        +searcher_class
        +StandardScaler scaler
        +__init__(models, metrics, label_mapping, cv=5, searcher_class=None)
        +make_metrics_by_labels(avg='binary')
        +make_metrics(avg='binary')
        +best_param_calculator(X, y, avg='binary', by_target_label=None, searcher_class=GridSearchCV, searcher_kwargs=None)
        +scale_data(X)
    }

    class Visualizer {
        +pd.DataFrame original_data
        +str target
        +numeric_features
        +categorical_features
        +boolean_columns
        +Dict resampled_data
        +__init__(data, target, numeric_features, categorical_features, boolean_features, resampled_data=None)
        +plot_pie_chart(y, target, save=False)
        +plot_boxplots(numeric_features, save=False)
        +plot_pie_resampled(save=False)
        +plot_numeric_distribution(save=False)
        +plot_numeric_distribution_resampled(save=False)
        +plot_boxplots_resampled(save=False)
        +plot_correlation_matrix(save=False)
        +plot_correlation_matrix_resampled(save=False)
        +plot_binary_distribution(save=False)
        +plot_binary_distribution_resampled(save=False)
        +plot_learning_curve(model, X, y, cv=5, n_jobs=-1, name_method=None)
        +plot_validation_curve(model, X, y, param_name, param_range, cv=5, ax=None)
        +learning_curves(augmented_data_X=None, augmented_data_y=None, name_method=None)
        +validation_curves(cv=5, augmented_data_X=None, augmented_data_y=None, resampling_method_name=None)
        +validation_curves_on_augmented_data()
    }

    %% =========================
    %% RELAZIONI TRA CLASSI
    %% =========================
    ModelTester --> CustomBestParamCalculator : usa per ottimizzare iperparametri
    ModelTester --> Visualizer : genera grafici
    ModelTester --> DataHandler : gestisce encoding e resampling
    Visualizer --> pandas.DataFrame : usa per visualizzazioni
    CustomBestParamCalculator --> sklearn.model_selection.GridSearchCV : implementa ricerca iperparametri
    CustomBestParamCalculator --> sklearn.preprocessing.StandardScaler : normalizza dati

