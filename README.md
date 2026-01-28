The Classification Model Tester is a comprehensive Python library for end-to-end machine learning model testing, optimization, and visualization. It follows a modular architecture with six core classes that work together to provide a complete pipeline from data preprocessing to model deployment.

```mermaid
graph TB
    subgraph "Classification Model Tester - Architecture Overview"
        
        %% Main Classes
        MT[ModelTester<br/>Main Orchestrator]
        DH[DataHandler<br/>Data Preprocessing]
        CBC[CustomBestParamCalculator<br/>Hyperparameter Optimization]
        VZ[Visualizer<br/>Visualization & Analysis]
        CEC[CustomEnsambleBestParamCalculator<br/>Ensemble Optimization]
        COS[CreateOptunaStudy<br/>Optuna Configuration]
        
        %% Subcomponents
        subgraph "ModelTester Components"
            MT_INIT[Initialization]
            MT_OPT[Optimization Orchestration]
            MT_ENS[Ensemble Management]
            MT_IMP[Model Implementation]
            MT_PERF[Performance Tracking]
        end
        
        subgraph "DataHandler Components"
            DH_ENC[Encoding Engine]
            DH_RES[Resampling Module]
            DH_SPL[Data Splitting]
            DH_MAP[Mapping Management]
        end
        
        subgraph "CustomBestParamCalculator Components"
            CBC_GRID[GridSearch/RandomSearch]
            CBC_OPTUNA[Optuna Integration]
            CBC_SCORER[Metric Scorer Factory]
            CBC_VAL[Cross-Validation]
            CBC_TIMEOUT[Timeout Management]
        end
        
        subgraph "Visualizer Components"
            VZ_PLOTS[Statistical Plots]
            VZ_CURVES[Learning/Validation Curves]
            VZ_CORR[Correlation Analysis]
            VZ_DIST[Distribution Analysis]
        end
        
        subgraph "Ensemble Optimizer Components"
            CEC_VOT[Voting Classifier]
            CEC_STACK[Stacking Classifier]
            CEC_PARSE[DataFrame Parser]
            CEC_FACT[Model Factory]
        end
        
        %% External Dependencies
        subgraph "External Dependencies"
            SKL[scikit-learn]
            PD[pandas]
            NP[numpy]
            OPT[optuna]
            XGB[xgboost]
            IMB[imbalanced-learn]
            MAT[matplotlib]
            SNS[seaborn]
        end
        
        %% Core Relationships
        MT --> DH
        MT --> CBC
        MT --> VZ
        MT --> CEC
        
        DH --> DH_ENC
        DH --> DH_RES
        DH --> DH_SPL
        DH --> DH_MAP
        
        CBC --> CBC_GRID
        CBC --> CBC_OPTUNA
        CBC --> CBC_SCORER
        CBC --> CBC_VAL
        CBC --> CBC_TIMEOUT
        
        CBC_OPTUNA --> COS
        
        CEC --> CEC_VOT
        CEC --> CEC_STACK
        CEC --> CEC_PARSE
        CEC --> CEC_FACT
        
        VZ --> VZ_PLOTS
        VZ --> VZ_CURVES
        VZ --> VZ_CORR
        VZ --> VZ_DIST
        
        %% External Dependencies
        DH --> SKL
        DH --> PD
        DH --> NP
        DH --> IMB
        
        CBC --> SKL
        CBC --> OPT
        
        VZ --> MAT
        VZ --> SNS
        VZ --> PD
        
        CEC --> SKL
        CEC --> XGB
        CEC --> PD
        
        COS --> OPT
    end

graph TD
    subgraph "Complete Architecture & Data Flow"
    
        %% Input Layer
        RAW_DATA[Raw Dataset<br/>CSV/DataFrame]
        MODEL_SPEC[Model Specifications<br/>Dictionary]
        METRIC_SPEC[Metric Specifications<br/>Dictionary]
        
        %% Core Processing Pipeline
        subgraph "Phase 1: Data Preprocessing"
            DH_PRE[DataHandler.init]
            DH_AUTO[Automatic Type Detection]
            DH_OHE[One-Hot Encoding]
            DH_LE[Label Encoding]
            DH_CONCAT[Dataset Reconstruction]
            DH_RESAMP[Optional Resampling]
            DH_SPLIT[Train/Test Split]
        end
        
        subgraph "Phase 2: Hyperparameter Optimization"
            CBC_INIT[CustomBestParamCalculator]
            CBC_STRAT[Strategy Selection<br/>GridSearch/Optuna]
            CBC_OBJ[Objective Function Creation]
            CBC_TRIAL[Trial Execution<br/>with Timeout]
            CBC_CV[Cross-Validation Scoring]
            CBC_BEST[Best Parameters Extraction]
        end
        
        subgraph "Phase 3: Model Implementation"
            MT_IMP_INIT[ModelTester.implement_models]
            MT_MODEL_SET[Model Configuration<br/>with Best Params]
            MT_ENS_SET[Ensemble Configuration]
            MT_CV_SCORE[Final Cross-Validation]
            MT_PERF_STORE[Performance Storage]
        end
        
        subgraph "Phase 4: Ensemble Optimization"
            CEC_INIT[EnsembleOptimizer]
            CEC_BASE[Base Model Creation]
            CEC_ENS_CREATE[Ensemble Creation<br/>Voting/Stacking]
            CEC_GRID[Ensemble Parameter Grid]
            CEC_OPT[Ensemble Optimization]
            CEC_BEST_ENS[Best Ensemble Selection]
        end
        
        subgraph "Phase 5: Visualization & Analysis"
            VZ_INIT[Visualizer]
            VZ_DIST_PLOTS[Distribution Analysis]
            VZ_CORR_PLOTS[Correlation Analysis]
            VZ_LC_PLOTS[Learning Curves]
            VZ_VC_PLOTS[Validation Curves]
            VZ_SAVE[Plot Saving]
        end
        
        %% Output Layer
        PERF_DF[Performance DataFrame<br/>Model/Metric/Hyperparams/Score]
        BEST_MODELS[Optimized Models<br/>Ready for Deployment]
        VISUAL_REPORTS[Visual Reports<br/>PNG/PDF Files]
        ENSEMBLE_MODELS[Optimized Ensemble Models]
        
        %% Data Flow Connections
        RAW_DATA --> DH_PRE
        MODEL_SPEC --> DH_PRE
        METRIC_SPEC --> DH_PRE
        
        DH_PRE --> DH_AUTO
        DH_AUTO --> DH_OHE
        DH_OHE --> DH_LE
        DH_LE --> DH_CONCAT
        DH_CONCAT --> DH_RESAMP
        DH_RESAMP --> DH_SPLIT
        
        DH_SPLIT --> CBC_INIT
        MODEL_SPEC --> CBC_INIT
        METRIC_SPEC --> CBC_INIT
        
        CBC_INIT --> CBC_STRAT
        CBC_STRAT --> CBC_OBJ
        CBC_OBJ --> CBC_TRIAL
        CBC_TRIAL --> CBC_CV
        CBC_CV --> CBC_BEST
        
        CBC_BEST --> MT_IMP_INIT
        DH_SPLIT --> MT_IMP_INIT
        
        MT_IMP_INIT --> MT_MODEL_SET
        MT_MODEL_SET --> MT_ENS_SET
        MT_ENS_SET --> MT_CV_SCORE
        MT_CV_SCORE --> MT_PERF_STORE
        
        MT_PERF_STORE --> CEC_INIT
        CEC_INIT --> CEC_BASE
        CEC_BASE --> CEC_ENS_CREATE
        CEC_ENS_CREATE --> CEC_GRID
        CEC_GRID --> CEC_OPT
        CEC_OPT --> CEC_BEST_ENS
        
        DH_SPLIT --> VZ_INIT
        MT_PERF_STORE --> VZ_INIT
        
        VZ_INIT --> VZ_DIST_PLOTS
        VZ_DIST_PLOTS --> VZ_CORR_PLOTS
        VZ_CORR_PLOTS --> VZ_LC_PLOTS
        VZ_LC_PLOTS --> VZ_VC_PLOTS
        VZ_VC_PLOTS --> VZ_SAVE
        
        MT_PERF_STORE --> PERF_DF
        MT_IMP_INIT --> BEST_MODELS
        VZ_SAVE --> VISUAL_REPORTS
        CEC_BEST_ENS --> ENSEMBLE_MODELS
        
        %% Configuration Components
        subgraph "Configuration Layer"
            COS_CONFIG[create_optuna_study<br/>Study Configuration]
            COS_SAMPLER[Sampler Selection<br/>TPE/Random]
            COS_PRUNER[Pruner Configuration<br/>MedianPruner]
            COS_STORAGE[SQLite Storage Setup]
        end
        
        CBC_STRAT --> COS_CONFIG
        COS_CONFIG --> COS_SAMPLER
        COS_SAMPLER --> COS_PRUNER
        COS_PRUNER --> COS_STORAGE
    end

classDiagram
    class ModelTester {
        -DataHandler data_handler
        -dict performances
        -list ensembleModelList
        -Visualizer visualizer
        -dict modelList
        -dict metrics
        -str target
        +__init__(modelList, metrics, data, target, ensembleModelList, resamplingMethods, n_jobs)
        +best_param_calculator(cv, avg, searcher_class)
        +best_param_calculator_by_label(cv, avg, searcher_class)
        +best_param_calculator_ensemble(cv, avg, searcher_class)
        +best_param_calculator_from_augmented_data(cv, avg, searcher_class)
        +implement_models(cv, avg, metric, ensemble_by_base_models, specific, by_label, resampled_data, base_models, performance_dataset)
        +make_dataframe_performances() pd.DataFrame
        +plot_pie_chart(resampled, save)
        +plot_boxplots(resampled, save)
        +learning_curves(resampled)
        +validation_curves(cv, resampled, save)
    }
    
    class DataHandler {
        -pd.DataFrame original_data
        -str target
        -dict categorical_features
        -dict label_mapping
        -pd.DataFrame encoded_data
        -np.ndarray y_encoded
        -dict resampled_data_dict
        +__init__(data, target)
        +encode_features() pd.DataFrame
        +encode_target() np.ndarray
        +concat_encoded_dataset() pd.DataFrame
        +split_data(X, y, test_size, random_state)
        +dataResampler(methods, X, y, target_name, mapping, boolean_columns, numeric_features) dict
        +get_label_mapping() dict
        +get_encoded_data() tuple
    }
    
    class CustomBestParamCalculator {
        -dict models
        -dict metrics
        -dict label_mapping
        -int cv
        -dict scorers
        -StandardScaler scaler
        -int n_jobs
        +__init__(models, metrics, label_mapping, cv, searcher_class, n_jobs)
        +make_metrics_by_labels(avg) dict
        +make_metrics(avg) dict
        +best_param_calculator(X, y, avg, by_target_label, searcher_class, searcher_kwargs, brute_force) dict
        +create_optuna_objective(model, X, y, scorer, param_grid, timeout) function
        +validation_model_CV(model, X, y, cv, by_label, avg) dict
        +scale_data(X) np.ndarray
    }
    
    class MemoryEfficientGridSampler {
        -dict search_space
        -iterator _param_iter
        +__init__(search_space)
        +reseed_rng()
        +infer_relative_search_space(study, trial) dict
        +sample_relative(study, trial, search_space) dict
        +sample_independent(study, trial, param_name, param_distribution) any
        -_generate_param_combinations() generator
    }
    
    class Visualizer {
        -pd.DataFrame original_data
        -pd.Series y_data
        -str target
        -pd.DataFrame numeric_features
        -dict categorical_features
        -list modelList
        +__init__(data, y_data, target, numeric_features, categorical_features, boolean_features, model_list, resampled_data)
        +plot_pie_chart(y, target, save)
        +plot_boxplots(data, save)
        +plot_numeric_distribution(data, save)
        +plot_correlation_matrix(data, save)
        +plot_correlation_matrix_2(data, save)
        +plot_binary_distribution(data, title, figsize, save)
        +plot_learning_curve(model, X, y, cv, n_jobs, name_method)
        +plot_validation_curve(model, X, y, param_name, param_range, cv, ax, dt_name)
        +get_numeric_columns(df) pd.DataFrame
        +get_categorical_columns(df) pd.DataFrame
        +get_boolean_columns(df) pd.DataFrame
    }
    
    class EnsembleOptimizer {
        -str scoring_metric
        -int cv_folds
        -int random_state
        -any best_ensemble_
        -float best_score_
        -dict best_params_
        +__init__(scoring_metric, cv_folds, random_state)
        +create_base_models(model_info) list
        +create_ensemble_model(ensemble_type, base_models, final_estimator, **kwargs) Classifier
        +optimize_ensemble(ensemble_type, base_models, X, y, param_grid, scorers) tuple
        +mean_results(scorers, grid_search_obj) dict
    }
    
    class DataFrameParser {
        +parse_hyperparameters(hyperparams_str) dict
        +extract_model_info(df, target_models, target_metric, exp_type) dict
    }
    
    class ModelFactory {
        +create_model(model_obj, hyperparams) Classifier
    }
    
    %% Relationships
    ModelTester --> DataHandler : uses for preprocessing
    ModelTester --> CustomBestParamCalculator : uses for optimization
    ModelTester --> Visualizer : uses for visualization
    ModelTester --> "custom_ensamble_best_param_calculator" : uses for ensemble optimization
    
    CustomBestParamCalculator --> MemoryEfficientGridSampler : uses for grid search
    CustomBestParamCalculator --> "create_optuna_study" : uses for Optuna configuration
    
    EnsembleOptimizer --> DataFrameParser : uses for data parsing
    EnsembleOptimizer --> ModelFactory : uses for model creation
    
    DataHandler --> "sklearn.preprocessing" : depends on
    CustomBestParamCalculator --> "optuna" : depends on
    Visualizer --> "matplotlib" : depends on
    Visualizer --> "seaborn" : depends on

flowchart TB

    %% Main Orchestrator
    ModelTester["ModelTester\n(Main Orchestrator)"]

    %% Core Modules
    DataHandler["DataHandler\n(Data Preprocessing Engine)"]
    CustomBestParamCalculator["CustomBestParamCalculator\n(Hyperparameter Optimization)"]
    Visualizer["Visualizer\n(Visualization & Analysis)"]
    CustomEnsembleBestParamCalculator["CustomEnsembleBestParamCalculator\n(Ensemble Optimization)"]
    CreateOptunaStudy["CreateOptunaStudy\n(Optuna Configuration)"]

    %% External Libraries
    Sklearn["scikit-learn"]
    Imblearn["imbalanced-learn\n(SMOTE, ADASYN)"]
    Optuna["Optuna"]
    Matplotlib["matplotlib / seaborn"]
    SQLite["SQLite Storage"]

    %% Data Flow
    ModelTester --> DataHandler
    DataHandler --> CustomBestParamCalculator
    CustomBestParamCalculator --> ModelTester

    ModelTester --> CustomEnsembleBestParamCalculator
    CustomEnsembleBestParamCalculator --> ModelTester

    ModelTester --> Visualizer

    %% Tooling connections
    DataHandler --> Imblearn
    DataHandler --> Sklearn

    CustomBestParamCalculator --> Sklearn
    CustomBestParamCalculator --> Optuna
    CustomBestParamCalculator --> CreateOptunaStudy

    CreateOptunaStudy --> Optuna
    CreateOptunaStudy --> SQLite

    Visualizer --> Matplotlib

    %% Output
    ModelTester --> Results["Results\n(DataFrames, Metrics, Plots)"]



sequenceDiagram
    participant User
    participant ModelTester
    participant DataHandler
    participant Optimizer as CustomBestParamCalculator
    participant EnsembleOpt as CustomEnsembleBestParamCalculator
    participant Visualizer

    User->>ModelTester: Provide dataset, models, metrics
    ModelTester->>DataHandler: Preprocess data
    DataHandler-->>ModelTester: Encoded & resampled dataset

    ModelTester->>Optimizer: Optimize hyperparameters
    Optimizer-->>ModelTester: Best parameters

    ModelTester->>EnsembleOpt: Optimize ensemble models
    EnsembleOpt-->>ModelTester: Optimized ensembles

    ModelTester->>Visualizer: Generate plots & analysis
    Visualizer-->>ModelTester: Visual outputs

    ModelTester-->>User: Performance DataFrames & Results
