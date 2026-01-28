# Classification Model Tester - Complete Documentation

The Classification Model Tester is a comprehensive Python library for end-to-end machine learning model testing, optimization, and visualization. It follows a modular architecture with six core classes that work together to provide a complete pipeline from data preprocessing to model deployment.

## Class-by-Class Detailed Documentation

### 1. **ModelTester** - The Main Orchestrator
**Location:** `model_tester.py`
**Purpose:** Central coordinator that manages the entire testing pipeline

**Core Responsibilities:**
- Orchestrates all operations across different modules
- Manages model initialization and configuration
- Coordinates hyperparameter optimization processes
- Handles ensemble model creation and optimization
- Controls the visualization pipeline
- Tracks and stores performance metrics
- Provides high-level API for end users

**Key Features:**
- Unified interface for all library functionalities
- Automatic performance tracking across experiments
- Support for base models and ensemble models
- Integration with resampling techniques
- Cross-validation management
- Results aggregation and DataFrame generation

**Data Flow:**
1. Receives raw data, model specifications, and metrics
2. Delegates preprocessing to DataHandler
3. Calls CustomBestParamCalculator for optimization
4. Implements optimized models
5. Generates visualizations through Visualizer
6. Aggregates all results into structured format

### 2. **DataHandler** - Data Preprocessing Engine
**Location:** `data_handler.py`
**Purpose:** Comprehensive data preprocessing and transformation

**Core Responsibilities:**
- Automatic data type detection (categorical, numerical, boolean)
- One-Hot Encoding for categorical features
- Label Encoding for target variables
- Dataset reconstruction after encoding
- Data resampling (SMOTE, ADASYN, etc.)
- Train/Test splitting with stratification
- Mapping management for encoded features

**Key Features:**
- Intelligent type detection without manual specification
- Preserves data relationships during encoding
- Support for multiple resampling techniques
- Memory-efficient dataset reconstruction
- Automatic handling of boolean columns
- Integration with imbalanced-learn library

**Processing Pipeline:**
1. Type detection on all columns
2. One-Hot Encoding for categoricals
3. Label Encoding for target
4. Dataset concatenation
5. Optional resampling
6. Train/Test split

### 3. **CustomBestParamCalculator** - Hyperparameter Optimization Core
**Location:** `custom_best_param_calulator.py`
**Purpose:** Advanced hyperparameter optimization with multiple strategies

**Core Responsibilities:**
- GridSearchCV and RandomizedSearchCV integration
- Optuna optimization with TPE sampler
- Cross-validation strategy management
- Metric scorer factory creation
- Best parameter extraction and storage

**Key Features:**
- Optuna integration with intelligent pruning
- Custom objective functions with timeout protection
- Multi-metric optimization (accuracy, precision, recall, F1)
- Per-class metric optimization capability
- SQLite storage for resumable optimization

**Optimization Strategies:**
1. **Traditional scikit-learn**: GridSearchCV, RandomizedSearchCV
2. **Optuna TPE**: Tree-structured Parzen Estimator
3. **Brute-force Grid**: MemoryEfficientGridSampler for exhaustive search
4. **Hybrid Approaches**: Combination of different strategies

### 4. **Visualizer** - Visualization and Analysis Suite
**Location:** `visualizer.py`
**Purpose:** Comprehensive visualization and model analysis

**Core Responsibilities:**
- Statistical distribution analysis
- Learning curve generation
- Validation curve plotting
- Correlation matrix visualization
- Model performance analysis
- Plot saving and formatting
- Data quality assessment

**Key Features:**
- Automatic plot sizing and formatting
- Support for original and resampled data
- Integration with matplotlib and seaborn
- Custom color schemes and styling
- Learning curve analysis for bias/variance
- Validation curves for parameter tuning
- Correlation analysis for feature selection

**Visualization Categories:**
1. **Data Analysis**: Pie charts, box plots, histograms
2. **Correlation**: Heatmaps, correlation matrices
3. **Model Analysis**: Learning curves, validation curves
4. **Distribution**: Feature distributions, target distributions
5. **Quality**: Data quality assessments, missing value analysis

### 5. **CustomEnsambleBestParamCalculator** - Ensemble Optimization Module
**Location:** `custom_ensamble_best_param_calculator.py`
**Purpose:** Specialized optimization for ensemble models

**Core Responsibilities:**
- Voting Classifier optimization
- Stacking Classifier configuration
- Ensemble weight optimization
- Base model selection and combination
- Performance DataFrame parsing
- Factory pattern for model creation
- Ensemble-specific hyperparameter tuning

**Key Components:**
- **EnsembleOptimizer**: Main optimization engine
- **DataFrameParser**: Extracts model information from performance DataFrames
- **ModelFactory**: Creates models with specific hyperparameters

**Key Features:**
- Automatic ensemble creation from base models
- Optimization of voting weights (soft/hard)
- Stacking with customizable final estimators
- Integration with existing optimization results
- Support for heterogeneous model ensembles
- Cross-validation for ensemble performance

### 6. **CreateOptunaStudy** - Optuna Configuration Utility
**Location:** `create_optuna_study.py`
**Purpose:** Optuna study configuration and management

**Core Responsibilities:**
- Optuna study creation with proper configuration
- Sampler selection (TPE, Random)
- Pruner configuration (MedianPruner)
- Storage setup (SQLite for persistence)
- Study naming and organization
- Random seed management

**Key Features:**
- Pre-configured optimal settings for Optuna
- SQLite integration for resumable studies
- Automatic study naming with UUID
- Configurable direction (maximize/minimize)
- Integration with main optimization pipeline

## Architecture Diagrams

### High-Level Architecture
```mermaid
flowchart TB
    subgraph "Main Orchestrator"
        MT[ModelTester<br/>Central Coordinator]
    end
    
    subgraph "Core Processing Modules"
        DH[DataHandler<br/>Preprocessing Engine]
        CBC[CustomBestParamCalculator<br/>Optimization Core]
        VZ[Visualizer<br/>Visualization Suite]
        CEC[CustomEnsembleBestParamCalculator<br/>Ensemble Optimizer]
    end
    
    subgraph "Configuration Layer"
        COS[CreateOptunaStudy<br/>Optuna Config]
    end
    
    subgraph "External Dependencies"
        SKL[scikit-learn]
        OPT[Optuna]
        IMB[imbalanced-learn]
        MAT[matplotlib/seaborn]
        PD[pandas/numpy]
    end
    
    %% Core Relationships
    MT --> DH
    MT --> CBC
    MT --> VZ
    MT --> CEC
    
    DH --> SKL
    DH --> IMB
    DH --> PD
    
    CBC --> SKL
    CBC --> OPT
    CBC --> COS
    
    VZ --> MAT
    VZ --> PD
    
    CEC --> SKL
    CEC --> PD
    
    COS --> OPT
    
    %% Output
    MT --> Results[Performance Results<br/>Optimized Models<br/>Visual Reports]

flowchart TD
    %% Input Layer
    Input[Raw Data + Model Specs + Metrics]
    
    %% Processing Pipeline
    subgraph "Processing Pipeline"
        P1[Phase 1: Data Preprocessing]
        P2[Phase 2: Hyperparameter Optimization]
        P3[Phase 3: Model Implementation]
        P4[Phase 4: Ensemble Optimization]
        P5[Phase 5: Visualization]
    end
    
    %% Output Layer
    Output[Performance Metrics<br/>Optimized Models<br/>Visual Reports<br/>Ensemble Models]
    
    %% Detailed Flow
    Input --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> Output
    
    %% Phase Details
    subgraph " "
        P1_Details[One-Hot Encoding<br/>Label Encoding<br/>Resampling<br/>Train/Test Split]
        P2_Details[GridSearch/Optuna<br/>Cross-Validation<br/>Parameter Tuning]
        P3_Details[Model Configuration<br/>Performance Validation<br/>Results Storage]
        P4_Details[Voting/Stacking<br/>Ensemble Weights<br/>Combination Optimization]
        P5_Details[Learning Curves<br/>Validation Curves<br/>Statistical Plots<br/>Correlation Analysis]
    end
    
    P1 --> P1_Details
    P2 --> P2_Details
    P3 --> P3_Details
    P4 --> P4_Details
    P5 --> P5_Details

graph TB
    subgraph "Classification Model Tester - Core Architecture"
        %% Main Classes
        MT[ModelTester<br/>Main Orchestrator]
        DH[DataHandler<br/>Data Preprocessing]
        CBC[CustomBestParamCalculator<br/>Hyperparameter Optimization]
        VZ[Visualizer<br/>Visualization & Analysis]
        CEC[CustomEnsambleBestParamCalculator<br/>Ensemble Optimization]
        COS[CreateOptunaStudy<br/>Optuna Configuration]
        
        %% External Dependencies
        subgraph "External Libraries"
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
        
        %% Module Dependencies
        DH --> SKL
        DH --> PD
        DH --> NP
        DH --> IMB
        
        CBC --> SKL
        CBC --> OPT
        CBC --> COS
        
        VZ --> MAT
        VZ --> SNS
        VZ --> PD
        
        CEC --> SKL
        CEC --> XGB
        CEC --> PD
        
        COS --> OPT
    end
