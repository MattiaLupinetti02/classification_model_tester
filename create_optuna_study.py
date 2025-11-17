import optuna
import uuid
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def create_optuna_study(
    direction: str = "maximize",
    storage_path: str | None = None,
    sampler_type: str = "tpe",
    seed: int = 42
):
    """
    Crea una Optuna Study configurata con TPE e pruning.
    Se viene fornito un percorso di storage, salva i risultati per ripresa successiva.
    """
    if sampler_type == "tpe":
        sampler = TPESampler(seed=seed, multivariate=True)
    elif sampler_type == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    pruner = MedianPruner(n_warmup_steps=5)

    # Usa UUID standard invece della funzione rimossa
    study_name = f"study_{uuid.uuid4().hex[:8]}"

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{storage_path}" if storage_path else None,
        load_if_exists=True
    )

    return study
