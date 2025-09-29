import optuna


def create_optuna_study(optim_mode, n_trials):
    def objective(trial):
        if 'also' in optim_mode:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'pilr': trial.suggest_float('pilr', 1e-5, 1e-3, log=True),
                'pidecay': trial.suggest_float('pidecay', 1e-3, 1, log=True),
            }
        elif 'drago' in optim_mode:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'pilr': trial.suggest_float('pilr', 1e-5, 1e-3, log=True),
                'pidecay': trial.suggest_float('pidecay', 1e-3, 1, log=True),
                'freq': trial.suggest_int('freq', 1, 51, step=10)
            }
        elif 'dro' in optim_mode:
            if 'largescale' in optim_mode:
                return {
                    'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                    'size': trial.suggest_float('size', 0.001, 0.999),
                    'reg': trial.suggest_float('reg', 0.001, 1)
                }
            else:
                return {
                    'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                    'shift_cost': trial.suggest_float('shift_cost', 0.001, 1, log=True),
                    'ndraws': trial.suggest_float('ndraws', 0.001, 10, log=True)
                }
        elif 'exp' in optim_mode:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'tau': trial.suggest_float('tau', 0.01, 3.0),
            }
        elif 'adamw' in optim_mode:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            }
        else:
            1/0
    
    study = optuna.create_study(direction='maximize')
    return study, objective
