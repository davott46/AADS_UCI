import itertools
import numpy as np

class StudentDropoutEnsemble:
    """
    Ensemble utilities for Logistic Regression, Random Forest and XGBoost
    dropout-risk models.[1]
    """

    def __init__(self, logistic_model, rf_model, xgb_model):
        self.models = {
            'LogReg': logistic_model,
            'RF'    : rf_model,
            'XGB'   : xgb_model
        }

    # ------------------------------------------------------------------ #
    # internal helper
    # ------------------------------------------------------------------ #
    def _get_prob_vectors(self, X):
        """Return a dict {model_name : 1-D numpy array of P(dropout)}"""
        return {name: mdl.predict_proba(X)[:, 1] for name, mdl in self.models.items()}


    def average_combinations(self, X, min_models=1):
        """
        Compute simple-average probabilities for every model combination.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        min_models : int, default=1
            Smallest subset size to consider.
            * 1  → include single models as well
            * 2  → only pairwise and triple combinations, etc.

        Returns
        -------
        dict
            Keys  : e.g. 'LogReg', 'RF_XGB', 'LogReg_RF_XGB'
            Values: numpy array with averaged probabilities for that subset.
        """
        prob_dict   = self._get_prob_vectors(X)
        combo_probs = {}

        for r in range(min_models, len(prob_dict) + 1):
            for combo in itertools.combinations(prob_dict.keys(), r):
                name              = '_'.join(combo)      # readable key
                combo_matrix      = [prob_dict[m] for m in combo]
                combo_probs[name] = np.mean(combo_matrix, axis=0)

        return combo_probs