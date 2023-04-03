import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from src.utility_stuff import *
from scipy import linalg as alg 

# classes
class DataWithContext():
    """
    top-level class for data.
    primarily use this for
    (1) defining Experiment objects,
    (2) running some standard diagnostics
    """
    def __init__(
        self,
        data: pd.DataFrame,
        treatment_id: str,
        outcome_id: str,
        unit_id: str=None,
        time_id: str=None,
    ) -> None:
        self.data = data 
        self.unit_id = unit_id 
        self.treatment_id = treatment_id
        self.outcome_id = outcome_id
        self.time_id = time_id

        # determine if outcome is binary or non-binary
        self.is_binary_outcome = True
        if len(data[outcome_id].drop_duplicates()) > 2:
            self.is_binary_outcome = False

        # determine if treatment is binary or non-binary
        self.is_binary_treatment = True
        if len(data[outcome_id].drop_duplicates()) > 2:
            self.is_binary_treatment = False 

        # extract outcome data & treatment data
        self.outcome_data = self.data[self.outcome_id]
        self.treatment_data = self.data[self.treatment_id]

    def explore(self):
        self.explore_outcome()
        self.explore_treatment()
        self.explore_y_x()

    def explore_outcome(self):
        self._explore_univariate(
            self.outcome_data,
            self.outcome_id,
        )

    def explore_treatment(self):
        self._explore_univariate(
            self.treatment_data,
            self.treatment_id,
        )

    def _explore_univariate(self, srs, var_name=''): 
        # binary outcome data
        if self.is_binary_outcome:
            print(srs.value_counts())
            plt.clf()
            sns.barplot(x=srs)
            plt.show()

        # continuous outcome data
        else:
            outcome_mean = srs.mean()
            print('{} mean: {:.3f}'.format(
                var_name,
                outcome_mean,
            ))

            print('\nQuantiles:')
            print(srs.quantile([
                0.05,
                0.10,
                0.50,
                0.90,
                0.95
            ]))

            plt.clf()
            print('\nFull Histogram:')
            sns.histplot(
                x=srs,
                stat='proportion',
                bins=15,
                kde=True,
            )
            plt.show()

            srs_inliers = GetInlierDataFromQuantiles(srs)
            plt.clf()
            print('Histogram with Outliers Removed')
            sns.histplot(
                x=srs_inliers,
                stat='proportion',
                bins=15,
                kde=True,
            )
            plt.show()
    
    def explore_y_x(self):
        self._explore_bivariate(
            self.treatment_data,
            self.outcome_data,
        )

    def _explore_bivariate(
        self,
        x,
        y,
    ):
        print('Full Scatterplot:')
        plt.clf()
        sns.scatterplot(x=x, y=y)
        plt.show()

    def make_experiment(
        self,
        data_structure='cross',
        continuous_covariates=[],
        discrete_covariates=[],
        interactions=[],
    ):
        if data_structure in ('cross', 'ols'):
            return CrossExperiment(
                self.data,
                self.outcome_id,
                self.treatment_id,
                continuous_covariates,
                discrete_covariates,
                interactions,
            )
        elif data_structure in ('long', 'longitudinal', 'panel'):
            raise TodoException
        else:
            raise NameError('invalid experiment data structure: {}'.format(
                data_structure
            ))


class CrossExperiment():
    def __init__(
        self,
        data,
        outcome_id,
        treatment_id,
        continuous_covariates=[],
        discrete_covariates=[],
        interactions=[],
        add_constant=True,
    ) -> None:
        self.data = data
        self.outcome_id = outcome_id 
        self.treatment_id = treatment_id 
        self.continuous_covariates = continuous_covariates
        self.discrete_covariates = discrete_covariates
        self.interactions = interactions 

        # extract Y,X,Interact as vectors/matrices
        self.Y, self.Y_name_list = PandasToArrayAndNames(self.data, [self.outcome_id])
        self.X_treatment, self.X_treatment_name_list = PandasToArrayAndNames(
            self.data,
            [self.treatment_id]
        )
        # TODO: fix this code below
        self.X_continuous, self.X_continuous_name_list = PandasToArrayAndNames(
            self.data,
            self.continuous_covariates
        )
        if add_constant:
            start_size_X = self.X_continuous.size
            self.X_continuous = sm.add_constant(self.X_continuous)
            if self.X_continuous.size > start_size_X:
                self.X_continuous_name_list = ['_constant'] + self.X_continuous_name_list
        # TODO: fix this code below
        self.X_discrete, self.X_discrete_name_list = PandasToArrayAndNames(
            self.data,
            self.discrete_covariates
        )
        # TODO: handle interactions
        self.X_name_list = self.X_treatment_name_list \
            + self.X_continuous_name_list \
            + self.X_discrete_name_list
        self.X = np.concatenate(
            (
                self.X_treatment,
                self.X_continuous,
                self.X_discrete, 
                # TODO: add X_interactions
            ),
            axis=1,
        )

        # make formula
        self.formula = self.Y_name_list[0] \
            + '\n~\n' \
            + '\n+ '.join(self.X_name_list)

    def estimate(
        self,
        std_error_type: str='iid',
    ):
        if std_error_type == 'iid':
            return self._estimate_iid(std_error_type)
        elif std_error_type == 'White':
            return self._estimate_white(std_error_type)
        else:
            raise TodoException

    def _estimate_iid(self, std_error_type):
        XX_inv = alg.inv(self.X.T @ self.X)  # (k,k) matrix
        beta = (
            XX_inv
            @ self.X.T
            @ self.Y
        )
        residuals = self.Y - self.X @ beta
        degrees_of_freedom = len(beta)
        residual_var = np.var(residuals, ddof=degrees_of_freedom)  # int
        N = len(self.Y)  # int 
        # beta_std_error = np.sqrt(np.diag(
        #     XX_inv
        #     * residual_var 
        # ))
        beta_var_matrix = (
            XX_inv
            * residual_var
        )
        beta_std_error = np.sqrt(np.diag(
            beta_var_matrix
        ))

        return CrossResult(
            beta, 
            self.X_name_list, 
            beta_std_error,
            std_error_type,
            self.Y,
            self.X, 
            residuals,
            self.formula,
        )
    
    def _estimate_white(self, std_error_type):
        XX_inv = alg.inv(self.X.T @ self.X)  # (k,k) matrix
        print('XX_inv shape {}'.format(XX_inv.shape))
        beta = (
            XX_inv 
            @ self.X.T 
            @ self.Y
        )
        residuals = self.Y - self.X @ beta  # (n,1) vector
        # G = self.X.T @ residuals  # (k,1) vector
        N = len(self.Y)  # int
        # GG_diag = np.diag(np.diag(G @ G.T))
        beta_var_matrix = (
            XX_inv 
            @ self.X.T 
            @ np.diag(np.diag(residuals @ residuals.T))
            @ self.X
            @ XX_inv
        )
        beta_std_error = np.sqrt(np.diag(
            beta_var_matrix
        ))
        # TODO: Write up lessons:
        # 1) You can't just use "G" i.e. self.X.T @ residuals, directly (why?...)
        # 2) Convince myself about the correctness of not dividing anything here by N?...
        #   I guess that's just what the derivation gives?
        
        return CrossResult(
            beta, 
            self.X_name_list, 
            beta_std_error,
            std_error_type,
            self.Y,
            self.X, 
            residuals,
            self.formula,
        ) 


class CrossResult():
    def __init__(
        self,
        beta,
        beta_names,
        beta_std_error,
        std_error_type,
        Y,
        X,
        residuals,
        formula,
    ) -> None:
        self.beta = beta 
        self.beta_names = beta_names 
        self.beta_std_error = beta_std_error 
        self.std_error_type = std_error_type 
        self.Y = Y 
        self.X = X 
        self.residuals = residuals 
        self.formula = formula
    
    def summary(self):
        raise TodoException
