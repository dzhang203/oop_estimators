import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from src.utility_stuff import *

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
        raise TodoException


class Experiment():
    """
    top-level class for experiments.
    contains data with context (unit id variable, treatment id variable, etc.).
    we'll primarily use this for estimating regression models and returning result objects.
    """
    def __init__(self, data) -> None:
        self.data = data 