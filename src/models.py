import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from utility_stuff import *

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
        unit_id: str,
        treatment_id: str,
        outcome_id: str,
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

    def explore_outcome(self):
        outcome_data = self.data[self.outcome_id]  # pd.Series
        
        # binary outcome data
        if self.is_binary_outcome:
            print(outcome_data.value_counts())
            plt.clf()
            sns.barplot(x=outcome_data)
            plt.show()

        # continuous outcome data
        else:
            outcome_mean = outcome_data.mean()
            print('{} mean: {:.3f}'.format(
                self.outcome_id,
                outcome_mean
            ))

            print('\nQuantiles:')
            print(outcome_data.quantile([
                0.05,
                0.10,
                0.50,
                0.90,
                0.95
            ]))

            plt.clf()
            print('\nFull Histogram:')
            sns.histplot(
                x=outcome_data,
                stat='proportion',
                bins=15,
                kde=True,
            )
            plt.show()

            plt.clf()
            print('Histogram with Outliers Removed')
            outcome_data_inliers = outcome_data[(
                (outcome_data > outcome_data.quantile(0.02))
                & (outcome_data < outcome_data.quantile(0.98))
            )]
            sns.histplot(
                x=outcome_data_inliers,
                stat='proportion',
                bins=15,
                kde=True,
            )
            plt.show()

    def explore_treatment(self):
        # TODO: write similar to explore_outcome
        # but, maybe refactor to belong to Experiment() class?
        raise TodoException


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