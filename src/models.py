import pandas as pd 

# constants
UNIT_ID = 'unit_id'
TREATMENT_ID = 'treatment_id'
TIME_ID = 'time_id'


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
            time_id=None,
            ignore_var_list=None,
        ) -> None:
        self.data = data 
        self.unit_id = unit_id 
        self.treatment_id = treatment_id
        self.outcome_id = outcome_id
        self.time_id = time_id


class Experiment():
    """
    top-level class for experiments.
    contains data with context (unit id variable, treatment id variable, etc.).
    we'll primarily use this for estimating regression models and returning result objects.
    """
    def __init__(self, data) -> None:
        self.data = data 