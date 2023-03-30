# constants and utility functions/classes
import pandas as pd

# CONVENIECE FUNCS FOR DATA HANDLING
def GetInlierDataFromQuantiles(
    srs: pd.Series,
    q_left: float=0.02,
    q_right: float=0.98,
):
    return srs[(
        (srs >= srs.quantile(q_left))
        & (srs <= srs.quantile(q_right))
    )]


# new Exception subclass
class TodoException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("Sorry, this method is not yet implemented :(")

