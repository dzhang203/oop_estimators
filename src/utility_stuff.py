# constants and utility functions/classes

# constants related to DataFrames
# UNIT_ID = 'unit_id'
# TREATMENT_ID = 'treatment_id'
# TIME_ID = 'time_id'


# new Exception subclass
class TodoException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("Sorry, this method is not yet implemented :(")

