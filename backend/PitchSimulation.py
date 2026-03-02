import pandas as pd
# src/pitch_simulation.py
import pandas as pd

class PitchSimulation:
    FEATURES = [
        "pitcher",
        "batter",
        "balls",
        "strikes",
        "game_year",
        "outs_when_up",
        "inning",
        "at_bat_number",
        "bat_score",
        "fld_score",
        "game_type",
        "stand",
        "p_throws",
        "if_fielding_alignment",
        "of_fielding_alignment",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.index = 0
        self.total = len(self.df)

    def next_pitch(self, return_features_only=True) -> pd.Series:
        """
        Returns the next row in the simulation. If return_features_only=True,
        only the columns in FEATURES are returned.
        """
        if self.index >= self.total:
            raise StopIteration("End of held-out dataset reached.")

        row = self.df.iloc[self.index]
        self.index += 1

        if return_features_only:
            return row[self.FEATURES]
        else:
            return row

    def reset(self):
        """Reset the simulation to the first row."""
        self.index = 0