import unittest
from unittest.mock import patch, MagicMock
import pandas as pd


# Patch model loading before importing main
with patch(
    "backend.utils.load_production_model",
    return_value=(MagicMock(), ["strike", "ball", "hit", "ns foul"])
):
    import backend.main as main


class TestPredictEndpoint(unittest.TestCase):

    def setUp(self):

        # Fake dataframe representing heldout pitch data
        self.df = pd.DataFrame([
            {
                "pitcher": 1,
                "batter": 10,
                "balls": 2,
                "strikes": 1,
                "game_year": 2024,
                "outs_when_up": 1,
                "inning": 5,
                "at_bat_number": 12,
                "bat_score": 3,
                "fld_score": 2,
                "game_type": "R",
                "stand": "R",
                "p_throws": "L",
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
            }
        ])

        main.app.state.df = self.df
        main.app.state.simulation = MagicMock()

    @patch("backend.main.sessions.advance_pitch")
    @patch("backend.main.sessions.get_user")
    def test_predict_returns_normalized_probs_and_advances_pitch(
        self,
        mock_get_user,
        mock_advance_pitch
    ):

        # Simulate existing user session
        mock_get_user.return_value = {"pitch_index": 0}

        # Mock model probabilities
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.3, 0.1, 0.4]]

        main.model = mock_model
        main.labels = ["strike", "ball", "hit", "ns foul"]

        req = main.PeekRequest(user_id="abc123")

        response = main.predict(req)

        probs = response["probabilities"]

        # ns foul should be removed and remaining probabilities normalized
        self.assertAlmostEqual(probs["strike"], 0.2 / 0.6)
        self.assertAlmostEqual(probs["ball"], 0.3 / 0.6)
        self.assertAlmostEqual(probs["hit"], 0.1 / 0.6)
        self.assertNotIn("ns foul", probs)

        # Ensure pitch index advanced
        mock_advance_pitch.assert_called_once_with("abc123", 0)

        # Ensure only expected features returned
        self.assertEqual(
            set(response["pitch"].keys()),
            set(main.PitchSimulation.FEATURES)
        )


if __name__ == "__main__":
    unittest.main()