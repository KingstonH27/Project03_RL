"""Per-tick agent hook. Add agent logic inside ``update``."""

from . import API



def update(env):
    """Run once after each game tick and return the current player data."""
    data = update(env)
    score = data["current_score"]
    position = data["player_position"]




    return API.get_player_data(env)
