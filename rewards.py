"""
Rewards module for the Cat Rescue Reinforcement Learning environment.
Defines reward constants and computes rewards for different agent events.
"""

class CatRescueRewards:
    """
    Manages and calculates rewards for the Cat Rescue environment.
    """
    STEP_PENALTY = -0.1
    WALL_HIT_PENALTY = -0.3
    CAT_RESCUED_REWARD = 1.0
    ALL_CATS_BONUS = 2.0

    def calculate(self, event: str, context: dict = None) -> float:
        """
        Calculates the appropriate reward for an event.
        
        Args:
            event (str): The name of the event ("step", "wall_hit", "cat_rescued", "all_cats_rescued").
            context (dict, optional): Additional context for the event (e.g., cats_remaining). Defaults to None.
            
        Returns:
            float: The reward value for the selected event. Returns 0.0 for unknown events.
        """
        if context is None:
            context = {}
            
        if event == "step":
            return self.STEP_PENALTY
        elif event == "wall_hit":
            return self.WALL_HIT_PENALTY
        elif event == "cat_rescued":
            return self.CAT_RESCUED_REWARD
        elif event == "all_cats_rescued":
            # If all cats rescued, return CAT_RESCUED_REWARD plus ALL_CATS_BONUS combined
            return self.CAT_RESCUED_REWARD + self.ALL_CATS_BONUS
        else:
            return 0.0

    def calculate_episode_total(self, events: list) -> float:
        """
        Calculates the total cumulative reward for a list of event strings.
        
        Args:
            events (list of str): A list of event strings that occurred during an episode.
            
        Returns:
            float: The total reward accumulated, rounded to 2 decimal places.
        """
        total_reward = 0.0
        for event in events:
            total_reward += self.calculate(event)
        return round(total_reward, 2)

    def get_reward_table(self) -> dict:
        """
        Retrieves a dictionary mapping of all reward constants.
        
        Returns:
            dict: The configured reward values for display or debugging purposes.
        """
        return {
            "STEP_PENALTY": self.STEP_PENALTY,
            "WALL_HIT_PENALTY": self.WALL_HIT_PENALTY,
            "CAT_RESCUED_REWARD": self.CAT_RESCUED_REWARD,
            "ALL_CATS_BONUS": self.ALL_CATS_BONUS
        }


if __name__ == "__main__":
    rewards = CatRescueRewards()
    
    print("--- Reward Table ---")
    for key, value in rewards.get_reward_table().items():
        print(f"{key}: {value:>5}")
        
    print("\n--- Single Event Calculations ---")
    print(f"step:              {rewards.calculate('step')}")
    print(f"wall_hit:          {rewards.calculate('wall_hit')}")
    print(f"cat_rescued:       {rewards.calculate('cat_rescued', {'cats_remaining': 1})}")
    print(f"all_cats_rescued:  {rewards.calculate('all_cats_rescued', {'cats_remaining': 0})}")
    print(f"unknown_event:     {rewards.calculate('sleep')}")
    
    print("\n--- Episode Total Calculation ---")
    sample_events = [
        "step", "step", "wall_hit", "step", 
        "cat_rescued", "step", "step", "all_cats_rescued"
    ]
    total = rewards.calculate_episode_total(sample_events)
    print(f"Events: {sample_events}")
    print(f"Total Cumulative Reward: {total}")
