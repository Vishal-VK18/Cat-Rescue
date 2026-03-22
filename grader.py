"""
Grader module for the Cat Rescue Reinforcement Learning environment.
Evaluates agent performance based on episode logs.
"""

class CatRescueGrader:
    """
    Evaluates an episode log from the Cat Rescue environment and produces
    structured grading results and human-readable summaries.
    """

    def grade(self, episode_log: dict) -> dict:
        """
        Grades an episode based on the provided episode log.
        
        Args:
            episode_log (dict): A dictionary containing:
                - total_steps (int): Total steps taken by the agent.
                - cats_rescued (int): Number of cats successfully rescued.
                - total_cats (int): Total number of cats in the episode.
                - total_reward (float): Total reward accumulated.
                - hit_walls (int): Number of times the agent hit a wall.
                - rescue_log (list of bool): True if rescued, one per cat.
                
        Returns:
            dict: A structured evaluation result containing:
                - cats_rescued_count (int)
                - all_cats_rescued (bool)
                - rescue_details (list of dict)
                - total_steps (int)
                - wall_hits (int)
                - total_reward (float)
                - efficiency_score (float)
                - passed (bool)
                - grade_label (str)
        """
        total_steps = episode_log.get("total_steps", 0)
        cats_rescued = episode_log.get("cats_rescued", 0)
        total_cats = episode_log.get("total_cats", 0)
        total_reward = episode_log.get("total_reward", 0.0)
        hit_walls = episode_log.get("hit_walls", 0)
        rescue_log = episode_log.get("rescue_log", [])
        
        all_cats_rescued = (cats_rescued == total_cats) and (total_cats > 0)
        passed = all_cats_rescued
        
        rescue_details = [{"cat_id": i, "rescued": rescued} for i, rescued in enumerate(rescue_log)]
        
        efficiency_score = 0.0
        if total_steps > 0:
            efficiency_score = min(1.0, float(cats_rescued) / total_steps)
            
        if passed and efficiency_score >= 0.8:
            grade_label = "PERFECT"
        elif passed and efficiency_score >= 0.5:
            grade_label = "GOOD"
        elif cats_rescued > 0 and not passed:
            grade_label = "PARTIAL"
        elif cats_rescued == 0:
            grade_label = "FAIL"
        else:
            # Fallback if passed but efficiency < 0.5
            grade_label = "PASS"
            
        return {
            "cats_rescued_count": cats_rescued,
            "all_cats_rescued": all_cats_rescued,
            "rescue_details": rescue_details,
            "total_steps": total_steps,
            "wall_hits": hit_walls,
            "total_reward": total_reward,
            "efficiency_score": efficiency_score,
            "passed": passed,
            "grade_label": grade_label
        }

    def summary(self, grade_result: dict) -> str:
        """
        Generates a human-readable summary of the evaluation result.
        
        Args:
            grade_result (dict): The result dictionary produced by the grade method.
            
        Returns:
            str: A formatted summary string.
        """
        status = "PASSED" if grade_result.get("passed") else "FAILED"
        grade_label = grade_result.get("grade_label", "UNKNOWN")
        cats_rescued = grade_result.get("cats_rescued_count", 0)
        total_steps = grade_result.get("total_steps", 0)
        efficiency = grade_result.get("efficiency_score", 0.0)
        
        return (
            f"--- Grader Summary ---\n"
            f"Status: {status}\n"
            f"Grade Label: {grade_label}\n"
            f"Cats Rescued: {cats_rescued}\n"
            f"Total Steps: {total_steps}\n"
            f"Efficiency Score: {efficiency:.2f}\n"
            f"----------------------"
        )


if __name__ == "__main__":
    sample_log = {
        "total_steps": 5,
        "cats_rescued": 3,
        "total_cats": 3,
        "total_reward": 15.5,
        "hit_walls": 1,
        "rescue_log": [True, True, True]
    }
    
    grader = CatRescueGrader()
    result = grader.grade(sample_log)
    print("Structured Result:")
    import json
    print(json.dumps(result, indent=2))
    print("\nSummary:")
    print(grader.summary(result))
