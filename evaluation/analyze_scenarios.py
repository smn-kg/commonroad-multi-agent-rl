import os
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader


def main():

    current_path = Path(__file__).parent
    scenario_dir = current_path.parent / "scenarios" / "highD"

    scenario_files = [f for f in os.listdir(scenario_dir) if f.endswith(".xml")]

    print(f"Found {len(scenario_files)} scenario files in {scenario_dir}")

    num_obstacles_list = []
    planning_problem_len_list = []

    for i, scenario_file in enumerate(scenario_files):

        print(f"Processing scenario {i + 1}/{len(scenario_files)}")

        scenario, planning_problem_set = CommonRoadFileReader(scenario_dir / scenario_file).open()
        planning_problem = planning_problem_set.planning_problem_dict.values().__iter__().__next__()

        num_obstacles = len(scenario.obstacles)
        planning_problem_len = (planning_problem.goal.state_list[0].time_step.start + planning_problem.goal.state_list[0].time_step.end) / 2


        num_obstacles_list.append(num_obstacles)
        planning_problem_len_list.append(planning_problem_len)

        if i == 500:
            break

    print(f"Average number of obstacles: {sum(num_obstacles_list) / len(num_obstacles_list)}")
    print(f"Average planning problem length: {sum(planning_problem_len_list) / len(planning_problem_len_list)}")





if __name__ == "__main__":
    main()