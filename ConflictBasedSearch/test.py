import cpp_cbs
import os
import numpy as np
import yaml
import subprocess
import csv
import threading
import multiprocessing
import queue

def compute_episode_length(solution):
    ep_len = 0
    for agent in solution:
        if len(agent) > ep_len:
            ep_len = len(agent)
    return ep_len

def compute_max_step(solution):
    max_step = 0
    for agent in solution:
        prev_pos = None
        step_count = 0
        for timestep in agent:
            curr_pos = (timestep[0], timestep[1])
            if prev_pos is not None and prev_pos != curr_pos:
                step_count += 1
            prev_pos = curr_pos
        if step_count > max_step:
            max_step = step_count
    return max_step

def generate_steps(solution):
    steps = []
    for agent in solution:
        prev_pos = None
        step_count = 0
        for timestep in agent:
            curr_pos = (timestep[0], timestep[1])
            if prev_pos is not None and prev_pos != curr_pos:
                step_count += 1
            prev_pos = curr_pos
        steps.append(step_count)
    return steps

def convert_solution(ts_solution):
    n_agents = len(ts_solution)
    solution = [[] for _ in range(n_agents)]
    for agent in range(n_agents):
        timestep = 0
        ep_len = len(ts_solution[agent])
        for timestep in range(ep_len):
            solution[agent].append(tuple([ts_solution[agent][timestep].x, ts_solution[agent][timestep].y, timestep]))
    return solution

def get_csv_logger(model_dir, default_model_name):
    csv_dir = os.path.join(model_dir, default_model_name)
    
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    csv_path = os.path.join(csv_dir, default_model_name + ".csv")

    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def convert_to_tuple(positions):
    new_positions = ()
    for pos in positions:
        new_positions += (tuple(pos),)
    return new_positions

def write_txt_file(filepath, obs, starts, goals):
    with open(filepath, "w") as file:
        # Write dimensions
        file.write(f"{obs.shape[0]} {obs.shape[1]}\n")
        
        # Write obstacle indices
        obstacles = np.where(obs == 1)
        obstacle_indices = [i * obs.shape[1] + j for i, j in zip(obstacles[0], obstacles[1])]
        file.write(" ".join(map(str, obstacle_indices)) + "\n")
        
        # Write start and goal indices for each agent
        for start, goal in zip(starts, goals):
            start_index = start[0] * obs.shape[1] + start[1]
            goal_index = goal[0] * obs.shape[1] + goal[1]
            file.write(f"{start_index} {goal_index}\n")

def cpp_cbs_find_path_with_timeout(time_limit=10):
    # Create a Queue to hold the result
    result_queue = multiprocessing.Queue()

    # Define a function to be run in a separate process
    def find_path(q):
        path = cpp_cbs.find_path()  # map, start, goals, inflation, time_limit
        q.put(path)

    # Start the function in a new process
    process = multiprocessing.Process(target=find_path, args=(result_queue,))
    process.start()

    # Wait for the specified time limit
    process.join(timeout=time_limit)

    # Check if the process is still alive (i.e., the function hasn't returned yet)
    if process.is_alive():
        print("Not solved")
        process.terminate()
        process.join()
        solved = False
        path = None
    else:
        print("Solved")
        solved = True
        path = result_queue.get()

    return solved, path

if __name__ == '__main__':

    dataset_dir = "/home/andrea/Thesis/baselines/Dataset/"
    # map_name = "15_simple_warehouse"
    map_name = "50_55_simple_warehouse"
    model_save_name = "CBS"
    # results_path = "/home/andrea/Thesis/results/simple_warehouse_env/"
    results_path = "/home/andrea/Thesis/results/50_55_simple_warehouse/"
    
    # list_num_agents = [4, 8, 12, 16, 20, 22]
    list_num_agents = [4, 8, 16, 32, 64, 128, 256]
    num_cases = 200

    # Load map
    map_path = os.path.join(dataset_dir, map_name, "input", "map/")
    obs = np.load(map_path + map_name + ".npy")
        
    # Create output directory
    output_dir = os.path.join(dataset_dir, map_name, "output", model_save_name + "/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize csv file
    csv_file, csv_logger = get_csv_logger(results_path, model_save_name)


    for num_agents in list_num_agents:
        agent_dir = os.path.join(dataset_dir, map_name, "input", "start_and_goal", str(num_agents) + "_agents/")
        
        # Create output directory for the current number of agents
        agent_output_dir = os.path.join(output_dir, str(num_agents) + "_agents/")
        if not os.path.exists(agent_output_dir):
            os.makedirs(agent_output_dir)

        # Metrics arrays initialization
        success = []
        total_step_list = []
        avg_step_list = []
        max_step_list = []
        episode_length_list = []

        all_steps = []

        for case_id in range(num_cases):
            case_filepath = agent_dir + map_name + "_" + str(num_agents) + "_agents_ID_" + str(case_id).zfill(5) + ".npy"
            starts, goals = np.load(case_filepath, allow_pickle=True)

            for agent_id in range(num_agents):
                starts[agent_id] = [starts[agent_id][1], starts[agent_id][0]]
                goals[agent_id] = [goals[agent_id][1], goals[agent_id][0]]

            # Convert start and goals to tuple / # row col (y, x)
            # starts = convert_to_tuple(starts)
            # goals = convert_to_tuple(goals)
            write_txt_file("data/map.txt", np.array(obs), starts, goals)

            # print(starts)
            # print(goals)

            # Run OD_M* with the current input file
            # try:
            # odrm*


            timeout = 5 * 60 # seconds
            solved, path = cpp_cbs_find_path_with_timeout(timeout)

            # except OutOfTimeError:
            #     #M* timed out 
            #     print("timeout")
            #     solved = False
            # except NoSolutionError:
            #     print("nosol????")
            #     solved = False

            # print(mstar_path)
            

            
            # print(solution)

            out = dict()
            out["finished"] = solved
            success.append(out["finished"])
            solution = [[] for _ in range(num_agents)]
            if out["finished"]:
                solution = convert_solution(path)
                print(solution)
                steps = generate_steps(solution)
                avg_step = np.mean(steps)
                max_step = np.max(steps)
                tot_step = np.sum(steps)
                ep_len = len(path)
                
                total_step_list.append(tot_step)
                avg_step_list.append(avg_step)
                max_step_list.append(max_step)
                episode_length_list.append(ep_len)

                out["total_step"] = tot_step
                out["avg_step"] = avg_step
                out["max_step"] = max_step
                out["episode_length"] = ep_len
                out["steps"] = steps
                all_steps.append(steps)
            out["crashed"] = False
            out["collisions"] = 0

            # Save the output file
            case_output_filepath = agent_output_dir + "solution_" + model_save_name + "_" + map_name + "_" + str(num_agents) + "_agents_ID_" + str(case_id).zfill(5) + ".npy"
            save_dict = {"metrics": out, "solution": solution}
            np.save(case_output_filepath, save_dict)

        
        # # Compute metrics from all_steps
        # average_steps = np.mean(np.mean(all_steps))
        # max_steps = np.mean(np.max(all_steps))
        # total_steps = np.mean(np.sum(all_steps))
        # print(f"Average steps: {average_steps}; Max steps: {max_steps}; Total steps: {total_steps}")

    
        # Write row in csv file
        header = ["n_agents", "success_rate", "total_step", "avg_step", "max_step", "episode_length", "total_step_std", "avg_step_std", "max_step_std", "episode_length_std", "total_step_min", "avg_step_min", "max_step_min", "episode_length_min", "total_step_max", "avg_step_max", "max_step_max", "episode_length_max"]
        data = [num_agents, np.mean(success)*100, np.mean(total_step_list), np.mean(avg_step_list), np.mean(max_step_list), np.mean(episode_length_list), np.std(total_step_list), np.std(avg_step_list), np.std(max_step_list), np.std(episode_length_list), np.min(total_step_list), np.min(avg_step_list), np.min(max_step_list), np.min(episode_length_list), np.max(total_step_list), np.max(avg_step_list), np.max(max_step_list), np.max(episode_length_list)]
        if num_agents == 4:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

            


            
            
            





