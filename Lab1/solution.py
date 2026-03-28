import sys
import queue
import heapq

start_state: str = ""    # The name of the starting state
final_states: list[str] = []   # A list of final state names
transitions: dict[str, dict[str, float]] = {}  # A dictionary of dictionaries -> transitions[state][next_state] = cost
reverse_transitions: dict[str, dict[str, float]] = {}  # For optimistic check 
heuristic: dict[str, float] = {}  # A dictionary -> heuristic[state] = value

# Returns the heuristic file name for the purpose of printing
def read_input() -> str:
    global start_state, final_states, transitions, heuristic
    # Find the --ss flag, it's followed by the filename of the state space
    if "--ss" in sys.argv:
        ss_index = sys.argv.index("--ss") + 1
        ss_filename = sys.argv[ss_index]
        with open(ss_filename, "r") as file:
            # Remove comments
            lines = [line.strip() for line in file if not line.startswith("#")]
            # The first line is the start state
            start_state = lines[0]
            # The second line contains space separated final states
            final_states = lines[1].split(" ")
            
            # The rest of the lines look like this: "state: next_state_1,cost next_state_2,cost ..."
            for line in lines[2:]:
                try:
                    state, rest = line.split(": ")
                    transitions[state] = {}
                    for pair in rest.split(" "):
                        next_state, cost = pair.split(",")
                        transitions[state][next_state] = float(cost)
                except ValueError: # If the state doesn't have transitions "state: " then we don't care about it
                    continue
    # Find the --h flag, it's followed by the filename of the heuristic
    if "--h" in sys.argv:
        h_index = sys.argv.index("--h") + 1
        h_filename = sys.argv[h_index]
        with open(h_filename, "r") as file:
            # The lines look like this: "state: value"
            for line in file:
                if not line.startswith("#"):
                    line.strip()
                    state, value = line.split(": ")
                    heuristic[state] = float(value)
            return h_filename
    return ""

def reconstruct_path(called_by, current_state):
    path = [current_state]  # The path from the start state to the final state in that order ; starts with the final state
    path_length = 1         # Start state is already in the path
    total_cost = 0
    while current_state in called_by:
        path_length += 1
        total_cost += transitions[called_by[current_state]][current_state]
        current_state = called_by[current_state]
        path.insert(0, current_state)   # Insert at the beginning of the list
    
    return path, path_length, total_cost

def print_solution(found_solution, states_visited, path_length, total_cost, path):
    if found_solution:
        print("[FOUND_SOLUTION]: yes")
        print(f"[STATES_VISITED]: {states_visited}")
        print(f"[PATH_LENGTH]: {path_length}")
        print(f"[TOTAL_COST]: {total_cost}")
        print(f"[PATH]: {' => '.join(path)}")
    else:
        print("[FOUND_SOLUTION]: no")

def bfs():
    # Objects used to print the results
    found_solution = False
    states_visited = 0
    called_by = {}  # Used so we can trace back the path ; dictionary -> called_by[state] = previous_state

    # Objects used for functionality
    q = queue.Queue()
    touched_states = set([start_state]) # States that were added to the queue (not necessarily visited yet)
    current_state = ""
    
    q.put(start_state)
    while not q.empty():
        current_state = q.get()
        states_visited += 1
        
        if current_state in final_states:
            found_solution = True
            break
        
        # Add all the next states to the queue, mark them as touched and remember how we got there
        for next_state in transitions.get(current_state, {}):
            if next_state not in touched_states:
                q.put(next_state)
                touched_states.add(next_state)
                called_by[next_state] = current_state
    
    path, path_length, total_cost = reconstruct_path(called_by, current_state)
    print_solution(found_solution, states_visited, path_length, total_cost, path)

def ucs():
    # Objects used to print the results
    found_solution = False
    states_visited = 0
    called_by = {}

    # Objects used for functionality
    heap = [(0, start_state)]   # Priority queue / Min heap - (cost, state_name)
    best_cost = {start_state: 0}    # Dictionary -> best_cost[state] = cost
    current_state = ""
    
    while len(heap) > 0:
        # Pop the state with the lowest cost
        current_cost, current_state = heapq.heappop(heap)
        states_visited += 1
        if current_state in final_states:
            found_solution = True
            break
        
        # Add all the new or cheaper states to the heap
        for next_state, cost in transitions.get(current_state, {}).items():
            new_cost = current_cost + cost
            if next_state not in best_cost or best_cost[next_state] > new_cost:
                best_cost[next_state] = new_cost
                heapq.heappush(heap, (new_cost, next_state))
                called_by[next_state] = current_state
    
    path, path_length, total_cost = reconstruct_path(called_by, current_state)
    print_solution(found_solution, states_visited, path_length, total_cost, path)

def a_star():
    # Objects used to print the results
    found_solution = False
    states_visited = 0
    called_by = {}

    # Objects used for functionality
    # The heap contains tuples (cost + heuristic, cost, state_name), the second element (cost) is there as a tiebreaker
    heap = [(heuristic[start_state], 0, start_state)]
    best_cost = {start_state: 0}
    current_state = ""
    
    while len(heap) > 0:
        # Pop the state with the lowest cost
        _, _, current_state = heapq.heappop(heap)
        states_visited += 1
        if current_state in final_states:
            found_solution = True
            break
        
        for next_state, cost in transitions.get(current_state, {}).items():
            new_cost = best_cost[current_state] + cost
            # Add all the new or cheaper states to the heap
            if next_state not in best_cost or best_cost[next_state] > new_cost:
                best_cost[next_state] = new_cost
                # Add sorted by the cost + heuristic (and the cost as a tiebreaker)
                heapq.heappush(heap, (new_cost + heuristic[next_state], new_cost, next_state))
                called_by[next_state] = current_state
    
    path, path_length, total_cost = reconstruct_path(called_by, current_state)
    print_solution(found_solution, states_visited, path_length, total_cost, path)

def check_optimistic(h_filename):
    global reverse_transitions
    # Create the reverse transitions dictionary
    for state, next_states in transitions.items():
        for next_state in next_states:
            if next_state not in reverse_transitions:
                reverse_transitions[next_state] = {}
            reverse_transitions[next_state][state] = next_states[next_state]
    
    # We will do Dijkstra do find real costs
    # We need to reverse the transitions to find how far each node is from the final states
    # We need to perform Dijkstra for each final state, but we can stop a path early if we already have a shorter one when starting from final state
    best_cost: dict[str, float] = {} # A dictionary that stores the best cost across all final state starts
    for final_state in final_states:
        heap = [(0, final_state)]   # Priority queue / Min heap - (cost, state_name)
        best_cost[final_state] = 0
        current_state = ""
        
        while len(heap) > 0:
            # Pop the state with the lowest cost
            current_cost, current_state = heapq.heappop(heap)
            
            # Add all the new or cheaper states to the heap
            for next_state, cost in reverse_transitions.get(current_state, {}).items():
                new_cost = current_cost + cost
                if next_state not in best_cost or best_cost[next_state] > new_cost:
                    best_cost[next_state] = new_cost
                    heapq.heappush(heap, (new_cost, next_state))

    # Now we can compare the real costs with the heuristic costs
    print("# HEURISTIC-OPTIMISTIC " + h_filename)
    is_optimistic = True
    for state in heuristic:
        if best_cost[state] < heuristic[state]:
            print(f"[CONDITION]: [ERR] h({state}) <= h*: {heuristic[state]} <= {float(best_cost[state])}")
            is_optimistic = False
        else:
            print(f"[CONDITION]: [OK] h({state}) <= h*: {heuristic[state]} <= {float(best_cost[state])}")
    if is_optimistic:
        print("[CONCLUSION]: Heuristic is optimistic.")
    else:
        print("[CONCLUSION]: Heuristic is not optimistic.")

def check_consistent(h_filename):
    # A heuristic is consistent/monotonic if and only if
    # For every state s and every successor s' of s
    # h(s) <= h(s') + cost(s -> s')
    print("# HEURISTIC-CONSISTENT " + h_filename)
    is_consistent = True
    for state, transition in transitions.items():
        for next_state, cost in transition.items():
            if heuristic[state] > heuristic[next_state] + cost:
                print(f"[CONDITION]: [ERR] h({state}) <= h({next_state}) + c: {heuristic[state]} <= {heuristic[next_state]} + {cost}")
                is_consistent = False
            else:
                print(f"[CONDITION]: [OK] h({state}) <= h({next_state}) + c: {heuristic[state]} <= {heuristic[next_state]} + {cost}")
    if is_consistent:
        print("[CONCLUSION]: Heuristic is consistent.")
    else:
        print("[CONCLUSION]: Heuristic is not consistent.")

# MAIN
h_filename = read_input()
# Find the --alg flag, it's followed by the algorithm name
if "--alg" in sys.argv:
    alg_index = sys.argv.index("--alg") + 1
    alg = sys.argv[alg_index]
    if alg == "bfs":
        bfs()
    elif alg == "ucs":
        ucs()
    elif alg == "astar":
        a_star()

if "--check-optimistic" in sys.argv:
    check_optimistic(h_filename)
if "--check-consistent" in sys.argv:
    check_consistent(h_filename)