import csv
import queue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def read_file():
    adjacency_list = {}
    max_speed = 0
    with open(edgeFile) as f:
        first_line = f.readline().strip()
        for line in f:
            s = line.strip().split(',')
            s[0], s[1], s[2], s[3] = int(s[0]), float(s[1]), float(s[2]), float(s[3])
            if s[0] not in adjacency_list:
                adjacency_list[s[0]] = []
            adjacency_list[s[0]].append((s[1], s[2], s[3]))
            max_speed = max(max_speed, s[3])
    return adjacency_list, max_speed


def read_h_file():
    heuristic_list = {}
    with open(heuristicFile) as f:
        first_line = f.readline().strip()
        for line in f:
            s = line.strip().split(',')
            node, goal_0, goal_1, goal_2 = int(s[0]), float(s[1]), float(s[2]), float(s[3])
            heuristic_list[node] = (goal_0, goal_1, goal_2)
    return heuristic_list


def astar_time(start, end):
    adjacency_list, max_speed = read_file()
    heuristic_list = read_h_file()

    goal = 0
    max_speed = max_speed / 60 /60 * 1000
    pq = queue.PriorityQueue()
    pq.put((heuristic_list[start][goal] / max_speed, start, [start]))
    visited = set()
    num_visited = 0

    while (pq):
        num_visited += 1
        node = pq.get()

        if node[1] == end:
            dist = node[0]
            path = node[2]
            break

        if (node[1] in visited) or (node[1] not in adjacency_list):
            continue

        for neighbor in adjacency_list[node[1]]:
            if neighbor[0] not in visited:
                temp_path = node[2] + [neighbor[0]]
                new_time = node[0] + neighbor[1] / neighbor[2] + heuristic_list[neighbor[0]][goal] / max_speed - heuristic_list[node[1]][goal] / max_speed
                pq.put((new_time, neighbor[0], temp_path))

        visited.add(node[1])

    return path, dist, num_visited


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
