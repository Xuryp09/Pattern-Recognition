import csv
import queue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def read_file():
    adjacency_list = {}
    with open(edgeFile) as f:
        first_line = f.readline().strip()
        for line in f:
            s = line.strip().split(',')
            s[0], s[1], s[2] = int(s[0]), float(s[1]), float(s[2])
            if s[0] not in adjacency_list:
                adjacency_list[s[0]] = []
            adjacency_list[s[0]].append((s[1], s[2], s[3]))
    return adjacency_list


def read_h_file():
    heuristic_list = {}
    with open(heuristicFile) as f:
        first_line = f.readline().strip()
        for line in f:
            s = line.strip().split(',')
            node, goal_0, goal_1, goal_2 = int(s[0]), float(s[1]), float(s[2]), float(s[3])
            heuristic_list[node] = (goal_0, goal_1, goal_2)
    return heuristic_list


def astar(start, end):
    adjacency_list = read_file()
    heuristic_list = read_h_file()

    goal = 1
    pq = queue.PriorityQueue()
    pq.put((heuristic_list[start][goal], start, [start]))
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
                pq.put((node[0] + neighbor[1] + heuristic_list[neighbor[0]][goal] - heuristic_list[node[1]][goal], neighbor[0], temp_path))

        visited.add(node[1])

    return path, dist, num_visited


if __name__ == '__main__':
    path, dist, num_visited = astar(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')