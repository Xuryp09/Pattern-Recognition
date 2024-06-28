import csv
from collections import deque
edgeFile = 'edges.csv'


def bfs(start, end):
    adjacency_list = {}
    with open(edgeFile) as f:
        first_line = f.readline().strip()
        for line in f:
            s = line.strip().split(',')
            s[0], s[1], s[2] = int(s[0]), float(s[1]), float(s[2])
            if s[0] not in adjacency_list:
                adjacency_list[s[0]] = []
            adjacency_list[s[0]].append((s[1], s[2], s[3]))

    visited = set()
    q = deque([(start, 0, [start])])
    visited.add(start)
    num_visited = 0

    while(q):
        node = q.popleft()
        num_visited += 1

        if node[0] == end:
            dist = node[1]
            path = node[2]
            break

        if node[0] not in adjacency_list:
            continue

        for neighbor in adjacency_list[node[0]]:
            if neighbor[0] not in visited:
                visited.add(neighbor[0])
                temp_path = node[2] + [neighbor[0]]
                q.append((neighbor[0], node[1] + neighbor[1], temp_path))

    return path, dist, num_visited
            

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')