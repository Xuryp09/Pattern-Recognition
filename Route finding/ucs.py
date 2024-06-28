import csv
import queue
edgeFile = 'edges.csv'


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


def ucs(start, end):
    adjacency_list = read_file()
    pq = queue.PriorityQueue()
    pq.put((0, start, [start]))
    num_visited = 0
    visited = set()

    while(pq):
        node = pq.get()
        num_visited += 1

        if node[1] == end:
            dist = node[0]
            path = node[2]
            break

        if (node[1] in visited) or (node[1] not in adjacency_list):
            continue

        for neighbor in adjacency_list[node[1]]:
            if neighbor[0] not in visited:
                temp_path = node[2] + [neighbor[0]]
                pq.put((node[0] + neighbor[1], neighbor[0], temp_path))

        visited.add(node[1])

    return path, dist, num_visited


if __name__ == '__main__':
    path, dist, num_visited = ucs(1718165260,  8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')