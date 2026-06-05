import heapq

class FactorySearch:
    def __init__(self):
        self.locations = {
            'Enter': (0, 0),
            'Warehouse': (2, 5),
            'Area_A': (5, 2),
            'Area_B': (5, 8),
            'Test_Area': (10, 5)
        }

        self.graph = {
            'Enter': {'Warehouse': 3, 'Area_A': 6},
            'Warehouse': {'Enter': 3, 'Area_A': 4, 'Area_B': 4, 'Test_Area': 8},
            'Area_A': {'Enter': 6, 'Warehouse': 4, 'Area_B': 10},
            'Area_B': {'Warehouse': 4, 'Area_A': 10, 'Test_Area': 5},
            'Test_Area': {'Area_B': 5, 'Warehouse': 8}
        }

    def heuristic(self, current, goal):
        (x1, y1) = self.locations[current]
        (x2, y2) = self.locations[goal]
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def a_star(self, start, goal):
        # frontier: (priority, current_node, path, current_cost)
        frontier = [(0 + self.heuristic(start, goal), start, [], 0)]
        visited = {} 

        while frontier:
            (priority, current, path, g_cost) = heapq.heappop(frontier)

            if current == goal:
                return path + [current], g_cost

            if current in visited and visited[current] <= g_cost:
                continue
            
            visited[current] = g_cost

            for neighbor, weight in self.graph.get(current, {}).items():
                new_g_cost = g_cost + weight
                f_cost = new_g_cost + self.heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_cost, neighbor, path + [current], new_g_cost))
        
        return None, float('inf')

    def get_full_path(self, start, goal, needs_warehouse=False, needs_test=False):
        final_path = []
        total_cost = 0
        current_start = start

        # Step 1: Warehouse
        if needs_warehouse:
            p1, c1 = self.a_star(current_start, 'Warehouse')
            if p1:
                final_path.extend(p1)
                total_cost += c1
                current_start = 'Warehouse'
            else: return None, float('inf')

        # Step 2: Goal
        p2, c2 = self.a_star(current_start, goal)
        if p2:
            if final_path:
                final_path.extend(p2[1:]) # Evita duplicati del nodo intermedio
            else:
                final_path.extend(p2)
            total_cost += c2
        else: return None, float('inf')

        # Step 3: Test Area
        if needs_test:
            p3, c3 = self.a_star(goal, 'Test_Area')
            if p3:
                final_path.extend(p3[1:])
                total_cost += c3
            else: return None, float('inf')

        return final_path, total_cost