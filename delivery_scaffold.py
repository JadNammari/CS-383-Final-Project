"""
Campus Delivery Project -- Starter Scaffold
==============================================

This file gives you two things:

  1. load_data()          -- reads the three CSV files
  2. check_solution()     -- an INDEPENDENT checker: tells you whether
                              your delivery plan is legal, and what it
                              costs. Run this constantly while you work.

It does NOT tell you how to solve the problem. That's your team's job.
You've learned linear/binary search, sorting, heaps, BSTs, hash tables,
graph traversal, Dijkstra, MSTs, dynamic programming, and greedy
algorithms this semester -- which of those you use, and how you combine
them, is a design decision your team makes and defends in your report.

Some teams might reach for a hash table plus a priority queue plus
Dijkstra. Others might combine Dijkstra with dynamic programming on a
small subset. Others might do something we haven't listed. All of
these are legitimate starting points -- there is no single intended
answer.


THE PROBLEM
-------------
A delivery agent starts at the depot (see DATASET_GUIDE.txt), carries
out a list of delivery requests, and returns to the depot. Each
request means: go to its pickup building, then go to its dropoff
building. The agent can carry several items at once, up to the weight
limit. Your job is to decide the ORDER in which to do everything, to
keep total travel cost as low as possible.


YOUR SOLUTION FORMAT
-----------------------
Return a plan as a list of stops, like this:

    plan = [
        {"building": "B16", "action": "start"},
        {"building": "B04", "action": "pickup",  "request_id": "R01"},
        {"building": "B05", "action": "dropoff", "request_id": "R01"},
        {"building": "B16", "action": "end"},
    ]

Rules:
  - The plan must start and end at the depot.
  - Every request must be picked up once and dropped off once, dropoff
    after pickup.
  - Total weight carried at once must never exceed capacity.

You don't need to use directly-connected buildings for consecutive
stops -- check_solution() assumes you walk the shortest path between
whatever stops you choose, and computes that cost for you. You are
choosing the ORDER of stops, not the individual footpaths.
"""

import csv
import heapq


# ---------------------------------------------------------------------
# Loading the data
# ---------------------------------------------------------------------

def load_data(building_file="buildings.csv", pathway_file="pathways.csv",
              delivery_file="deliveries.csv"):
    """Read the three CSV files. Returns (buildings, pathways, deliveries),
    each a list of dicts."""
    with open(building_file, newline="") as f:
        buildings = list(csv.DictReader(f))
    with open(pathway_file, newline="") as f:
        pathways = list(csv.DictReader(f))
        for p in pathways:
            p["distance_m"] = float(p["distance_m"])
            p["time_s"] = float(p["time_s"])
    with open(delivery_file, newline="") as f:
        deliveries = list(csv.DictReader(f))
        for d in deliveries:
            d["size_kg"] = float(d["size_kg"])
    return buildings, pathways, deliveries


# ---------------------------------------------------------------------
# Independent cost model, used by check_solution().
# You'll likely want your own faster version inside your solver --
# this one favors clarity.
# ---------------------------------------------------------------------

def _shortest_costs(pathways, buildings):
    graph = {b["building_id"]: [] for b in buildings}
    for p in pathways:
        graph[p["building_a"]].append((p["building_b"], p["distance_m"]))
        graph[p["building_b"]].append((p["building_a"], p["distance_m"]))

    costs = {}
    for source in graph:
        dist = {source: 0}
        queue = [(0, source)]
        seen = set()
        while queue:
            d, node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            for neighbour, w in graph[node]:
                nd = d + w
                if neighbour not in dist or nd < dist[neighbour]:
                    dist[neighbour] = nd
                    heapq.heappush(queue, (nd, neighbour))
        costs[source] = dist
    return costs


# ---------------------------------------------------------------------
# check_solution -- do not modify. This is what we use for grading and
# for comparing results across teams, so use the SAME function you're
# handed rather than a lookalike you wrote yourself.
# ---------------------------------------------------------------------

def check_solution(buildings, pathways, deliveries, plan,
                    depot="B16", capacity=15.0):
    """
    Check whether `plan` is a legal delivery plan, and score it.

    Returns a dict:
        {
          "valid": bool,
          "errors": [str, ...],
          "total_distance_m": number,
          "requests_delivered": int,
          "requests_total": int,
        }
    """
    errors = []
    costs = _shortest_costs(pathways, buildings)
    valid_buildings = set(costs)
    requests = {d["request_id"]: d for d in deliveries}

    if not plan or plan[0].get("building") != depot or plan[0].get("action") != "start":
        errors.append("plan must start at the depot with action 'start'")
    if not plan or plan[-1].get("building") != depot or plan[-1].get("action") != "end":
        errors.append("plan must end at the depot with action 'end'")

    picked_up, dropped_off = set(), set()
    onboard = 0.0
    total_distance = 0.0

    for i, stop in enumerate(plan):
        building = stop.get("building")
        action = stop.get("action")

        if building not in valid_buildings:
            errors.append(f"stop {i}: unknown building {building!r}")
            continue

        if i > 0:
            prev = plan[i - 1]["building"]
            if prev in valid_buildings:
                leg = costs[prev].get(building)
                if leg is None:
                    errors.append(f"stop {i}: no route from {prev} to {building}")
                else:
                    total_distance += leg

        request_id = stop.get("request_id")

        if action == "pickup":
            if request_id not in requests:
                errors.append(f"stop {i}: unknown request {request_id!r}")
                continue
            if request_id in picked_up:
                errors.append(f"request {request_id} picked up more than once")
            if requests[request_id]["pickup"] != building:
                errors.append(f"request {request_id} picked up at the wrong building")
            picked_up.add(request_id)
            onboard += requests[request_id]["size_kg"]
            if onboard > capacity + 1e-9:
                errors.append(f"stop {i}: over capacity ({onboard:.1f} kg > {capacity} kg)")

        elif action == "dropoff":
            if request_id not in requests:
                errors.append(f"stop {i}: unknown request {request_id!r}")
                continue
            if request_id not in picked_up:
                errors.append(f"request {request_id} dropped off before pickup")
            if request_id in dropped_off:
                errors.append(f"request {request_id} dropped off more than once")
            if requests[request_id]["dropoff"] != building:
                errors.append(f"request {request_id} dropped off at the wrong building")
            dropped_off.add(request_id)
            onboard -= requests[request_id]["size_kg"]

        elif action not in ("start", "end"):
            errors.append(f"stop {i}: unknown action {action!r}")

    missing = set(requests) - dropped_off
    if missing:
        errors.append(f"{len(missing)} request(s) never delivered: {sorted(missing)[:5]}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_distance_m": round(total_distance, 1),
        "requests_delivered": len(dropped_off),
        "requests_total": len(requests),
    }


def print_report(result, label=""):
    """Pretty-print a check_solution() result. Put this in your report."""
    print(f"--- {label} ---" if label else "---")
    print(f"valid:               {result['valid']}")
    if not result["valid"]:
        for e in result["errors"][:6]:
            print(f"    - {e}")
    print(f"total distance:      {result['total_distance_m']} m")
    print(f"requests delivered:  {result['requests_delivered']} / {result['requests_total']}")


# ---------------------------------------------------------------------
# YOUR WORK STARTS HERE
# ---------------------------------------------------------------------

def build_graph(pathways):
    """Build an undirected adjacency list from pathways.csv rows.

    Each neighbor entry is [building_id, distance_m, time_s], matching
    the edge format used by dijkstra() below.
    """
    graph = {}
    for p in pathways:
        a, b = p["building_a"], p["building_b"]
        graph.setdefault(a, []).append([b, p["distance_m"], p["time_s"]])
        graph.setdefault(b, []).append([a, p["distance_m"], p["time_s"]])
    return graph


def dijkstra(adj, src):
    """Shortest paths from src by distance.

    Returns (dist, path, time_path):
      dist[v]      -- shortest distance (m) from src to v
      path[v]      -- list of building ids along that route
      time_path[v] -- travel time (s) along that same route
    """
    pq = []
    dist = {u: float("inf") for u in adj}
    dist[src] = 0
    heapq.heappush(pq, (0, src))

    path = {u: [] for u in adj}
    path[src] = [src]

    time_path = {u: 0 for u in adj}
    time_path[src] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for x in adj[u]:
            v, weight, time = x[0], x[1], x[2]
            if dist[v] > d + weight:
                path[v] = path[u] + [v]
                dist[v] = d + weight
                time_path[v] = time_path[u] + time
                heapq.heappush(pq, (dist[v], v))

    return dist, path, time_path


def all_pairs_shortest(pathways):
    """Run Dijkstra from every building. Useful lookup for the solver."""
    adj = build_graph(pathways)
    return {src: dijkstra(adj, src) for src in adj}


def print_routes(results):
    """Print the best route for every pair of buildings (each pair once)."""
    order = sorted(results.keys())
    print("\nBest route for each pair (shortest distance)")
    print(f"{'From':<6}{'To':<6}{'Dist (m)':>9}{'Time (s)':>10}   Route")
    print("-" * 60)
    for i, s in enumerate(order):
        dist, path, time_path = results[s]
        for d in order[i + 1:]:
            route = " -> ".join(path[d])
            print(f"{s:<6}{d:<6}{dist[d]:>9.0f}{time_path[d]:>10.0f}   {route}")


def solve(buildings, pathways, deliveries, depot="B16", capacity=15.0):
    """
    Produce a delivery plan (see SOLUTION FORMAT above).

    Before writing code, discuss as a team:
      - What do you need to look up repeatedly, and how will you store
        it so those lookups are fast?
      - How will you decide what order to visit things in?
      - How will you decide which requests to carry together, given
        the capacity limit?
    """
    # Precompute shortest paths between every pair of buildings.
    # results[src] = (dist, path, time_path)
    results = all_pairs_shortest(pathways)

    # TODO: use results + deliveries + capacity to build a smarter plan
    # (batch requests, reorder stops). Baseline: one request at a time.
    plan = [{"building": depot, "action": "start"}]
    for req in deliveries:
        rid = req["request_id"]
        plan.append({"building": req["pickup"], "action": "pickup", "request_id": rid})
        plan.append({"building": req["dropoff"], "action": "dropoff", "request_id": rid})
    plan.append({"building": depot, "action": "end"})
    return plan


if __name__ == "__main__":
    buildings, pathways, deliveries = load_data()

    # Show all-pairs shortest routes (distance + time + path)
    results = all_pairs_shortest(pathways)
    print_routes(results)

    for label, n in [("SMALL", 8), ("MEDIUM", 16), ("FULL", 30)]:
        subset = deliveries[:n]
        plan = solve(buildings, pathways, subset)
        result = check_solution(buildings, pathways, subset, plan)
        print_report(result, label=f"{label} (n={n})")
