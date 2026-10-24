def wire(data):
    x, y = (0, 0)
    for dir, step in [(x[0], int(x[1:])) for x in data.split(",")]:
        for _ in range(step):
            x += -1 if dir == "L" else 1 if dir == "R" else 0
            y += -1 if dir == "D" else 1 if dir == "U" else 0
            yield x, y


def aoc(data):
    wires = [set(wire(line)) for line in data.split("\n")]
    dist = set()
    for cord in wires[0] | wires[1]:
        if cord in wires[0] and cord in wires[1]:
            dist.add(abs(cord[0]) + abs(cord[1]))
    return min(dist)