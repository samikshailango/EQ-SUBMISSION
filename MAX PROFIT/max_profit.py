from dataclasses import dataclass

@dataclass(frozen=True)
class Building:
    code: str      
    name: str
    cost: int      
    rate: int 

BUILDINGS = [
    Building("T", "Theatre", cost=5, rate=1500),
    Building("P", "Pub", cost=4, rate=1000),
    Building("C", "Commercial Park", cost=10, rate=2000),
]

def max_profit(n: int):
    if n < 0:
        raise ValueError("n must be non-negative")
    f = [0] * (n + 1)
    choice = [None] * (n + 1)
    for t in range(1, n + 1):
        best = 0
        best_b = None
        for b in BUILDINGS:
            if b.cost <= t:
                candidate = b.rate * (t - b.cost) + f[t - b.cost]
                if candidate > best:
                    best = candidate
                    best_b = b
        f[t] = best
        choice[t] = best_b

    counts = {"T": 0, "P": 0, "C": 0}
    sequence = []
    t = n
    while t > 0 and choice[t] is not None:
        b = choice[t]
        counts[b.code] += 1
        sequence.append(b.code)
        t -= b.cost
    return f[n], counts, sequence

def format_output(n: int):
    earnings, counts, sequence = max_profit(n)
    mix = f"T: {counts['T']} P: {counts['P']} C: {counts['C']}"
    return earnings, mix, sequence

def get_n_from_user():
    while True:
        raw = input("Enter n (total units of time available): ").strip()
        try:
            n = int(raw)
            if n < 0:
                print("Please enter a non-negative integer.")
                continue
            return n
        except ValueError:
            print("That's not a valid integer, try again.")

if __name__ == "__main__":
    n = get_n_from_user()
    earnings, mix, seq = format_output(n)
    print()
    print(f"n = {n}")
    print(f"Max Earnings: ${earnings}")
    print(f"Building mix -> {mix}")
    print(f"Build order (start sequence): {seq}")