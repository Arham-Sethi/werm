import random

def simulate(trials=500000):
    wins = 0
    for _ in range(trials):
        W, R = 2000, 10000
        while W > 0 and R > 0:
            if W >= R / 5:
                bet = 1
            else:
                # rescue bet: sized to land back above W = R/5 on a win
                # after bet b: new W = W+b, new R = R-b
                # need W+b >= (R-b)/5 => 5W+5b >= R-b => 6b >= R-5W => b >= (R-5W)/6
                bet = max(1, (R - 5*W) // 6 + 1)
                bet = min(bet, W)
            R -= bet
            if R <= 0:
                wins += 1
                break
            W += bet if random.random() < 0.4 else -bet
    return wins / trials

result = simulate()
print(f"Win probability: {result:.4f} ({result*100:.2f}%)")