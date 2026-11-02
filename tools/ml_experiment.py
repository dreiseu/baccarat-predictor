"""Can a trained model predict the next baccarat hand?

Simulates real 8-deck shoes (punto banco third-card rules, face-up burn card
plus face-down burns, cut card), builds features from everything a player can
see before a hand, trains several standard models, and tests them on shoes
they never saw. The bar to beat is "always bet Banker".

Features per hand:
  - last 12 results (Banker / Player / Tie), the patterns players track
  - current streak (signed length), Banker and Player counts so far this shoe
  - exact count of every card value left in the shoe, and cards remaining

Usage (needs numpy and scikit-learn):
  python3 tools/ml_experiment.py [train_shoes] [test_shoes]
"""
import math
import random
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HISTORY = 12
DECKS = 8


def banker_draws(bt, p3):
    return (bt <= 2 or (bt == 3 and p3 != 8) or (bt == 4 and 2 <= p3 <= 7)
            or (bt == 5 and 4 <= p3 <= 7) or (bt == 6 and p3 in (6, 7)))


def simulate_shoe(rng):
    """Yield (features, result, banker_total) for every hand in one shoe."""
    shoe = [0] * (16 * DECKS) + [v for v in range(1, 10) for _ in range(4 * DECKS)]
    rng.shuffle(shoe)
    counts = [16 * DECKS] + [4 * DECKS] * 9

    def take(known=True):
        v = shoe.pop()
        if known:
            counts[v] -= 1
        return v

    burn = take()
    for _ in range(10 if burn == 0 else burn):
        take(known=False)  # face-down burns are unknown to the player
    cut = rng.randint(14, 30)
    results = []
    while len(shoe) > cut:
        hist = [{'B': 1, 'P': -1, 'T': 0}[r] for r in results[-HISTORY:]]
        mask = [1] * len(hist)
        hist = [0] * (HISTORY - len(hist)) + hist
        mask = [0] * (HISTORY - len(mask)) + mask
        decided = [r for r in results if r != 'T']
        streak = 0
        if decided:
            last = decided[-1]
            for r in reversed(decided):
                if r != last:
                    break
                streak += 1
            streak *= 1 if last == 'B' else -1
        known_left = sum(counts)
        feats = (hist + mask + [streak, results.count('B'), results.count('P'), results.count('T'), len(shoe)]
                 + [c / known_left for c in counts])

        p1, b1, p2, b2 = take(), take(), take(), take()
        pt, bt = (p1 + p2) % 10, (b1 + b2) % 10
        if pt < 8 and bt < 8:
            if pt <= 5:
                p3 = take()
                pt = (pt + p3) % 10
                if banker_draws(bt, p3):
                    bt = (bt + take()) % 10
            elif bt <= 5:
                bt = (bt + take()) % 10
        r = 'P' if pt > bt else 'B' if bt > pt else 'T'
        results.append(r)
        yield feats, r, bt


def build(n_shoes, seed):
    rng = random.Random(seed)
    X, y, b6 = [], [], []
    for _ in range(n_shoes):
        for feats, r, bt in simulate_shoe(rng):
            X.append(feats)
            y.append(r)
            b6.append(r == 'B' and bt == 6)
    return np.array(X, dtype=np.float32), np.array(y), np.array(b6)


def ci95(acc, n):
    half = 1.96 * math.sqrt(acc * (1 - acc) / n)
    return f'{acc * 100:.2f}% (95% range {100 * (acc - half):.2f}–{100 * (acc + half):.2f}%)'


def bet_result(picks, y, b6):
    """Average result per 100 units betting each pick at No Commission payouts."""
    win = np.where(picks == 'P', y == 'P', y == 'B')
    lose = np.where(picks == 'P', y == 'B', y == 'P')
    pay = np.where((picks == 'B') & b6, 0.5, 1.0)
    return 100 * (win * pay - lose).mean()


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    t0 = time.time()
    Xtr, ytr, _ = build(n_train, seed=1)
    Xte, yte, b6te = build(n_test, seed=2)
    print(f'Simulated {n_train} training shoes ({len(ytr):,} hands) and {n_test} test shoes '
          f'({len(yte):,} hands) in {time.time() - t0:.0f}s\n')

    # Ties push Banker and Player bets, so judge picks on decided hands only.
    tr, te = ytr != 'T', yte != 'T'
    Xd, yd, Xt, yt = Xtr[tr], ytr[tr], Xte[te], yte[te]
    n = len(yt)

    print(f'{"Model":<34}{"Training accuracy":>19}   Test accuracy (unseen shoes)')
    base_tr = (yd == 'B').mean()
    base_te = (yt == 'B').mean()
    print(f'{"Always bet Banker (baseline)":<34}{base_tr * 100:>18.2f}%   {ci95(base_te, n)}')

    models = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        'Gradient boosting': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        'Random forest (deep trees)': RandomForestClassifier(n_estimators=40, min_samples_leaf=1, max_samples=0.25,
                                                             n_jobs=-1, random_state=0),
        'Neural network (64-32)': make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=40,
                                                                                early_stopping=True, random_state=0)),
    }
    bets = {'Always bet Banker (baseline)': bet_result(np.full(len(yte), 'B'), yte, b6te),
            'Always bet Player': bet_result(np.full(len(yte), 'P'), yte, b6te)}
    for name, model in models.items():
        t = time.time()
        model.fit(Xd, yd)
        acc_tr = (model.predict(Xd) == yd).mean()
        acc_te = (model.predict(Xt) == yt).mean()
        picks_all = model.predict(Xte)
        bets[name] = bet_result(picks_all, yte, b6te)
        share_b = (model.predict(Xt) == 'B').mean() * 100
        print(f'{name:<34}{acc_tr * 100:>18.2f}%   {ci95(acc_te, n)}   '
              f'[picks Banker {share_b:.0f}% of the time, {time.time() - t:.0f}s]')

    print('\nBetting 100 on every pick at No Commission payouts (all test hands, ties push):')
    for name, v in bets.items():
        print(f'  {name:<34}{v:+.2f} per 100 bet')


if __name__ == '__main__':
    main()
