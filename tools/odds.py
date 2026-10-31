"""Exact next-hand odds given cards already dealt from an 8-deck shoe.
Usage: python3 odds.py [--no-commission] "A 5 K 9 3 Q ..."   (ranks: A 2-9 10 J Q K; 10/J/Q/K all count 0)"""
import sys
from functools import lru_cache
DECKS = 8
def rank_val(t):
    t = t.upper()
    if t in ('A', '1'): return 1
    if t in ('10', 'J', 'Q', 'K', 'T', '0'): return 0
    return int(t)
def odds(dealt):
    counts = [16 * DECKS] + [4 * DECKS] * 9            # value 0 has 16 ranks/deck
    for c in dealt:
        v = rank_val(c); counts[v] -= 1
        if counts[v] < 0: raise SystemExit(f"Too many cards of value {v}")
    res = {'B': 0.0, 'P': 0.0, 'T': 0.0, 'B6': 0.0}
    def draws(cnt):
        n = sum(cnt)
        for v in range(10):
            if cnt[v]:
                p = cnt[v] / n; cnt[v] -= 1
                yield v, p
                cnt[v] += 1
    def settle(pt, bt, w):
        res['P' if pt > bt else 'B' if bt > pt else 'T'] += w
        if bt > pt and bt == 6: res['B6'] += w
    cnt = counts[:]
    for p1, w1 in draws(cnt):
        for b1, w2 in draws(cnt):
            for p2, w3 in draws(cnt):
                for b2, w4 in draws(cnt):
                    w = w1 * w2 * w3 * w4
                    pt, bt = (p1 + p2) % 10, (b1 + b2) % 10
                    if pt >= 8 or bt >= 8: settle(pt, bt, w); continue
                    if pt <= 5:
                        for p3, w5 in draws(cnt):
                            npt = (pt + p3) % 10
                            draw = bt <= 2 or (bt == 3 and p3 != 8) or (bt == 4 and 2 <= p3 <= 7) \
                                or (bt == 5 and 4 <= p3 <= 7) or (bt == 6 and p3 in (6, 7))
                            if draw:
                                for b3, w6 in draws(cnt): settle(npt, (bt + b3) % 10, w * w5 * w6)
                            else: settle(npt, bt, w * w5)
                    elif bt <= 5:
                        for b3, w6 in draws(cnt): settle(pt, (bt + b3) % 10, w * w6)
                    else: settle(pt, bt, w)
    return res
if __name__ == '__main__':
    args = sys.argv[1:]
    no_commission = '--no-commission' in args
    args = [a for a in args if a != '--no-commission']
    dealt = args[0].replace(',', ' ').split() if args else []
    r = odds(dealt)
    # No Commission: Banker pays 1:1, except a Banker win on 6 pays 1:2.
    evB = (r['B'] - r['B6']) + r['B6'] * 0.5 - r['P'] if no_commission else r['B'] * 0.95 - r['P']; evP = r['P'] - r['B']; evT = r['T'] * 8 - (1 - r['T'])
    print(f"Cards removed: {len(dealt)}")
    print(f"Banker {r['B']*100:.3f}%  Player {r['P']*100:.3f}%  Tie {r['T']*100:.3f}%")
    if no_commission: print(f"Banker wins with 6: {r['B6']*100:.3f}%")
    print(f"EV per 1 unit -> Banker {evB:+.4f}  Player {evP:+.4f}  Tie(8:1) {evT:+.4f}")
