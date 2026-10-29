"""Tableau Q2b : BCR, EKF et RMSE (ecarts-types) aux instants choisis, cas A, B, C.

Reutilise experiments.py sans le modifier. Memes graines que main().
Usage : python tableau_q2b.py [--runs 500] [--mc 15000] [--steps 30] [--k 5 30]
"""
import argparse
import numpy as np

import experiments as e


def stats(truth, est, cov, bcr, k):
    """Ecarts-types (s, v) a l'instant k : BCR, EKF annonce, RMSE reel."""
    bcr_std = np.sqrt(np.diag(bcr[k]))
    ekf_std = np.sqrt(np.mean(cov[:, k, [0, 1], [0, 1]], axis=0))
    rmse = np.sqrt(np.mean((est[:, k] - truth[:, k])**2, axis=0))
    return bcr_std, ekf_std, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--mc', type=int, default=15000)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--k', type=int, nargs='+', default=[5, 30])
    args = parser.parse_args()

    zero = np.zeros_like(e.Q)

    trueA, yA = e.simulate(args.runs, args.steps, zero, seed=2026)
    estA, covA = e.run_ekf(trueA, yA, zero)
    bcrA = e.information_curve(args.steps, zero, args.mc, seed=2027)

    trueBC, yBC = e.simulate(args.runs, args.steps, e.Q, seed=3030)
    estBC, covBC = e.run_ekf(trueBC, yBC, e.Q)
    bcrB = e.information_curve(args.steps, zero, args.mc, seed=2027)
    bcrC = e.information_curve(args.steps, e.Q, args.mc, seed=3031)

    cases = [('A', trueA, estA, covA, bcrA),
             ('B', trueBC, estBC, covBC, bcrB),
             ('C', trueBC, estBC, covBC, bcrC)]

    print('Ecarts-types (s en m, v en m/s)')
    print(f'{"cas":<4}{"k":>3}   {"BCR s":>8}{"BCR v":>8}   {"EKF s":>8}{"EKF v":>8}   {"RMSE s":>8}{"RMSE v":>8}')
    for name, truth, est, cov, bcr in cases:
        for k in args.k:
            b, p, r = stats(truth, est, cov, bcr, k)
            print(f'{name:<4}{k:>3}   {b[0]:8.3f}{b[1]:8.3f}   {p[0]:8.3f}{p[1]:8.3f}   {r[0]:8.3f}{r[1]:8.3f}')


if __name__ == '__main__':
    main()
