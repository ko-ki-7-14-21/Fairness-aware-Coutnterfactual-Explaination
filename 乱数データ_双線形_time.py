import argparse
import csv
from typing import Optional, Sequence, Tuple

import numpy as np

from 乱数データ_双線形 import (
    ALPHA_BETA_LIST_DEFAULT,
    build_h_vector,
    compute_metrics,
    generate_random_matrices,
    solve_FACE,
    solve_original_LP,
    status_label,
)

K_FIXED = 8
M_FIXED = 20
N_FIXED = 20
SEED_LIST_DEFAULT = [0, 1, 2, 3, 4]
TIME_LIMITS_DEFAULT: Sequence[int] = [6, 12, 24, 48]

def run_time_experiments(
    output_csv: str,
    time_limits: Sequence[int] = TIME_LIMITS_DEFAULT,
    seed_list: Sequence[int] = SEED_LIST_DEFAULT,
    alpha_beta_list: Sequence[Tuple[float, float]] = ALPHA_BETA_LIST_DEFAULT,
) -> None:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "K",
                "m",
                "n",
                "alpha",
                "beta",
                "limit",
                "目的関数値",
                "G",
                "priceterm",
                "fairness_term_1",
                "fairness_term_2",
                "計算時間",
                "status",
            ]
        )

        K = K_FIXED
        m = M_FIXED
        n = N_FIXED

        for seed_value in seed_list:
            rng = np.random.default_rng(np.random.SeedSequence([seed_value, K, n, m]))
            A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
            x_hat = solve_original_LP(A, b_vecs, c_hat)
            h = build_h_vector(rng, x_hat, c_hat)

            for alpha, beta in alpha_beta_list:
                if alpha + beta > 1:
                    continue
                for time_limit in time_limits:
                    (
                        X_sol,
                        c_sol,
                        v_sol,
                        v_ave_sol,
                        v_max_sol,
                        runtime,
                        _,
                        status,
                    ) = solve_FACE(A, b_vecs, c_hat, h, x_hat, alpha, beta, time_limit=time_limit)

                    if (
                        X_sol is None
                        or c_sol is None
                        or v_sol is None
                        or v_ave_sol is None
                        or v_max_sol is None
                    ):
                        writer.writerow(
                            [
                                seed_value,
                                K,
                                m,
                                n,
                                alpha,
                                beta,
                                time_limit,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                runtime,
                                status_label(status),
                            ]
                        )
                        continue

                    (
                        _,
                        _,
                        _,
                        price_term,
                        fairness_term1,
                        fairness_term2,
                        G,
                        objective_val,
                    ) = compute_metrics(c_sol, X_sol, c_hat, x_hat, alpha, beta)
                    writer.writerow(
                        [
                            seed_value,
                            K,
                            m,
                            n,
                            alpha,
                            beta,
                            time_limit,
                            objective_val,
                            G,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                            runtime,
                            status_label(status),
                        ]
                    )
                    print(
                        f"Done seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                        f"limit={time_limit}, status={status_label(status)}, t={runtime:.2f}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-size FACE experiments with varying time limits."
    )
    parser.add_argument("--output", default="face_random_time.csv", help="Output CSV path.")
    parser.add_argument(
        "--alpha-beta",
        action="append",
        help="Run only specific alpha,beta pairs in 'alpha,beta' format. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        action="append",
        help="Override time limits (repeatable). If omitted, uses defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_beta_list = parse_alpha_beta_list(args.alpha_beta) or ALPHA_BETA_LIST_DEFAULT
    time_limits = args.limit or TIME_LIMITS_DEFAULT
    run_time_experiments(
        output_csv=args.output,
        time_limits=time_limits,
        alpha_beta_list=alpha_beta_list,
    )


def parse_alpha_beta_list(
    alpha_beta: Optional[Sequence[str]],
) -> Optional[Sequence[Tuple[float, float]]]:
    if not alpha_beta:
        return None
    parsed: list[Tuple[float, float]] = []
    for item in alpha_beta:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"--alpha-beta must be 'alpha,beta' format: {item}")
        alpha, beta = (float(p) for p in parts)
        parsed.append((alpha, beta))
    return parsed


if __name__ == "__main__":
    main()
