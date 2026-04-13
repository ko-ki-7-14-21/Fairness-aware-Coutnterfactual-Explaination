import argparse
import csv
from typing import Sequence, Tuple

import numpy as np

from 乱数データ_2段階 import (
    ALPHA_BETA_LIST_DEFAULT,
    build_c_bar_1,
    build_c_bar_2,
    build_h_vector,
    compute_metrics,
    generate_random_matrices,
    solve_dd_face,
    solve_original_LP,
    status_label,
)

K_FIXED = 8
M_FIXED = 20
N_FIXED = 20
SEED_LIST_DEFAULT = [0, 1, 2, 3, 4]
LIMIT_PAIRS_DEFAULT: Sequence[Tuple[int, int]] = [
    (10, 10),
    (20, 20),
    (30, 30),
    (40, 40),
    (50, 50),
    (60, 60),
]


def run_time_experiments(
    output_csv: str,
    limit_pairs: Sequence[Tuple[int, int]] = LIMIT_PAIRS_DEFAULT,
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
                "limit_1",
                "limit_2",
                "目的関数値",
                "G",
                "priceterm",
                "fairness_term_1",
                "fairness_term_2",
                "1段階目の計算時間",
                "2段階目の計算時間",
                "2段階の合計計算時間",
                "1段階目のstatus",
                "2段階目のstatus",
            ]
        )

        K = K_FIXED
        m = M_FIXED
        n = N_FIXED
        for seed_value in seed_list:
            rng = np.random.default_rng(np.random.SeedSequence([seed_value, K, n, m]))
            A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
            x_hat = solve_original_LP(A, b_vecs, c_hat)
            h = build_h_vector(rng, x_hat)

            for alpha, beta in alpha_beta_list:
                c_bar_1 = build_c_bar_1(c_hat)
                for limit_1, limit_2 in limit_pairs:
                    (
                        X_1,
                        Y_1,
                        c_1,
                        _,
                        v_1,
                        _,
                        _,
                        dd1_runtime,
                        dd1_first_feas,
                        dd1_gap,
                        dd1_status,
                    ) = solve_dd_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        c_bar_1,
                        alpha,
                        beta,
                        limit_1,
                        numeric_focus=2,
                        mip_focus=1,
                        heuristics=1.0,
                    )
                    print(
                        "D-D-FACEP_1 Done "
                        f"seed={seed_value}, K={K}, n={n}, m={m}, "
                        f"alpha={alpha}, beta={beta}, limit_1={limit_1}, "
                        f"status={status_label(dd1_status)}, t={dd1_runtime:.2f}, "
                        f"first_feas={dd1_first_feas}"
                    )

                    if X_1 is None or Y_1 is None or c_1 is None or v_1 is None:
                        writer.writerow(
                            [
                                seed_value,
                                K,
                                m,
                                n,
                                alpha,
                                beta,
                                limit_1,
                                limit_2,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                dd1_runtime,
                                np.nan,
                                np.nan,
                                status_label(dd1_status),
                                "",
                            ]
                        )
                        continue

                    c_bar_2 = build_c_bar_2(c_1, c_hat)
                    (
                        X_2,
                        Y_2,
                        c_2,
                        _,
                        v_2,
                        _,
                        _,
                        dd2_runtime,
                        dd2_first_feas,
                        dd2_gap,
                        dd2_status,
                    ) = solve_dd_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        c_bar_2,
                        alpha,
                        beta,
                        limit_2,
                        numeric_focus=2,
                        mip_focus=0,
                        heuristics=0.0,
                        start_solution=(X_1, Y_1, c_1),
                    )
                    print(
                        "D-D-FACEP_2 Done "
                        f"seed={seed_value}, K={K}, n={n}, m={m}, "
                        f"alpha={alpha}, beta={beta}, limit_2={limit_2}, "
                        f"status={status_label(dd2_status)}, t={dd2_runtime:.2f}, "
                        f"first_feas={dd2_first_feas}"
                    )

                    if X_2 is None or c_2 is None:
                        writer.writerow(
                            [
                                seed_value,
                                K,
                                m,
                                n,
                                alpha,
                                beta,
                                limit_1,
                                limit_2,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                dd1_runtime,
                                dd2_runtime,
                                dd1_runtime + dd2_runtime,
                                status_label(dd1_status),
                                status_label(dd2_status),
                            ]
                        )
                        continue

                    (_, price_term, fairness_term1, fairness_term2, G, objective_val) = compute_metrics(
                        c_2, X_2, c_hat, x_hat, alpha, beta
                    )
                    total_time = dd1_runtime + dd2_runtime
                    writer.writerow(
                        [
                            seed_value,
                            K,
                            m,
                            n,
                            alpha,
                            beta,
                            limit_1,
                            limit_2,
                            objective_val,
                            G,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                            dd1_runtime,
                            dd2_runtime,
                            total_time,
                            status_label(dd1_status),
                            status_label(dd2_status),
                        ]
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-size D-D-FACE experiments with varying time limits."
    )
    parser.add_argument("--output", default="random_D_D_time.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_time_experiments(output_csv=args.output)


if __name__ == "__main__":
    main()
