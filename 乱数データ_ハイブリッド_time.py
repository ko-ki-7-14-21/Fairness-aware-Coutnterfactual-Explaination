import argparse
import csv
from typing import Optional, Sequence, Tuple

import numpy as np

from 乱数データ_2段階 import (
    ALPHA_BETA_LIST_DEFAULT,
    build_c_bar_1,
    build_c_bar_2,
    build_h_vector,
    generate_random_matrices,
    solve_dd_face,
    solve_original_LP,
    status_label,
)
from 乱数データ_ハイブリット import solve_hybrid_face
from 乱数データ_双線形 import compute_metrics


K_FIXED = 8
M_FIXED = 20
N_FIXED = 20
SEED_LIST_DEFAULT = [0, 1, 2, 3, 4]
LIMIT_PAIRS_DEFAULT: Sequence[Tuple[int, int]] = [
    (5, 1),
    (10, 2),
    (20, 4),
    (40, 8),
]
HYBRID_LIMIT_DEFAULT = 5


def run_time_experiments(
    output_csv: str,
    limit_pairs: Sequence[Tuple[int, int]] = LIMIT_PAIRS_DEFAULT,
    hybrid_limit: int = HYBRID_LIMIT_DEFAULT,
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
                "hybrid_limit",
                "objective",
                "G",
                "price_term",
                "fairness_term1",
                "fairness_term2",
                "dd1_time",
                "dd2_time",
                "hybrid_time",
                "total_time",
                "dd1_status",
                "dd2_status",
                "hybrid_status",
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
                        _,
                        _,
                        _,
                        dd1_runtime,
                        _,
                        _,
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
                        mip_focus=3,
                        heuristics=0.5,
                    )
                    if X_1 is None or Y_1 is None or c_1 is None:
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
                                hybrid_limit,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                dd1_runtime,
                                np.nan,
                                np.nan,
                                np.nan,
                                status_label(dd1_status),
                                "",
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
                        _,
                        _,
                        _,
                        dd2_runtime,
                        _,
                        _,
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
                        mip_focus=3,
                        heuristics=0.5,
                        start_solution=(X_1, Y_1, c_1),
                    )
                    if X_2 is None or Y_2 is None or c_2 is None:
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
                                hybrid_limit,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                dd1_runtime,
                                dd2_runtime,
                                np.nan,
                                np.nan,
                                status_label(dd1_status),
                                status_label(dd2_status),
                                "",
                            ]
                        )
                        continue

                    (
                        X_h,
                        Y_h,
                        c_h,
                        _,
                        _,
                        _,
                        h_runtime,
                        _,
                        h_status,
                        _,
                    ) = solve_hybrid_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        X_2,
                        alpha,
                        beta,
                        time_limit=hybrid_limit,
                        start_solution=(Y_2, c_2),
                    )
                    if X_h is None or Y_h is None or c_h is None:
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
                                hybrid_limit,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                dd1_runtime,
                                dd2_runtime,
                                h_runtime,
                                dd1_runtime + dd2_runtime + h_runtime,
                                status_label(dd1_status),
                                status_label(dd2_status),
                                status_label(h_status),
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
                    ) = compute_metrics(c_h, X_h, c_hat, x_hat, alpha, beta)
                    total_time = dd1_runtime + dd2_runtime + h_runtime
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
                            hybrid_limit,
                            objective_val,
                            G,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                            dd1_runtime,
                            dd2_runtime,
                            h_runtime,
                            total_time,
                            status_label(dd1_status),
                            status_label(dd2_status),
                            status_label(h_status),
                        ]
                    )
                    print(
                        "Done "
                        f"seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                        f"limit1={limit_1}, limit2={limit_2}, hlimit={hybrid_limit}, "
                        f"dd1={status_label(dd1_status)}, dd2={status_label(dd2_status)}, "
                        f"hybrid={status_label(h_status)}, t={total_time:.2f}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-size hybrid experiments with varying time limits."
    )
    parser.add_argument("--output", default="random_hybrid_time.csv", help="Output CSV path.")
    parser.add_argument(
        "--alpha-beta",
        action="append",
        help="Run only specific alpha,beta pairs in 'alpha,beta' format. Can be repeated.",
    )
    parser.add_argument("--limit-1", type=int, help="Override first-stage time limit.")
    parser.add_argument("--limit-2", type=int, help="Override second-stage time limit.")
    parser.add_argument("--hybrid-limit", type=int, default=HYBRID_LIMIT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.limit_1 is None) != (args.limit_2 is None):
        raise ValueError("--limit-1 and --limit-2 must be set together.")
    limit_pairs = (
        [(args.limit_1, args.limit_2)] if args.limit_1 is not None else LIMIT_PAIRS_DEFAULT
    )
    alpha_beta_list = parse_alpha_beta_list(args.alpha_beta) or ALPHA_BETA_LIST_DEFAULT
    run_time_experiments(
        output_csv=args.output,
        limit_pairs=limit_pairs,
        hybrid_limit=args.hybrid_limit,
        alpha_beta_list=alpha_beta_list,
    )


if __name__ == "__main__":
    main()
