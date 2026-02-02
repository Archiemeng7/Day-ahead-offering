# ro/scripts/dat_scripts/offering_no_network_gen_eval_table.py

import argparse
import itertools


def _enum_list(x):
    """
    Helper: if user passes scalar -> list[scalar], if passes list -> list,
    if passes "none" -> [None]
    """
    if x is None:
        return [None]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_run_cmds(args):
    """
    Generate list of commands for running ml-ccg on offering_no_network.
    No per-instance enumeration like KP; optionally enumerate Gamma / lambda_rt / price file variants.
    """
    cmd_list = []

    enum_gamma = _enum_list(args.onn_gamma)
    enum_lambda_rt = _enum_list(args.onn_lambda_rt)
    enum_price_path = _enum_list(args.onn_price_mat_path)

    for gamma, lam_rt, price_path in itertools.product(enum_gamma, enum_lambda_rt, enum_price_path):

        # base cmd
        if args.opt_type == "adversarial":
            cmd = "python -m ro.scripts.05_run_ml_ccg "
        elif args.opt_type == "pga":
            cmd = "python -m ro.scripts.05_run_ml_ccg_pga "
        else:
            raise ValueError(f"opt_type must be adversarial or pga, got {args.opt_type}")

        # common args
        cmd += f"--problem {args.problem} "
        cmd += f"--model_type {args.model_type} "

        # optimization params
        cmd += f"--mp_gap {args.mp_gap} "
        cmd += f"--adversarial_gap {args.adversarial_gap} "
        cmd += f"--mp_time {args.mp_time} "
        cmd += f"--adversarial_time {args.adversarial_time} "
        cmd += f"--mp_inc_time {args.mp_inc_time} "
        cmd += f"--adversarial_inc_time {args.adversarial_inc_time} "
        cmd += f"--mp_focus {args.mp_focus} "
        cmd += f"--adversarial_focus {args.adversarial_focus} "
        cmd += f"--opt_type {args.opt_type} "

        # offering_no_network specific override args (ONLY if user requested)
        # IMPORTANT: these flags must be supported by your 05_run_ml_ccg / dm/params usage.
        if gamma is not None:
            cmd += f"--onn_gamma {gamma} "
        if lam_rt is not None:
            cmd += f"--onn_lambda_rt {lam_rt} "
        if price_path is not None:
            cmd += f'--onn_price_mat_path "{price_path}" '

        # pga-specific
        if args.opt_type == "pga":
            cmd += f"--pga_samples {args.pga_samples} "
            cmd += f"--pga_epochs {args.pga_epochs} "
            cmd += f"--n_procs {args.pga_n_procs} "

        cmd_list.append(cmd.strip())

    return cmd_list


def get_eval_cmds(args):
    """
    Generate list of commands for evaluating ml-ccg objective.
    """
    cmd_list = []

    enum_gamma = _enum_list(args.onn_gamma)
    enum_lambda_rt = _enum_list(args.onn_lambda_rt)
    enum_price_path = _enum_list(args.onn_price_mat_path)

    for gamma, lam_rt, price_path in itertools.product(enum_gamma, enum_lambda_rt, enum_price_path):

        cmd = "python -m ro.scripts.06_eval_ml_ccg_obj "
        cmd += f"--problem {args.problem} "
        cmd += f"--model_type {args.model_type} "
        cmd += f"--opt_type {args.opt_type} "

        # offering_no_network specific overrides (ONLY if user requested)
        if gamma is not None:
            cmd += f"--onn_gamma {gamma} "
        if lam_rt is not None:
            cmd += f"--onn_lambda_rt {lam_rt} "
        if price_path is not None:
            cmd += f'--onn_price_mat_path "{price_path}" '

        cmd_list.append(cmd.strip())

    return cmd_list


def main(args):
    if args.cmd_type == "run":
        cmd_list = get_run_cmds(args)
    elif args.cmd_type == "eval":
        cmd_list = get_eval_cmds(args)
    else:
        raise ValueError("cmd_type must be one of ['run','eval']")

    if len(cmd_list) == 0:
        raise RuntimeError("No commands generated. Check your enumerations/arguments.")

    # write to text file
    with open(args.file_name, "w", encoding="utf-8") as f:
        for i, cmd in enumerate(cmd_list, start=1):
            f.write(f"{i} {cmd}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Writes commands to run/evaluate ml-ccg for offering_no_network."
    )

    parser.add_argument("--problem", type=str, default="offering_no_network")
    parser.add_argument("--model_type", type=str, default="set_encoder")

    parser.add_argument("--cmd_type", type=str, default="run", choices=["run", "eval"])

    # ml-opt choices (same as kp_gen_eval_table.py)
    parser.add_argument("--mp_gap", type=float, default=1e-4)
    parser.add_argument("--adversarial_gap", type=float, default=1e-4)
    parser.add_argument("--mp_time", type=float, default=10800)
    parser.add_argument("--adversarial_time", type=float, default=10800)
    parser.add_argument("--mp_inc_time", type=float, default=180)
    parser.add_argument("--adversarial_inc_time", type=float, default=180)
    parser.add_argument("--mp_focus", type=int, default=0)
    parser.add_argument("--adversarial_focus", type=int, default=0)

    # optimize type (pga or adversarial)
    parser.add_argument("--opt_type", type=str, default="adversarial")

    # pga specific parameters
    parser.add_argument("--pga_samples", type=int, default=20)
    parser.add_argument("--pga_epochs", type=int, default=100)
    parser.add_argument("--pga_n_procs", type=int, default=4)

    # offering_no_network enumerations (optional)
    # If you provide multiple values, it will generate Cartesian product combinations.
    parser.add_argument("--onn_gamma", type=float, nargs="*", default=None,
                        help="Optionally override Gamma; pass multiple to enumerate.")
    parser.add_argument("--onn_lambda_rt", type=float, nargs="*", default=None,
                        help="Optionally override RT price (constant); pass multiple to enumerate.")
    parser.add_argument("--onn_price_mat_path", type=str, nargs="*", default=None,
                        help="Optionally override .mat path; pass multiple to enumerate.")

    # file name
    parser.add_argument("--file_name", type=str, default="table.dat")

    args = parser.parse_args()
    main(args)
