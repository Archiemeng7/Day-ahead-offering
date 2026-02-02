# # ro/utils/offering_no_network.py
# import os

# def get_path(data_path, cfg, which, suffix=""):
#     """
#     Compatible with scripts/00_init_directories.py which calls:
#         get_path(cfg.data_path, cfg, "", suffix="")
#     and with DataManager which calls:
#         get_path(cfg.data_path, cfg, "problem"/"ml_data"/"test_instances/")
#     """

#     # strip leading "./" so split("/") in scripts works as expected
#     if isinstance(data_path, str) and data_path.startswith("./"):
#         data_path = data_path[2:]
#     data_path = data_path.rstrip("/")

#     base = os.path.join(data_path, "offering_no_network")
#     os.makedirs(base, exist_ok=True)

#     # when which is empty, return a 3-segment path "data/offering_no_network/problem.pkl"
#     if which in ("", "problem"):
#         return os.path.join(base, "problem.pkl")

#     if which == "ml_data":
#         # suffix may already include ".pkl"
#         if suffix:
#             if suffix.endswith(".pkl"):
#                 return os.path.join(base, f"ml_data{suffix}")
#             return os.path.join(base, f"ml_data{suffix}.pkl")
#         return os.path.join(base, "ml_data.pkl")

#     if which == "test_instances/":
#         inst_dir = os.path.join(base, "test_instances")
#         os.makedirs(inst_dir, exist_ok=True)
#         return inst_dir + os.sep

#     # directory init scripts may pass other tokens; be permissive
#     return os.path.join(base, "problem.pkl")




# ro/utils/offering_no_network.py
import os


def _num_token(v):
    """
    Convert numbers to filename-safe tokens:
      12.0 -> "12"
      0.95 -> "0p95"
      -1.2 -> "m1p2"
    """
    try:
        fv = float(v)
    except Exception:
        return str(v)

    if abs(fv - round(fv)) <= 1e-12:
        return str(int(round(fv)))

    s = f"{fv:.12g}"  # compact, stable
    s = s.replace("-", "m").replace(".", "p")
    return s


def _cfg_tag(cfg):
    """
    Make a deterministic tag similar in spirit to cb/kp naming,
    so later training code can reliably locate the right files.
    """
    T = getattr(cfg, "T", 24)
    S = getattr(cfg, "n_scenarios", getattr(cfg, "scenarios", "S"))
    Gamma = getattr(cfg, "Gamma", "G")
    rt = getattr(cfg, "lambda_rt_value", getattr(cfg, "rt_price_const", "rt"))

    nsi = getattr(cfg, "n_samples_inst", "nsi")
    nsf = getattr(cfg, "n_samples_fs", "nsf")
    nsu = getattr(cfg, "n_samples_per_fs", "nsu")
    sd = getattr(cfg, "seed", "sd")

    ocs = getattr(cfg, "offer_curve_sampling", "ocs")

    tag = (
        f"T{_num_token(T)}"
        f"_S{_num_token(S)}"
        f"_G{_num_token(Gamma)}"
        f"_rt{_num_token(rt)}"
        f"_ocs-{str(ocs)}"
        f"_nsi{_num_token(nsi)}"
        f"_nsf{_num_token(nsf)}"
        f"_nsu{_num_token(nsu)}"
        f"_sd{_num_token(sd)}"
    )
    return tag


def get_path(data_path, cfg, which, suffix=""):
    """
    Compatible with:
      - scripts/00_init_directories.py: get_path(cfg.data_path, cfg, "", suffix="")
      - DataManager: get_path(cfg.data_path, cfg, "problem"/"ml_data"/"test_instances/")
    Produces config-tagged filenames like:
      data/offering_no_network/problem_T24_S25_G12_rt800_..._sd7.pkl
    """
    # keep relative "data/..." so scripts' split("/") logic works
    if isinstance(data_path, str) and data_path.startswith("./"):
        data_path = data_path[2:]
    data_path = str(data_path).rstrip("/")

    base = os.path.join(data_path, "offering_no_network")
    os.makedirs(base, exist_ok=True)

    tag = _cfg_tag(cfg)

    # 00_init_directories passes which="" just to infer "data/<problem>/..."
    if which in ("", "problem"):
        return os.path.join(base, f"problem_{tag}.pkl")

    if which == "ml_data":
        # suffix may be "", "_xxx", "_xxx.pkl", "xxx.pkl" etc.
        suf = str(suffix or "")
        if suf and not suf.startswith("_"):
            suf = "_" + suf
        if suf and not suf.endswith(".pkl"):
            suf = suf + ".pkl"
        return os.path.join(base, f"ml_data_{tag}{suf}" if suf else f"ml_data_{tag}.pkl")

    if which == "test_instances/":
        inst_dir = os.path.join(base, "test_instances")
        os.makedirs(inst_dir, exist_ok=True)
        return inst_dir + os.sep

    # be permissive for other tokens used by scripts
    return os.path.join(base, f"problem_{tag}.pkl")






# # ro/utils/offering_no_network.py
# import os


# def _clean_data_root(data_path: str) -> str:
#     """
#     Normalize data root so logs/paths match the repo style (e.g. 'data/...', not './data/...').
#     """
#     if data_path is None:
#         data_path = "data"
#     data_path = str(data_path).strip()
#     if data_path.startswith("./"):
#         data_path = data_path[2:]
#     return data_path.rstrip("/")


# def _ensure_dir(path: str) -> str:
#     os.makedirs(path, exist_ok=True)
#     return path


# def get_path(data_path, cfg, which, suffix=""):
#     """
#     Path factory for problem: offering_no_network

#     This is made compatible with existing scripts, which may call:
#         get_path(cfg.data_path, cfg, "", suffix="")
#         get_path(cfg.data_path, cfg, "problem")
#         get_path(cfg.data_path, cfg, "ml_data")
#         get_path(cfg.data_path, cfg, "random_search", suffix="xxx.pkl")
#         ...

#     Rules:
#     - For directory targets (random_search, eval_results, etc.):
#         - if suffix is empty -> return directory path with trailing os.sep
#         - if suffix is not empty -> return file path inside that directory
#     - For file targets (problem, ml_data):
#         - return full file path (suffix ignored unless you explicitly pass a non-empty suffix)
#     """
#     root = _clean_data_root(data_path)
#     prob = "offering_no_network"
#     base = os.path.join(root, prob)

#     # 00_init_directories.py uses which="" and expects something splittable like "data/offering_no_network/x"
#     if which == "" or which is None:
#         _ensure_dir(base)
#         return os.path.join(base, "problem.pkl")

#     # canonical files
#     if which == "problem":
#         _ensure_dir(base)
#         # allow overriding filename via suffix if caller wants (keep backward compatible)
#         if suffix:
#             return os.path.join(base, suffix)
#         return os.path.join(base, "problem.pkl")

#     if which == "ml_data":
#         _ensure_dir(base)
#         if suffix:
#             return os.path.join(base, suffix)
#         return os.path.join(base, "ml_data.pkl")

#     # canonical dirs (created by 00_init_directories.py)
#     dir_keys = {
#         "random_search": "random_search",
#         "ml_ccg_results": "ml_ccg_results",
#         "ml_ccg_pga_results": "ml_ccg_pga_results",
#         "eval_results": "eval_results",
#         "eval_results_pga": "eval_results_pga",
#         "eval_instances": "eval_instances",
#         "baseline_results": "baseline_results",
#         "test_instances/": "test_instances",
#     }

#     if which in dir_keys:
#         d = os.path.join(base, dir_keys[which])
#         _ensure_dir(d)
#         if suffix:
#             return os.path.join(d, suffix)
#         return d + os.sep

#     # allow scripts to pass trailing slash variants
#     if which.endswith("/") and which[:-1] in dir_keys:
#         d = os.path.join(base, dir_keys[which[:-1]])
#         _ensure_dir(d)
#         if suffix:
#             return os.path.join(d, suffix)
#         return d + os.sep

#     raise ValueError(
#         f"[offering_no_network.get_path] Unknown which={which}. "
#         f"Expected one of: {sorted(list(dir_keys.keys()) + ['problem', 'ml_data', ''])}"
#     )





