from types import SimpleNamespace


# ---------------------#
#   Knapsack Problem   #
# ---------------------#

kp = SimpleNamespace(
    # type of data (general or instance)
    data_type = "general",

    # variable parameters between isntances
    n_items = [20, 30, 40, 50, 60, 70, 80],
    correlation = ["UN", "WC", "ASC", "SC"],
    h = [40, 80],
    delta = [0.1, 0.5, 1.0],
    budget_factor = [0.1, 0.15, 0.20],

    # fixed parameters between isntances
    R = 1000,
    H = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size
    
    # n_samples_inst = 500,       # number of instances to samples
    # n_samples_fs = 10,          # number of first-stage decisions samples per problem
    # n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

    n_samples_inst = 50,       # number of instances to samples
    n_samples_fs = 5,          # number of first-stage decisions samples per problem
    n_samples_per_fs = 10,      # number of uncertainty samples per first-stage decision



    # generic parameters
    seed = 7,
    data_path = './data/',

)


# --------------------------- #
#   Capital Budgeting Problem #
# --------------------------- #

cb = SimpleNamespace(
    # problem parameters
    n_items=[10, 20, 30, 40, 50],
    k=.8,
    loans=0,
    l=.12,    # default but not needed for this instance
    m=1.2,     # default but also not needed
    xi_dim=4,

    # data generation parameters
    obj_label="fs_plus_ss_obj",     # label for objective prediction
    feas_label="min_budget_cost",   # label for feasibility prediction
    
    time_limit=30,  # for data generation only
    mip_gap=0.01,   # for data generation only
    verbose=0,      # for data generation only
    threads=1,      # for data generation only

    tr_split=0.80,  # train/test split size

    # n_samples_inst = 500,       # number of instances to samples
    # n_samples_fs = 10,          # number of first-stage decisions samples per problem
    # n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

    n_samples_inst = 50,       # number of instances to samples
    n_samples_fs = 5,          # number of first-stage decisions samples per problem
    n_samples_per_fs = 10,      # number of uncertainty samples per first-stage decision


    # generic parameters
    seed=1,
    inst_seed=range(1, 101),
    data_path='./data/',
)



# ----------------------------------#
# Offering strategy (no network)     #
# ----------------------------------#
offering_no_network = SimpleNamespace(
    # -------- 基本尺寸/数据来源 --------
    data_type="instance",          # 该问题更像“固定一个实例”（价格矩阵+负荷+PV边界）
    Sbase=1000.0,                  # 与Matlab一致：kW/kWh -> pu（除以Sbase）
    T=24,
    n_scenarios=25,
    price_mat_path="./data/offering_no_network/price_matrix_revised_100.mat",
    lambda_rt_value=800.0,

    # DA 价格矩阵（Matlab: price_matrix_revised_100.mat, 24x25）
    price_mat_file="./data/offering_no_network/price_matrix_revised_100.mat",
    price_mat_var="price_matrix_revised_100",

    # 概率（Matlab MP: uniform）
    rho_mode="uniform",

    # RT 偏差惩罚价（Matlab：lambda_rt = 800 * ones(24,S)）
    rt_price_mode="constant",
    rt_price_const=800.0,

    # -------- PV 不确定性（与Matlab一致：min/max + shift + Gamma预算）--------
    pv_min_kw=[240.0 * v for v in [
        0,0,0,0,0,0,0, 1.98275498,2.40239236,3.68092735,5.03178247,5.03178247,
        5.03178247,5.03178247,5.03178247,5.03178247,4.27448781,3.34392749,
        2.12267122,1.69047437,0,0,0,0
    ]],

    pv_max_kw=[240.0 * v for v in [
        0,0,0,0,0,0,0, 5.29490001,5.71453739,6.99307238,8.34392749,8.34392749,
        8.34392749,8.34392749,8.34392749,8.34392749,7.58663283,6.65607251,
        5.43481625,5.00261939,0,0,0,0
    ]],

    pv_max_shift_kw=1.0,           # Matlab: p_pv_max = p_pv_max + 1*ones(1,24)
    Gamma=12.0,

    # -------- 聚合负荷（Matlab示例：10*[...] / Sbase）--------
    load_sum_kw=[10.0 * v for v in [
        111.3822392,101.5065653,94.30973857,93.17348758,94.48769219,99.68165177,
        112.8326054,116.1722534,124.5485385,131.4933456,143.5932615,160.0526555,
        167.6674216,182.9392865,203.4062827,221.0867556,239.826844,244.9674648,
        232.526104,211.0252955,198.7750239,168.5545511,142.6519859,122.7069434
    ]],

    # -------- ESS 参数（与Matlab一致）--------
    N_es=10,
    E_ess_per_unit_kwh=14.5,       # Matlab: E_ess_0 = N_es*14.5/Sbase
    P_ess_per_unit_kw=11.3,        # Matlab: P_ess_max_0 = N_es*11.3/Sbase
    eta=0.95,

    # 成本（Matlab示例保持同量级）
    ESS_cost=1e-6,                 # 你也可以按Matlab写法在建模处换算
    PV_cost=1e-6,

    # --------------------------------------------------------------------
    # 关键：保留 offer curve “单调性”采样方式（贴近Matlab MP 的单调性约束）
    # Matlab 约束：if lambda_da(t,s) > lambda_da(t,sp) => p_DA(t,s) >= p_DA(t,sp)
    # --------------------------------------------------------------------
    offer_curve_sampling="monotone_by_price",  # 必须保留
    # 为了“最小可跑”，需要给采样一个合理范围（pu）
    # 下面 margin 用来放宽边界，避免过窄导致样本分布奇怪
    p_da_bound_margin_pu=0.05,

    # -------- 数据集生成（最小可跑）--------
    time_limit=30,
    mip_gap=0.02,
    verbose=0,
    threads=1,
    tr_split=0.80,
    n_samples_inst=500,              # 最小可跑：1个实例
    n_samples_fs=10,                # 最小可跑：每个实例采2条offer curve
    n_samples_per_fs=500,            # 最小可跑：每条curve采3个不确定性xi

    # -------- 通用 --------
    seed=7,
    data_path="./data/",
)





