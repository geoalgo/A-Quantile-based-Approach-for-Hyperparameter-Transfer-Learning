
class names:
    # put names into a class to add structure and avoid having lots of imports
    RS = "RS"

    # ablation
    GP = "GP"
    GCP_ho_prior = "GCP + homosk. prior"
    GCP = "GCP"
    GCP_prior = "GCP + prior (ours)"
    GP_prior = "GP + prior"
    CTS_ho_prior = "CTS + homosk. prior"
    CTS_prior = "CTS (ours)"
    TS_prior = "TS"
    GP_prior = "GP + prior"

    # multi-objectives
    MO_suffix = " + MO"
    GP_prior_mo = GP_prior + MO_suffix
    GP_mo = GP + MO_suffix
    GCP_prior_mo = "GCP + prior" + MO_suffix + " (ours)"
    GCP_mo = GCP + MO_suffix
    CTS_prior_mo = "CTS + prior" + MO_suffix + " (ours)"
    TS_prior_mo = TS_prior + MO_suffix

    # baselines
    WS_BEST = 'WS GP'
    AUTORANGE_GP = "AutoGP"
    AUTORANGE_RS = "AutoRS"

    BOHB = 'BOHB'
    REA = 'R-EA'
    REINFORCE = 'REINFORCE'
    ABLR = "ABLR"
    ABLR_COPULA = 'ABLR Copula'

    SGPT = "SGPT"
    SGPT_COPULA = "SGPT Copula"

    EHI = "EHI"
    SMS = "SMS"
    SUR = "SUR"
    EMI = "EMI"


def method_name(dataset_name):
    for prefix in ["fcnet", "xgboost"]:
        if prefix in dataset_name:
            return prefix
    if 'nas102' in dataset_name:
        return 'NAS'
    return "DeepAR"


def rename_results(df):
    rename_dict = {
        'ablr_norm_fixed_set_tr': names.ABLR,
        'ablr_copula': names.ABLR_COPULA,
        'copula_gp_1_5_random_fix_sigma_5_tr': names.GCP_ho_prior,
        'copula_gp_1_5_random_pred_sigma_5_tr': names.GCP_prior,
        'copula_gp_1_5_random_pred_sigma_std_5_tr': names.GP_prior,
        'copula_rs_1_fix_sigma_tr': names.CTS_ho_prior,
        'copula_rs_1_pred_sigma_std_tr': names.TS_prior,
        'copula_rs_1_pred_sigma_tr': names.CTS_prior,
        'gp_fixed_set_tr': names.GP,
        'random_fixed_set_tr': names.RS,
        'warm-start-gp-top1-1init': names.WS_BEST,
        'auto-range-gp': names.AUTORANGE_GP,
        'copula_gp_no_proir': names.GCP,
        'sgpt_0.01': names.SGPT,
        #'sgpt_0.10': names.SGPT_010,
        #'sgpt_1.00': names.SGPT_100,
        'sgpt_0.01_copula': names.SGPT_COPULA
    }

    df.method = df.method.apply(lambda name: rename_dict[name] if name in rename_dict else "")
    df = df.loc[df.method != "", :]

    df.dataset = df.dataset.apply(
        lambda name: name.replace("xgboost_", "")
            .replace("_max_resource", "")
            .replace("fcnet_", "")
            .replace("nas102_", "")
            .replace("_lookup", "")
    )

    df = df[df.dataset != 'skin_nonskin']

    return df