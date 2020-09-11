from matplotlib import cm

from experiments.optimizer_names import names


def _method_dict():
    cmap = cm.Set1

    def style(prior: bool = False, copula: bool = False):
        ms = 's' if prior else ""
        ls = '--' if copula else '-'
        return ls, ms

    rs_copula_color = cmap(0)
    rs_color = cmap(0)

    gcp_color = cmap(1)
    gp_color = cmap(1)

    styles = {
        names.GCP_prior: (style(prior=True, copula=True), gcp_color),
        names.GCP_prior_mo: (style(prior=True, copula=True), gcp_color),
        names.GCP: (style(prior=False, copula=True), gcp_color),
        names.GP_prior: (style(prior=True, copula=False), gp_color),
        names.GP_prior_mo: (style(prior=True, copula=False), gp_color),
        names.GP: (style(prior=False, copula=False), gp_color),
        names.CTS_prior: (style(prior=True, copula=True), rs_copula_color),
        names.CTS_prior_mo: (style(prior=True, copula=True), rs_copula_color),
        names.TS_prior: (style(prior=True), rs_color),
        names.TS_prior_mo: (style(prior=True), rs_color),
        names.RS: (style(prior=False), rs_color),
        names.AUTORANGE_GP: (style(), cmap(2)),
        names.WS_BEST: (style(), cmap(3)),
        names.AUTORANGE_GP: (style(), cmap(4)),
        names.ABLR: (style(), cmap(2)),
        names.ABLR_COPULA: (style(copula=True), cmap(2)),
        names.BOHB: (style(), cmap(6)),
        names.REA: (style(), cmap(7)),
        names.REINFORCE: (style(), cmap(8)),
        names.GCP_ho_prior: (style(), "black"),
        names.CTS_ho_prior: (style(), "black"),
        names.EHI: (style(), cmap(2)),
        names.SMS: (style(), cmap(3)),
        names.SUR: (style(), cmap(4)),
        names.EMI: (style(), cmap(5)),
        names.SGPT: (style(), cmap(9)),
        names.SGPT_COPULA: (style(copula=True), cmap(9)),
    }

    return styles


def optimizer_style(method: str):
    styles = _method_dict()

    #method = method.strip(names.MO_suffix)
    assert method in styles, f"method {method} is missing a style"

    return styles[method]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    m = list(_method_dict().items())
    plt.figure(figsize=(5, 5))
    for i, (method, ((ls, ms), color)) in enumerate(m):
        plt.plot(range(10), [i] * 10, ls=ls, marker=ms, color=color, label=method)
    plt.legend()
    plt.show()
