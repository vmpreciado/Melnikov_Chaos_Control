"""
Regenerate the LaTeX number sources from results/*.json.

Writes, at the repository root:
  numbers.tex        inline macros (ratios, savings, matched margins, |c0|)
  tab_duffing.tex    body rows of the MEMS/Duffing results table
  tab_josephson.tex  body rows of the Josephson results table
  tab_compare.tex    body rows of the one-branch vs two-branch comparison

No result number is typed into the .tex by hand; every value flows from a
solver through results/*.json to here. Run compute_results.py first.
"""

import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')


def load(name):
    with open(os.path.join(RESULTS, f'{name}.json')) as f:
        return json.load(f)


def f4(x):
    return f'{x:.4f}'


def f2(x):
    return f'{x:.2f}'


def ratio(num, den):
    return f'{num / den:.2f}'


def pct_saving(pnew, pref):
    return f'{round((1.0 - pnew / pref) * 100):d}'


def pct_larger(tnew, tref):
    return f'{round((tnew / tref - 1.0) * 100):d}'


def results_table(res):
    """Rows: N, two-branch tau*, two-branch P*, tau ratio, P ratio, solve time."""
    tb = res['two_branch']
    Ns = res['Ns']
    n1 = str(Ns[0])
    tau1 = tb['tau'][n1]
    p1 = tb['Pmin'][n1]
    rows = []
    for N in Ns:
        s = str(N)
        tau = tb['tau'][s]
        p = tb['Pmin'][s]
        t = tb['solve_min'][s]
        rows.append(
            f"\t\t\t{N} & {f4(tau)} & {f4(p)} & {ratio(tau, tau1)} & "
            f"{ratio(p, p1)} & {t:.3f} \\\\")
    # \bottomrule lives inside the file so the final \\ is followed in-stream
    # (an \input boundary right after \\ triggers a spurious \noalign error).
    return '\n'.join(rows) + '\n\t\t\t\\bottomrule\n'


def compare_rows(label, res):
    ob, tb = res['one_branch'], res['two_branch']
    rows = []
    for i, N in enumerate(res['Ns']):
        s = str(N)
        sys_col = label if i == 0 else ''
        rows.append(
            f"\t\t\t{sys_col} & {N} & {f4(ob['tau'][s])} & "
            f"{f4(ob['tau_mirror'][s])} & {f4(tb['tau'][s])} & "
            f"{f4(tb['Pmin'][s])} \\\\")
    return '\n'.join(rows)


def helmholtz_maxmargin_rows(hel):
    """Rows: N, max-margin tau*, tau*/tau*_N1."""
    t = hel['max_margin']['tau']
    t1 = t['1']
    rows = [f"\t\t\t{N} & {f4(t[str(N)])} & {ratio(t[str(N)], t1)} \\\\"
            for N in hel['Ns']]
    return '\n'.join(rows) + '\n\t\t\t\\bottomrule\n'


def helmholtz_minpower_rows(hel):
    """Rows: N, min-power P* at tau_target (N=1 infeasible)."""
    P = hel['min_power']['P']
    rows = []
    for N in hel['Ns']:
        v = P[str(N)]
        cell = 'infeasible' if v == 'INFEASIBLE' else f4(v)
        rows.append(f"\t\t\t{N} & {cell} \\\\")
    return '\n'.join(rows) + '\n\t\t\t\\bottomrule\n'


def main():
    duff = load('duffing')
    jos = load('josephson')
    hel = load('helmholtz')

    # --- table bodies ---
    with open(os.path.join(ROOT, 'tab_duffing.tex'), 'w') as f:
        f.write(results_table(duff))
    with open(os.path.join(ROOT, 'tab_josephson.tex'), 'w') as f:
        f.write(results_table(jos))
    with open(os.path.join(ROOT, 'tab_compare.tex'), 'w') as f:
        f.write(compare_rows('MEMS/Duffing', duff) + '\n\t\t\t\\midrule\n' +
                compare_rows('Josephson', jos) + '\n\t\t\t\\bottomrule\n')
    with open(os.path.join(ROOT, 'tab_helmholtz_max.tex'), 'w') as f:
        f.write(helmholtz_maxmargin_rows(hel))
    with open(os.path.join(ROOT, 'tab_helmholtz_min.tex'), 'w') as f:
        f.write(helmholtz_minpower_rows(hel))

    # --- inline macros ---
    dtb, jtb = duff['two_branch'], jos['two_branch']
    dN, jN = duff['Ns'], jos['Ns']
    macros = []

    def mac(name, val):
        macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')

    # Duffing
    mac('DuffAbsCzero', f4(duff['abs_c0']))
    mac('DuffTauMatched', f4(dtb['tau'][str(dN[0])]))
    mac('DuffMarginRatio', ratio(dtb['tau']['3'], dtb['tau']['1']))
    mac('DuffPowerSaving', pct_saving(dtb['Pmin']['3'], dtb['Pmin']['1']))
    mac('DuffPowMatchedNone', f2(dtb['Pmin']['1']))
    mac('DuffPowMatchedNthree', f2(dtb['Pmin']['3']))
    # Josephson
    mac('JosAbsCzero', f'{jos["abs_c0"]:.3f}')
    mac('JosTauMatched', f4(jtb['tau'][str(jN[0])]))
    mac('JosMarginPct', pct_larger(jtb['tau']['3'], jtb['tau']['1']))
    mac('JosPowerSaving', pct_saving(jtb['Pmin']['3'], jtb['Pmin']['1']))
    # Helmholtz (escape oscillator)
    ht = hel['max_margin']['tau']
    hel_ratio = ht['2'] / ht['1']
    mac('HelAbsCzero', f'{hel["abs_c0"]:.2f}')
    mac('HelPmax', f'{hel["Pmax"]:.2f}')
    mac('HelTauMaxSingle', f4(ht['1']))
    mac('HelTauMaxPlateau', f4(ht['2']))
    mac('HelMarginRatio', f'{hel_ratio:.1f}')
    mac('HelTauTarget', f'{hel["tau_target"]:.2f}')
    mac('HelPowMin', f4(hel['min_power']['P']['2']))
    mac('HelShSup', f4(hel['sh_sup_tau']))
    mac('HelShSupPower', f4(hel['sh_sup_power']))
    mac('HelFullCancelPower', f4(hel['full_cancel_power']))
    mac('HelBetaOne', f4(hel['beta'][0]))
    mac('HelBetaTwo', f4(hel['beta'][1]))
    mac('HelBetaThree', f4(hel['beta'][2]))
    mac('HelCosOne', f4(hel['cos_coeffs'][0]))
    mac('HelCosTwo', f4(hel['cos_coeffs'][1]))
    mac('HelCosThree', f4(hel['cos_coeffs'][2]))
    mac('HelMuncLo', f4(hel['M_unc_range'][0]))
    mac('HelMuncHi', f4(hel['M_unc_range'][1]))
    # Global headline: largest correctly certified margin ratio across cases.
    headline = max(dtb['tau']['3'] / dtb['tau']['1'],
                   jtb['tau']['3'] / jtb['tau']['1'], hel_ratio)
    mac('HeadlineMarginRatio', f'{headline:.1f}')

    # Lenci-Rega head-to-head benchmark.
    lrpath = os.path.join(RESULTS, 'lenci_rega.json')
    if os.path.exists(lrpath):
        with open(lrpath) as lf:
            lr = json.load(lf)
        rows = [f"\t\t\t{r['N']} & {r['gain_lr']:.4f} & {r['gain_sdp']:.4f} \\\\"
                for r in lr['rows']]
        with open(os.path.join(ROOT, 'tab_lenci_rega.tex'), 'w') as f:
            f.write('\n'.join(rows) + '\n\t\t\t\\bottomrule\n')
        mac('LRGainInf', f"{lr['gain_infinite']:.0f}")

    # Peak-amplitude demonstration (MEMS).
    ppath = os.path.join(RESULTS, 'peak_demo.json')
    if os.path.exists(ppath):
        with open(ppath) as pf:
            pk = json.load(pf)
        mac('DuffPeakUmax', f"{pk['u_max']:.1f}")
        mac('DuffPeakTau', f4(pk['tau_peak']))
        mac('DuffPeakMeanSquare', f4(pk['peak_meansquare']))
        mac('DuffPeakTauUnc', f4(pk['tau_meansquare']))

    # Phase 4 dynamical-validation numbers (if computed).
    vpath = os.path.join(RESULTS, 'validation.json')
    if os.path.exists(vpath):
        with open(vpath) as vf:
            val = json.load(vf)

        def sgn(x, d=2):
            return f'{x:+.{d}f}'

        de = val['duffing_eps']
        i25 = de['eps'].index(0.25)
        mac('DuffLyapUnc', sgn(de['lambda_unc'][i25]))
        mac('DuffLyapCtrl', sgn(de['lambda_ctrl'][i25]))
        mac('DuffEpsLo', f"{min(de['eps']):.2f}")
        mac('DuffEpsHi', f"{max(de['eps']):.2f}")

        jl = val['josephson']['lambda_nominal']
        worst = max(jl.values())     # closest to zero, still negative
        mac('JosLyapMax', sgn(worst, 3))

        hb = val['helmholtz_basin']
        j = hb['kappa'].index(hb['k_star'])
        mac('HelBasinKstar', f"{hb['k_star']:g}")
        mac('HelSafeUnc', f"{round(hb['safe_unc'][j] * 100):d}")
        mac('HelSafeCtrl', f"{round(hb['safe_ctrl'][j] * 100):d}")

        mm = {(d['system'], d['N']): d for d in val['margin_meaning']}
        mac('DuffKappaNone', f"{mm[('duffing', 1)]['kappa_1st']:.2f}")
        mac('DuffKappaThree', f"{mm[('duffing', 3)]['kappa_1st']:.2f}")
        mac('JosKappaNone', f"{mm[('josephson', 1)]['kappa_1st']:.2f}")
        mac('JosKappaThree', f"{mm[('josephson', 3)]['kappa_1st']:.2f}")
        mac('HelKappaNone', f"{mm[('helmholtz', 1)]['kappa_1st']:.2f}")
        mac('HelKappaTwo', f"{mm[('helmholtz', 2)]['kappa_1st']:.2f}")

    with open(os.path.join(ROOT, 'numbers.tex'), 'w') as f:
        f.write('% Auto-generated by scripts/inject_numbers.py. Do not edit.\n')
        f.write('\n'.join(macros) + '\n')

    print('wrote numbers.tex, tab_duffing.tex, tab_josephson.tex, tab_compare.tex')
    for m in macros:
        print('  ' + m)


if __name__ == "__main__":
    main()
