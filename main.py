"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Model
-----
An Exophase earn rate is a ratio: r_i(t) = E_i(t) / N(t), where N(t) is the
number of players Exophase tracks for the game and E_i(t) the number of them
who have earned trophy i. Both series are observed in the case history. The
house baselines ignore that structure: persistence says the ratio stays put,
and the curve baseline extrapolates the ratio's logit linearly in log-time.
Both miss the two forces that actually move the ratio -- new players flooding
in (N grows, diluting every rate) and the tracked players progressing through
the game (E_i grows as they reach each trophy).

This entry models both sides:

* N(t): the tracked-player count. Three growth forms are fitted on the
  history (power law in log-time, quadratic in log-time, saturating
  exponential) and the one that backtests best on the last 30% of the time
  span is extrapolated to the target.
* E_i(t): a cohort model. Players join at rate J(s) = dN/ds. A player who
  joined at time s has, by age tau, probability F_i(tau) = c_i * G(tau - d_i)
  of holding trophy i, where G(tau) = 1 - exp(-tau/theta_i) is a progression
  shape whose time constant theta_i and delay d_i are fitted per trophy by
  model-averaging over a grid (trophies are earned fast or slow; a single
  global shape misfits both), and c_i is the trophy's eventual completion
  fraction. So E_i(t) = integral of J(s) * F_i(t-s) ds. With the
  exponential shape the convolution collapses to prefix sums, so each fit
  is closed form and the whole program runs in about a second.
* The forecast rate is 100 * E_i(target) / N(target).

Quantiles are asymmetric in logit space: the downside spread comes from the
dilution scenario (N(target) 1.5x larger, E fixed) and the upside spread from
the progression scenario (E_i(target) 1.5x larger, N fixed), each floored by
the model's in-sample residual spread and a horizon term. A trophy whose
history is all zeros (nobody has earned it yet) gets a tier/description
prior with a very wide upside band instead of a zero forecast, because the
scorer's logit scale punishes a 0.01% forecast against a 0.5% truth far
more than a 0.3% forecast against a 0.01% truth.

Degenerate history (the practice cases: every read identical, because
Exophase had not refreshed yet) cannot identify any of this, so it takes a
refresh path instead: the rules record the first refresh multiplying the
tracked count by ~6.06 (638 -> 3,868) with every rate but the 100% one
falling. New players earn each trophy at a fraction alpha of the old
players' rate, alpha larger for the common (early) trophies, and the
forecast band spans the alpha = 0 to alpha = 0.5 scenarios.

The program is deterministic, standard library only, reads every <id>.in in
the input directory and writes one <id>.out per case. It never uses the
network or any data beyond the case file itself.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of GTA VI on PS5: the time origin for every fit (hours).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# The first refresh the rules record: 638 -> 3,868 players tracked.
REFRESH_KAPPA = 3868.0 / 638.0


def _parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours(ts):
    return (ts - RELEASE).total_seconds() / 3600.0


def _clamp(x, lo, hi):
    return min(hi, max(lo, x))


def _logit(pct):
    p = _clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    lg = _clamp(lg, -9.21, 9.21)  # keeps the rate inside [0.01, 99.99]
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _ols(xs, ys):
    """Ordinary least squares y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return my - (sxy / sxx) * mx, sxy / sxx


def _ols2(xs, ys):
    """Least squares y ~ a + b*x + c*x^2. Returns (a, b, c) via normal equations."""
    n = float(len(xs))
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sx3 = sum(x ** 3 for x in xs)
    sx4 = sum(x ** 4 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum(x * x * y for x, y in zip(xs, ys))
    # Solve the 3x3 system by Cramer's rule.
    m = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    v = [sy, sxy, sx2y]

    def det3(a):
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    d = det3(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for col in range(3):
        mm = [row[:] for row in m]
        for row in range(3):
            mm[row][col] = v[row]
        out.append(det3(mm) / d)
    return tuple(out)


# --------------------------------------------------------------------------
# History preparation
# --------------------------------------------------------------------------

def _read_series(case):
    """Distinct reads as (t_hours, N, [rates]) plus the trophy list.

    Consecutive identical reads (Exophase had not refreshed) collapse to one:
    they carry no new information but would overweight flat stretches.
    """
    trophies = case["trophies"]
    rows = []
    for read in case.get("history") or []:
        try:
            t = _hours(_parse_ts(read["at"]))
            n = float(read["players_tracked"])
            rates = read["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        if n <= 0 or not isinstance(rates, list):
            continue
        if len(rates) != len(trophies):
            continue
        rows.append((t, n, rates))
    rows.sort(key=lambda r: r[0])
    distinct = []
    for row in rows:
        if distinct and distinct[-1][1] == row[1] and distinct[-1][2] == row[2]:
            # Same snapshot again: keep the later time, not a second copy.
            distinct[-1] = (row[0], row[1], row[2])
            continue
        distinct.append(row)
    return trophies, distinct


def _earners(n, rate):
    """Expected earners from a rate in percent (rates print to 2 decimals)."""
    if rate is None:
        return None
    return max(0.0, float(rate)) * n / 100.0


# --------------------------------------------------------------------------
# N(t): tracked-player growth, picked by backtest
# --------------------------------------------------------------------------

def _fit_n(rows, t_target):
    """Fit N(t) and extrapolate to t_target. Returns (N_target, form_name).

    Three growth forms are tried (power law in log-time, quadratic in
    log-time, saturating exponential). The one that backtests best on the
    last 30% of the time span is used; with too little history the power
    law is the default.
    """
    ts = [r[0] for r in rows]
    ns = [r[1] for r in rows]
    n_last = ns[-1]
    if len(rows) < 3:
        return n_last, "flat"

    def fit_lin(xs, ys):
        a, b = _ols(xs, ys)
        return ("lin", a, b)

    def fit_quad(xs, ys):
        q = _ols2(xs, ys)
        if q is None:
            return None
        return ("quad", q[0], q[1], q[2])

    def fit_sat(ts_, ns_):
        best = None
        for k in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            basis = [1.0 - math.exp(-k * max(0.0, t)) for t in ts_]
            denom = sum(v * v for v in basis)
            if denom <= 0:
                continue
            amp = sum(v * n for v, n in zip(basis, ns_)) / denom
            if amp <= 0:
                continue
            err = sum(
                (math.log(max(1.0, amp * v)) - math.log(max(1.0, n))) ** 2
                for v, n in zip(basis, ns_)
            )
            if best is None or err < best[0]:
                best = (err, ("sat", k, amp))
        return best[1] if best else None

    def predict(form, t):
        if form[0] == "lin":
            return form[1] + form[2] * math.log1p(max(0.0, t))
        if form[0] == "quad":
            x = math.log1p(max(0.0, t))
            return form[1] + form[2] * x + form[3] * x * x
        if form[0] == "sat":
            k, amp = form[1], form[2]
            return math.log(max(1.0, amp * (1.0 - math.exp(-k * max(0.0, t)))))
        return math.log(n_last)

    xs = [math.log1p(max(0.0, t)) for t in ts]
    ys = [math.log(max(1.0, n)) for n in ns]
    forms = {}
    forms["lin"] = fit_lin(xs, ys)
    fq = fit_quad(xs, ys)
    if fq is not None:
        forms["quad"] = fq
    fs = fit_sat(ts, ns)
    if fs is not None:
        forms["sat"] = fs

    # Backtest: fit on the first 70% of the time span, score log-N on the rest.
    chosen = forms["lin"]
    t0, t1 = ts[0], ts[-1]
    span = t1 - t0
    if span > 6 and len(rows) >= 8:
        cut = t0 + 0.7 * span
        ei = [j for j, t in enumerate(ts) if t <= cut]
        li = [j for j, t in enumerate(ts) if t > cut]
        if len(ei) >= 4 and len(li) >= 2:
            best_name, best_err = None, None
            for name in forms:
                if name == "lin":
                    f = fit_lin([xs[j] for j in ei], [ys[j] for j in ei])
                elif name == "quad":
                    f = fit_quad([xs[j] for j in ei], [ys[j] for j in ei])
                    if f is None:
                        continue
                else:
                    f = fit_sat([ts[j] for j in ei], [ns[j] for j in ei])
                    if f is None:
                        continue
                err = sum(
                    (predict(f, ts[j]) - ys[j]) ** 2 for j in li
                ) / len(li)
                if best_err is None or err < best_err:
                    best_name, best_err = name, err
            if best_name is not None:
                chosen = forms[best_name]

    n_t = math.exp(predict(chosen, t_target))
    # Guards: never below the last read, never absurdly above it.
    n_t = _clamp(n_t, n_last, n_last * 50.0)
    return n_t, chosen[0]


def _n_curve(rows, t_target):
    """A smooth N(t) callable from the fitted form, plus N(target)."""
    n_target, form_name = _fit_n(rows, t_target)
    ts = [r[0] for r in rows]
    ns = [r[1] for r in rows]
    xs = [math.log1p(max(0.0, t)) for t in ts]
    ys = [math.log(max(1.0, n)) for n in ns]
    if len(rows) >= 3:
        a, b = _ols(xs, ys)
        q = _ols2(xs, ys)
    else:
        a, b, q = ys[-1], 0.0, None
    # Saturating form parameters (re-fit, for the callable).
    sat = None
    best_sat = None
    for k in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        basis = [1.0 - math.exp(-k * max(0.0, t)) for t in ts]
        denom = sum(v * v for v in basis)
        if denom <= 0:
            continue
        amp = sum(v * n for v, n in zip(basis, ns)) / denom
        if amp <= 0:
            continue
        err = sum(
            (math.log(max(1.0, amp * v)) - y) ** 2 for v, y in zip(basis, ys)
        )
        if best_sat is None or err < best_sat[0]:
            best_sat = (err, k, amp)
    if best_sat is not None:
        sat = (best_sat[1], best_sat[2])
    if form_name == "quad" and q is not None:
        def n_at(t):
            x = math.log1p(max(0.0, t))
            return max(1.0, math.exp(q[0] + q[1] * x + q[2] * x * x))
    elif form_name == "sat" and sat is not None:
        k, amp = sat

        def n_at(t):
            return max(1.0, amp * (1.0 - math.exp(-k * max(0.0, t))))
    else:
        def n_at(t):
            x = math.log1p(max(0.0, t))
            return max(1.0, math.exp(a + b * x))
    return n_at, n_target


# --------------------------------------------------------------------------
# Cohort model for E_i(t)
# --------------------------------------------------------------------------

# Global progression time constants (hours) tried for the common shape.
THETAS = (6.0, 12.0, 24.0, 48.0, 96.0, 192.0)
# Per-trophy delays (hours) before the trophy becomes earnable in a journey.
DELAYS = (0.0, 2.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0)


def _exposure_table(n_at, theta, t_max):
    """Prefix sums for the exponential kernel with time constant theta.

    Returns a function x_at(delay, t) = integral of J(s) * (1 - exp(-(t - s -
    delay)/theta)) ds. The prefix arrays depend on theta but not on the
    delay, so one table serves every delay: X_d(t) = N(t-d) -
    exp(-(t-d)/theta) * A(t-d), where A(u) = integral_0^u J(s) exp(s/theta) ds.
    Bins are 1 hour.
    """
    n_bins = int(math.ceil(max(0.0, t_max))) + 1
    a_pref = [0.0] * (n_bins + 1)
    n_edge = [0.0] * (n_bins + 1)
    prev_n = n_at(0.0)
    n_edge[0] = prev_n
    for k in range(1, n_bins + 1):
        t_k = float(k)
        n_k = n_at(t_k)
        n_edge[k] = n_k
        j = max(0.0, n_k - prev_n)  # inflow during hour k
        a_pref[k] = a_pref[k - 1] + j * math.exp((t_k - 0.5) / theta)
        prev_n = n_k

    def x_at(delay, t):
        u = t - delay
        if u <= 0:
            return 0.0
        k = int(math.floor(u))
        frac = u - k
        if k >= n_bins:
            a_u = a_pref[n_bins]
            n_u = n_edge[n_bins]
        else:
            a_u = a_pref[k] + frac * (a_pref[k + 1] - a_pref[k])
            n_u = n_edge[k] + frac * (n_edge[k + 1] - n_edge[k])
        return max(0.0, n_u - math.exp(-u / theta) * a_u)

    return x_at


def _exposure(n_at, theta, delay, times):
    """X_d(t) for a list of times (thin wrapper over _exposure_table)."""
    x_at = _exposure_table(n_at, theta, max(times) if times else 0.0)
    return [x_at(delay, t) for t in times]


def _fit_scale(xs, ys, ws):
    """Closed-form c minimising sum w (c*x - y)^2. Returns (c, weighted SSE)."""
    denom = sum(w * x * x for x, y, w in zip(xs, ys, ws))
    if denom <= 0:
        return 0.0, float("inf")
    c = sum(w * x * y for x, y, w in zip(xs, ys, ws)) / denom
    return max(0.0, c), sum(w * (c * x - y) ** 2 for x, y, w in zip(xs, ys, ws))


def _logit_err(pred_rates, obs_rates, weights):
    """Weighted mean squared logit error between predicted and observed rates."""
    tot, wsum = 0.0, 0.0
    for pr, ob, w in zip(pred_rates, obs_rates, weights):
        if ob is None:
            continue
        d = _logit(pr) - _logit(max(0.005, ob))
        tot += w * d * d
        wsum += w
    return tot / wsum if wsum > 0 else float("inf")


def _fit_cohort(rows, trophies, t_target, n_at, n_target):
    """Fit the cohort model. Returns per-trophy dicts with median and spread.

    Each trophy's progression shape is its own: the time constant theta and
    the delay delta are fitted per trophy by model-averaging over a grid
    (a single global theta misfits trophies that players earn fast or slow),
    with the scale c in closed form for each (theta, delta) pair.
    """
    times = [r[0] for r in rows]
    ns = [r[1] for r in rows]
    n_t = len(trophies)
    earn = []
    for (_, n, rates) in rows:
        earn.append([_earners(n, r) for r in rates])
    t_max = max(times + [t_target])
    tables = {th: _exposure_table(n_at, th, t_max) for th in THETAS}

    # Global theta from the aggregate earners (trophies under 50% only, so
    # the 100% trophy does not dominate). It centres the per-trophy prior:
    # a trophy's own theta may differ, but the further it strays from the
    # aggregate the stronger the evidence the fit must show.
    agg_e, agg_w = [], []
    for j in range(len(rows)):
        tot = 0.0
        for i in range(n_t):
            e = earn[j][i]
            r = rows[j][2][i]
            if e is not None and r is not None and r < 50.0:
                tot += e
        agg_e.append(tot)
        agg_w.append(1.0 / (tot + 1.0))
    theta_global, theta_global_err = 24.0, None
    for theta in THETAS:
        xs = [tables[theta](0.0, t) for t in times]
        c, _ = _fit_scale(xs, agg_e, agg_w)
        if c <= 0:
            continue
        pred = [100.0 * c * x / n for x, n in zip(xs, ns)]
        obs = [100.0 * e / n for e, n in zip(agg_e, ns)]
        err = _logit_err(pred, obs, [1.0] * len(rows))
        if theta_global_err is None or err < theta_global_err:
            theta_global, theta_global_err = theta, err
    log_theta_g = math.log(theta_global)

    results = []
    for i in range(n_t):
        obs_e = [earn[j][i] for j in range(len(rows))]
        obs_r = [rows[j][2][i] for j in range(len(rows))]
        ws = [1.0 / ((e or 0.0) + 1.0) for e in obs_e]
        # Logit-error weights: downweight reads where the count is tiny and
        # the 2-decimal rounding dominates.
        lw = [min(e or 0.0, 20.0) + 1.0 for e in obs_e]
        if all(e is None or e <= 0.0 for e in obs_e):
            results.append({"zero": True, "i": i})
            continue
        # Model-average over (theta, delay): several shapes fit a short
        # history almost equally well but extrapolate differently, so the
        # forecast averages them in logit space, weighted by fit quality
        # and by a log-normal prior on theta centred on the aggregate fit.
        cands = []
        for theta in THETAS:
            x_at = tables[theta]
            prior = math.exp(-0.5 * ((math.log(theta) - log_theta_g) / 1.0) ** 2)
            for delay in DELAYS:
                xs = [x_at(delay, t) for t in times]
                xe = [x for x, e in zip(xs, obs_e) if e is not None]
                ye = [e for e in obs_e if e is not None]
                we = [w for w, e in zip(ws, obs_e) if e is not None]
                c, _ = _fit_scale(xe, ye, we)
                if c <= 0:
                    continue
                pred = [100.0 * c * x / n for x, n in zip(xs, ns)]
                err = _logit_err(pred, obs_r, lw)
                xs_t = x_at(delay, t_target)
                cands.append((err, theta, delay, c, xs_t, prior))
        if not cands:
            results.append({"zero": True, "i": i})
            continue
        err_min = min(c[0] for c in cands)
        weights = [
            math.exp(-(err - err_min) / 0.5) * prior
            for err, _, _, _, _, prior in cands
        ]
        wsum = sum(weights)
        med_logit = 0.0
        resid2 = 0.0
        e_t = 0.0
        for (err, _th, _dl, c, xs_t, _pr), w in zip(cands, weights):
            m = _clamp(100.0 * c * xs_t / n_target, 0.005, 99.95)
            med_logit += (w / wsum) * _logit(m)
            resid2 += (w / wsum) * err
            e_t += (w / wsum) * c * xs_t
        med = _clamp(_inv_logit(med_logit), 0.005, 99.95)
        results.append(
            {"zero": False, "i": i, "median": med, "resid": math.sqrt(max(resid2, 0.0)),
             "delay": cands[0][2], "c": cands[0][3], "e_t": e_t}
        )
    return None, results


# --------------------------------------------------------------------------
# Zero-history prior
# --------------------------------------------------------------------------

GRIND_KEYS = (
    "100% completion",
    "attain 100%",
    "250 stunts",
    "50 challenges",
    "collectible card",
    "every location",
    "100 locations",
    "8 mission strands",
    "gold medal in all",
    "every spot",
    "all garage properties",
)


def _is_grind(trophy):
    """True for completionist trophies almost nobody reaches in a week."""
    text = ((trophy.get("name") or "") + " " + (trophy.get("description") or "")).lower()
    return any(k in text for k in GRIND_KEYS)


def _zero_prior(trophy):
    """Median rate (%) for a trophy nobody has earned yet in the history.

    Story trophies (a chapter, the final mission) are reached by some players
    within a week; grind/collection trophies are not. The spread is wide.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = trophy.get("tier")
    story = any(
        k in text
        for k in ("chapter", "final mission", "complete the final", "complete chapter")
    )
    if story:
        return 0.5
    if tier == "platinum":
        return 0.01
    if tier == "gold":
        return 0.02
    if tier == "silver":
        return 0.02
    return 0.03


# --------------------------------------------------------------------------
# Refresh fallback (degenerate history: every read identical)
# --------------------------------------------------------------------------

def _refresh_forecast(case, trophies, rows):
    """Forecast after a refresh the history has not seen yet.

    The last read is the pre-refresh page. The next refresh multiplies the
    tracked count by REFRESH_KAPPA and new players earn each trophy at a
    fraction alpha of the old players' rate (alpha larger for the common,
    early trophies). The band spans the alpha = 0 to alpha = 0.5 scenarios.
    """
    _, n_old, last_rates = rows[-1]
    n_new = n_old * REFRESH_KAPPA
    r_max = 0.0
    for r in last_rates:
        if r is not None and r < 99.9:
            r_max = max(r_max, r)
    if r_max <= 0:
        r_max = 1.0
    out = []
    for i, trophy in enumerate(trophies):
        r = last_rates[i]
        if r is None:
            r = 0.0
        if r >= 99.9:
            q = [98.5, 99.4, 99.9, 99.95, 99.99]
            out.append({"id": trophy["id"], "q": q})
            continue
        alpha = 0.05 + 0.30 * (r / r_max)
        med = r * (n_old + alpha * (n_new - n_old)) / n_new
        lo = r * n_old / n_new
        hi = r * (n_old + 0.5 * (n_new - n_old)) / n_new
        med = _clamp(med, 0.005, 99.95)
        lo = _clamp(lo, 0.005, 99.95)
        hi = _clamp(hi, 0.005, 99.95)
        lm, ll, lh = _logit(med), _logit(lo), _logit(hi)
        qs = [
            lo,
            _round4(_inv_logit((lm + ll) / 2.0)),
            _round4(med),
            _round4(_inv_logit((lm + lh) / 2.0)),
            hi,
        ]
        # Enforce non-decreasing after rounding.
        fixed = []
        prev = 0.0
        for q in qs:
            q = max(prev, _clamp(float(q), 0.0, 100.0))
            fixed.append(_round4(q))
            prev = fixed[-1]
        out.append({"id": trophy["id"], "q": fixed})
    return {"trophies": out}


# --------------------------------------------------------------------------
# Main forecast
# --------------------------------------------------------------------------

def _chapter_order(trophies):
    """Story trophies in play order: Chapter 1..4, then the final mission."""
    order = []
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "").lower()
        if "complete chapter 1" in desc:
            order.append((1, i))
        elif "complete chapter 2" in desc:
            order.append((2, i))
        elif "complete chapter 3" in desc:
            order.append((3, i))
        elif "complete chapter 4" in desc:
            order.append((4, i))
        elif "complete the final mission" in desc:
            order.append((5, i))
    order.sort()
    return [i for _, i in order]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies, rows = _read_series(case)
    if not rows:
        raise ValueError(f"{name} has no history")
    if not trophies:
        raise ValueError(f"{name} has no trophies")
    horizon_days = case["horizon_minutes"] / 1440.0
    t_target = _hours(_parse_ts(case["target"]))

    # Degenerate history: every distinct read identical -> refresh path.
    distinct_states = {(n, tuple(rates)) for (_, n, rates) in rows}
    if len(distinct_states) <= 1:
        return _refresh_forecast(case, trophies, rows)

    n_at, n_target = _n_curve(rows, t_target)
    _, fitted = _fit_cohort(rows, trophies, t_target, n_at, n_target)

    # Backtest spread: fit on the first 70% of the time span, forecast the
    # last read, and measure the out-of-sample logit error. Scaled by the
    # square root of the horizon ratio (random-walk scaling), it is the
    # honest spread for a 4.5-day forecast of a chaotic launch week.
    s_bt = 0.0
    t_first, t_last = rows[0][0], rows[-1][0]
    span = t_last - t_first
    if span > 12 and len(rows) >= 10:
        t_cut = t_first + 0.7 * span
        trunc = [r for r in rows if r[0] <= t_cut]
        if len(trunc) >= 5:
            try:
                n_at_bt, n_bt = _n_curve(trunc, t_last)
                _, fitted_bt = _fit_cohort(trunc, trophies, t_last, n_at_bt, n_bt)
                errs = []
                for res in fitted_bt:
                    i = res["i"]
                    last_rate = rows[-1][2][i]
                    if last_rate is None or last_rate >= 99.9 or res.get("zero"):
                        continue
                    errs.append(
                        (_logit(res["median"]) - _logit(max(0.005, last_rate))) ** 2
                    )
                if errs:
                    e_bt = math.sqrt(sum(errs) / len(errs))
                    h_bt = max(1.0, t_last - t_cut)
                    h_f = max(1.0, t_target - t_last)
                    s_bt = min(2.0, e_bt * math.sqrt(h_f / h_bt))
            except Exception:
                s_bt = 0.0

    s_horizon = 0.30 * math.sqrt(max(0.0, horizon_days))

    # First pass: medians and spread ingredients per trophy.
    med = {}
    s_down = {}
    s_up = {}
    for res in fitted:
        i = res["i"]
        trophy = trophies[i]
        last_rate = rows[-1][2][i]
        if last_rate is not None and last_rate >= 99.9:
            med[i] = 99.9
            s_down[i] = s_up[i] = 0.12
            continue
        if res.get("zero"):
            med[i] = _zero_prior(trophy)
            s_down[i], s_up[i] = 1.0, 2.0
            continue
        m = res["median"]
        s_fit = max(0.25, res["resid"])
        s_base = math.sqrt(max(s_fit, s_bt) ** 2 + s_horizon ** 2)
        s_base = max(s_base, 0.5)
        e_t = res["e_t"]
        r_down = 100.0 * e_t / (n_target * 1.5)
        r_up = 100.0 * (e_t * 1.5) / n_target
        sd = max(s_base, abs(_logit(r_down) - _logit(m)))
        su = max(s_base, abs(_logit(r_up) - _logit(m)))
        med[i] = m
        s_down[i] = _clamp(sd, 0.30, 2.5)
        s_up[i] = _clamp(su, 0.30, 2.5)

    order = _chapter_order(trophies)

    # Effectively-zero fits: the model says essentially nobody earns the
    # trophy by the target, usually because it has barely started being
    # earned in the observed window. What it becomes depends on what it is:
    # a grind trophy stays near zero, an ordinary activity trophy follows
    # its tier's fitted trophies, a story trophy is handled just below.
    tier_med = {}
    for tier in ("bronze", "silver", "gold", "platinum"):
        vals = [
            med[i]
            for i, t in enumerate(trophies)
            if t.get("tier") == tier and 0.05 < med[i] < 99.9
        ]
        if vals:
            tier_med[tier] = _median(vals)
    story_idx = set(order)
    for i, trophy in enumerate(trophies):
        if med[i] > 0.05 or i in story_idx:
            continue
        if _is_grind(trophy):
            med[i] = _zero_prior(trophy)
            s_down[i], s_up[i] = 1.0, 2.0
        else:
            prior = tier_med.get(trophy.get("tier"))
            if prior is None:
                prior = 1.0
            med[i] = _clamp(prior, 0.5, 20.0)
            s_down[i] = max(s_down[i], 1.0)
            s_up[i] = max(s_up[i], 1.5)

    # Story prior: a chapter trophy the fit calls ~nobody for has usually
    # just not started being earned in the observed window (its delay is
    # longer than the history). Its rate by the target follows the chapter
    # sequence, so interpolate it in log space from the chapters that fit.
    if order:
        known = [(k, med[i]) for k, i in enumerate(order) if med[i] > 0.05]
        if len(known) >= 2:
            xs = [float(k) for k, _ in known]
            ys = [math.log(v) for _, v in known]
            a, b = _ols(xs, ys)
            for k, i in enumerate(order):
                if med[i] <= 0.05:
                    med[i] = _clamp(math.exp(a + b * k), 0.05, 99.95)
                    s_down[i] = max(s_down[i], 1.0)
                    s_up[i] = max(s_up[i], 1.5)
        else:
            for i in order:
                if med[i] <= 0.05:
                    med[i] = 0.5
                    s_down[i] = max(s_down[i], 1.0)
                    s_up[i] = max(s_up[i], 1.5)

    # Monotonicity the game enforces: you cannot complete Chapter k+1 (or
    # the final mission) without the earlier chapters, and the platinum
    # needs 100% completion, so cap the medians in that order.
    prev = None
    for i in order:
        if prev is not None:
            med[i] = min(med[i], prev)
        prev = med[i]
    plat = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    full = [
        i
        for i, t in enumerate(trophies)
        if "100% completion" in (t.get("description") or "").lower()
    ]
    if plat and full:
        med[plat[0]] = min(med[plat[0]], med[full[0]])

    out = []
    for i, trophy in enumerate(trophies):
        m = med[i]
        if m >= 99.9:
            out.append({"id": trophy["id"], "q": [98.5, 99.4, 99.9, 99.95, 99.99]})
            continue
        lm = _logit(m)
        qs = [
            _inv_logit(lm - 1.645 * s_down[i]),
            _inv_logit(lm - 0.674 * s_down[i]),
            m,
            _inv_logit(lm + 0.674 * s_up[i]),
            _inv_logit(lm + 1.645 * s_up[i]),
        ]
        fixed = []
        prev = 0.0
        for q in qs:
            q = _round4(_clamp(float(q), 0.0, 100.0))
            q = max(q, prev)
            fixed.append(q)
            prev = q
        out.append({"id": trophy["id"], "q": fixed})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
