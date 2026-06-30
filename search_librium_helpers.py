"""
dashboard.py
============
Post-analysis interactive HTML dashboard for SA discrete choice model search.
Reads siman_results and siman_pert files and produces a self-contained HTML.

Usage (standalone):
    python dashboard.py --id RUN_ID [--results PATH] [--pert PATH]

Called from siman.py finalise():
    from dashboard import generate_dashboard
    generate_dashboard(self.idnum)
"""

import re
import json
import math
import argparse
from pathlib import Path


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_results(filepath):
    """
    Parse siman_results[id].txt → list of solution dicts (Top 1-5 only).
    Reads model type, objective, alternatives and all parameters from file.
    """
    text = Path(filepath).read_text(encoding="utf-8").replace("\r\n", "\n")
    # Manual split on Top solution headers (regex lookahead unreliable with repeated chars)
    idxs   = [m.start() for m in re.finditer(r"_{3,}\nSolution: Top", text)]
    bounds = [0] + idxs
    if idxs:
        blocks = [text[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)] + [text[idxs[-1]:]]
    else:
        blocks = [text]

    first_block    = blocks[0] if blocks else ""
    n_alts         = _parse_n_alternatives(first_block)
    objective_name = _parse_objective_name(first_block)

    solutions = []
    for block in blocks:
        title_match = re.search(r"Top (\d+)", block)
        if not title_match:
            continue

        rank = int(title_match.group(1))
        sol  = {
            "rank":         rank,
            "model":        _parse_model(block),
            "objective":    objective_name,
            "n_alts":       n_alts,
            "converged":    "WARNING" not in block,
            "gtol_ok":      None,
            "ftol_ok":      None,
            "gtol_val":     None,
            "ftol_val":     None,
            "intercepts":   [],
            "fixed":        [],
            "random":       [],
            "correlations": {},
            "corrvars":     [],
            "bcvars":       [],
            "loglik":       None,
            "bic":          None,
            "aic":          None,
            "adjlik":       None,
            "obj_value":    None,
            "predicted":    [],
            "observed":     [],
            "has_bcvars":   False,
            "has_random":   False,
            "draws":                  None,
            "individuals":            None,
            "choices_per_individual": None,
        }
        sol["draws"], sol["individuals"], sol["choices_per_individual"], sol["total_choices"] = _parse_dataset_stats(block)

        # Convergence details
        gtol_line = re.search(r"gtol:.*", block)
        ftol_line = re.search(r"ftol:.*", block)
        if gtol_line:
            gn = re.search(r"gradient norm:\s*([\d.eE+\-]+)", gtol_line.group())
            sol["gtol_val"] = float(gn.group(1)) if gn else None
            sol["gtol_ok"]  = "OK" in gtol_line.group() or "X" not in gtol_line.group()
        if ftol_line:
            fv = re.search(r"function value:\s*([\d.eE+\-]+)", ftol_line.group())
            sol["ftol_val"] = float(fv.group(1)) if fv else None
            sol["ftol_ok"]  = "OK" in ftol_line.group()

        # Goodness of fit
        gof = re.search(
            r"LOGLIK\s*=\s*(-?[\d.]+)\s*\|\s*AIC\s*=\s*(-?[\d.]+)\s*\|\s*BIC\s*=\s*(-?[\d.]+)\s*\|\s*ADJLIK\s*=\s*(-?[\d.]+)",
            block
        )
        if gof:
            sol["loglik"] = float(gof.group(1))
            sol["aic"]    = float(gof.group(2))
            sol["bic"]    = float(gof.group(3))
            sol["adjlik"] = float(gof.group(4))

        # Objective value
        obj_m = re.search(r"\[0\]\.\s*\(.*?\)\s*\w+\s*=\s*([\d.]+)", block)
        if obj_m:
            sol["obj_value"] = float(obj_m.group(1))

        # Shares
        shares = re.search(
            r"Observed:\s*((?:[\d.]+\s+)+)Predicted:\s*((?:[\d.]+\s+)+)", block
        )
        if shares:
            sol["observed"]  = [float(x) for x in shares.group(1).split()]
            sol["predicted"] = [float(x) for x in shares.group(2).split()]

        _parse_parameters(block, sol)
        _parse_correlations(block, sol)

        bcvars_m = re.search(r"bcvars\s*=\s*\[([^\]]*)\]", block)
        if bcvars_m:
            sol["bcvars"] = re.findall(r"'([^']*)'", bcvars_m.group(1))

        sol["has_random"] = len(sol["random"]) > 0
        sol["has_bcvars"] = bool(sol["bcvars"]) or "TRANSFORMED PARAMETERS" in block
        solutions.append(sol)

    solutions.sort(key=lambda x: x["rank"])
    return solutions, objective_name, n_alts


def _parse_model(block):
    m = re.search(r"model = \s*(\S+)", block)
    return m.group(1) if m else "Unknown"


def _parse_dataset_stats(block):
    draws = re.search(r"Draws:\s*(\d+)", block)
    inds  = re.search(r"Individuals:\s*(\d+)", block)
    cpi   = re.search(r"Choices per Individual:\s*(\d+)", block)
    total = re.search(r"Total Choices:\s*(\d+)", block)
    return (
        int(draws.group(1)) if draws else None,
        int(inds.group(1))  if inds  else None,
        int(cpi.group(1))   if cpi   else None,
        int(total.group(1)) if total else None,
    )


def _parse_n_alternatives(block):
    m = re.search(r"Observed:\s*((?:[\d.]+\s+)+)", block)
    return len(m.group(1).split()) if m else 3


def _parse_objective_name(block):
    m = re.search(r"\[0\]\.\s*\((Minimise|Maximise)\)\s*(\w+)", block)
    return m.group(2).upper() if m else "BIC"


def _parse_parameters(block, sol):
    in_intercept = in_fixed = in_random = False
    current_rand = None

    for line in block.splitlines():
        if "INTERCEPTS"        in line: in_intercept, in_fixed, in_random = True,  False, False; continue
        if "FIXED PARAMETERS"  in line: in_intercept, in_fixed, in_random = False, True,  False; continue
        if "RANDOM PARAMETERS" in line: in_intercept, in_fixed, in_random = False, False, True;  continue
        if re.search(r"CORRELATION|GOODNESS|TRANSFORMED", line):
            in_intercept = in_fixed = in_random = False

        if in_intercept:
            m = re.match(
                r"\s{2}(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)",
                line
            )
            if m:
                sol["intercepts"].append({
                    "var":  m.group(1), "coeff": float(m.group(2)),
                    "se":   float(m.group(3)), "zval":  float(m.group(4)),
                    "pval": float(m.group(5)), "sig":   m.group(6).strip(),
                })

        if in_fixed:
            m = re.match(
                r"\s{2}(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)",
                line
            )
            if m:
                sol["fixed"].append({
                    "var":   m.group(1), "coeff": float(m.group(2)),
                    "se":    float(m.group(3)), "zval":  float(m.group(4)),
                    "pval":  float(m.group(5)), "sig":   m.group(6).strip(),
                })

        if in_random:
            m = re.match(
                r"\s{2}(\S+)\s+([a-z]{1,2})\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)",
                line
            )
            if m:
                current_rand = {
                    "var":     m.group(1), "dist":    m.group(2),
                    "mean":    float(m.group(3)), "se_mean": float(m.group(4)),
                    "zval":    float(m.group(5)), "pval":    float(m.group(6)),
                    "sig":     m.group(7).strip(),
                    "sd": None, "se_sd": None, "sig_sd": None,
                }
                sol["random"].append(current_rand)
                continue

            m = re.match(
                r"\s{2}sd\.(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)",
                line
            )
            if m and current_rand and current_rand["var"] == m.group(1):
                current_rand["sd"]    = float(m.group(2))
                current_rand["se_sd"] = float(m.group(3))
                current_rand["zval_sd"] = float(m.group(4))
                current_rand["pval_sd"] = float(m.group(5))
                current_rand["sig_sd"]= m.group(6).strip()


def _parse_correlations(block, sol):
    corr_section = re.search(r"CORRELATION MATRIX\s*[-]+\s*(.*?)[-]{10,}", block, re.DOTALL)
    if not corr_section:
        return
    lines = [l for l in corr_section.group(1).splitlines() if l.strip()]
    if not lines:
        return
    sol["corrvars"] = lines[0].split()
    matrix = {}
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        var  = parts[0]
        vals = re.findall(r"([-\d.]+)\s+\(([^)]+)\)", line)
        if vals:
            matrix[var] = vals
    sol["correlations"] = matrix


def parse_pert(filepath):
    iterations, bics, accepted, steps = [], [], [], []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                bics.append(float(parts[0]))
                accepted.append(parts[1].strip().lower() == "true")
                steps.append(int(parts[2].strip()))
                iterations.append(i + 1)
            except ValueError:
                continue

    best_bics, cur = [], float("inf")
    for b in bics:
        cur = min(cur, b)
        best_bics.append(cur)

    return iterations, bics, accepted, steps, best_bics


# ── Distribution helpers ──────────────────────────────────────────────────────

def compute_distribution(dist, mean, sd, n=600):
    dist   = dist.lower()
    abs_sd = abs(sd) if sd and sd != 0 else 1e-6

    if dist == "n":
        lo, hi = mean - 4*abs_sd, mean + 4*abs_sd
        xs = [lo + (hi-lo)*i/n for i in range(n+1)]
        ys = [math.exp(-0.5*((x-mean)/abs_sd)**2)/(abs_sd*math.sqrt(2*math.pi)) for x in xs]

    elif dist == "ln":
        mu, sigma = mean, abs_sd
        lo = max(1e-6, math.exp(mu - 4.0*sigma))
        hi = math.exp(mu + 4.0*sigma)
        xs = [lo+(hi-lo)*i/n for i in range(n+1)]
        ys = [math.exp(-0.5*((math.log(x)-mu)/sigma)**2)/(x*sigma*math.sqrt(2*math.pi))
              if x > 0 else 0 for x in xs]
           

    elif dist == "tn":
        lo = 0.0 if mean >= 0 else mean - 4*abs_sd
        hi = mean + 4*abs_sd if mean >= 0 else 0.0
        xs = [lo+(hi-lo)*i/n for i in range(n+1)]
        ys = [math.exp(-0.5*((x-mean)/abs_sd)**2)/(abs_sd*math.sqrt(2*math.pi)) for x in xs]

    elif dist == "u":
        hw  = abs_sd
        a, b = mean-hw, mean+hw
        pad  = hw * 0.15
        h    = 1.0/(b-a) if b != a else 1.0
        xs   = [a-pad, a-pad, a,   a,  b,  b,   b+pad, b+pad]
        ys   = [0,     0,     0,   h,  h,  0,   0,     0    ]

    elif dist == "t":
        hw  = abs_sd
        a   = mean - hw
        c   = mean + hw
        pad = hw * 0.15
        lo2, hi2 = a-pad, c+pad
        xs, ys = [], []
        for i in range(n+1):
            x = lo2 + (hi2-lo2)*i/n
            if x <= a or x >= c:
                y = 0.0
            elif x <= mean:
                y = 2*(x-a)/((c-a)*(mean-a)) if mean != a else 0
            else:
                y = 2*(c-x)/((c-a)*(c-mean)) if c != mean else 0
            xs.append(x); ys.append(y)
    else:
        xs, ys = [mean-1, mean, mean+1], [0, 1, 0]

    return xs, ys


def pct_negative(dist, mean, sd):
    dist   = dist.lower()
    abs_sd = abs(sd) if sd and sd != 0 else 1e-9

    if dist == "n":
        z = -mean / abs_sd
        return 0.5*(1 + math.erf(z/math.sqrt(2)))*100
    elif dist == "ln":
        return 0.0
    elif dist == "tn":
        return 100.0 if mean < 0 else 0.0
    elif dist == "u":
        hw   = abs_sd 
        a, b = mean-hw, mean+hw
        if b <= 0: return 100.0
        if a >= 0: return 0.0
        return (-a/(b-a))*100 if b != a else 50.0
    elif dist == "t":
        hw  = abs_sd
        a   = mean - hw
        c   = mean + hw
        if c <= 0: return 100.0
        if a >= 0: return 0.0
        if mean >= 0:
            neg = ((0-a)**2)/((c-a)*(mean-a)) if mean != a else 0
        else:
            pos = ((c-0)**2)/((c-a)*(c-mean)) if c != mean else 0
            neg = 1 - pos
        return min(100.0, max(0.0, neg*100))
    return 50.0

# ── Entry point ───────────────────────────────────────────────────────────────

def generate_dashboard(run_id, results_path=None, pert_path=None, siman_obj=None):
    rid          = str(run_id or getattr(siman_obj, 'idnum', 'run'))
    results_file = results_path or f"siman_results[{rid}].txt"
    pert_file    = pert_path    or f"siman_pert[{rid}].txt"
    output_file  = f"dashboard_{rid}.html"

    if not Path(results_file).exists():
        print(f"[Dashboard] Results file not found: {results_file}"); return
    if not Path(pert_file).exists():
        print(f"[Dashboard] Pert file not found: {pert_file}"); return

    print("[Dashboard] Parsing results ...")
    solutions, objective_name, n_alts = parse_results(results_file)
    if not solutions:
        print("[Dashboard] No Top solutions found."); return
    
    if siman_obj is not None:
        for i, sol in enumerate(solutions):
            if i < len(siman_obj.top_solutions):
                sol['grad_history'] = siman_obj.top_solutions[i].get('grad_history', None)
        gtol = getattr(siman_obj.param, 'gtol', 1e-5)
    else:
        gtol = 1e-5

    print("[Dashboard] Parsing convergence ...")
    iters, bics, accepted, steps, best_bics = parse_pert(pert_file)

    print("[Dashboard] Building HTML ...")
    html = build_html(solutions, iters, bics, accepted, steps, best_bics,
                      rid, objective_name, n_alts, gtol=gtol)

    Path(output_file).write_text(html, encoding="utf-8")
    print(f"[Dashboard] Saved → {output_file}")

# ── Estimation result parser ──────────────────────────────────────────────────

def parse_est_results(filepath, from_string=False):
    text = filepath if from_string else Path(filepath).read_text(encoding="utf-8").replace("\r\n", "\n")

    model_m = re.search(r"Model Summary:\s*(\S+)", text)
    conv_m  = re.search(r"WARNING.*[Cc]onvergence", text)

    sol = {
        "rank": 1, "model": model_m.group(1) if model_m else "Unknown",
        "converged": conv_m is None,
        "fixed": [], "random": [], "correlations": {}, "corrvars": [],
        "loglik": None, "aic": None, "bic": None, "adjlik": None,
        "predicted": [], "observed": [],
        "has_random": False, "has_bcvars": False,
        "gtol_ok": None, "ftol_ok": None, "gtol_val": None, "ftol_val": None,
        "obj_value": None, "objective": "BIC", "n_alts": 0,
    }

    # Shares
    obs_m  = re.search(r"Observed:\s*((?:[\d.]+\s+)+)",  text)
    pred_m = re.search(r"Predicted:\s*((?:[\d.]+\s+)+)", text)
    if obs_m:  sol["observed"]  = [float(x) for x in obs_m.group(1).split()]
    if pred_m: sol["predicted"] = [float(x) for x in pred_m.group(1).split()]
    sol["n_alts"] = len(sol["observed"]) or len(sol["predicted"])

    # GoF  — handles both separator styles: | and ;
    gof_m = re.search(
        r"LOGLIK\s*[=:]\s*(-?[\d.]+).*?AIC\s*[=:]\s*([\d.]+).*?BIC\s*[=:]\s*([\d.]+).*?ADJ\w*\s*(?:RATIO\s*)?[=:]\s*([\d.]+)",
        text, re.IGNORECASE
    )
    if gof_m:
        sol["loglik"] = float(gof_m.group(1))
        sol["aic"]    = float(gof_m.group(2))
        sol["bic"]    = float(gof_m.group(3))
        sol["adjlik"] = float(gof_m.group(4))

    # Parameters
    in_intercept = in_fixed = in_random = False
    current_rand = None
    for line in text.splitlines():
        if "INTERCEPTS"        in line: in_intercept, in_fixed, in_random = True,  False, False; continue
        if "FIXED PARAMETERS"  in line: in_intercept, in_fixed, in_random = False, True,  False; continue
        if "RANDOM PARAMETERS" in line: in_intercept, in_fixed, in_random = False, False, True;  continue
        if re.search(r"CORRELATION|GOODNESS|TRANSFORMED", line):
            in_intercept = in_fixed = in_random = False

        if in_intercept:
            m = re.match(
                r"\s{2}(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)",
                line
            )
            if m:
                sol["intercepts"].append({
                    "var":  m.group(1), "coeff": float(m.group(2)),
                    "se":   float(m.group(3)), "zval":  float(m.group(4)),
                    "pval": float(m.group(5)), "sig":   m.group(6).strip(),
                })
        
        if in_fixed:
            m = re.match(
                r"\s+(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)", line
            )
            if m:
                sol["fixed"].append({
                    "var": m.group(1), "coeff": float(m.group(2)),
                    "se":  float(m.group(3)), "zval":  float(m.group(4)),
                    "pval": float(m.group(5)), "sig":  m.group(6).strip(),
                })

        if in_random:
            m = re.match(
                r"\s+(\S+)\s+([a-z]{1,2})\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)", line
            )
            if m:
                current_rand = {
                    "var": m.group(1), "dist": m.group(2),
                    "mean": float(m.group(3)), "se_mean": float(m.group(4)),
                    "zval": float(m.group(5)), "pval":    float(m.group(6)),
                    "sig":  m.group(7).strip(),
                    "sd": None, "se_sd": None, "sig_sd": None,
                    "zval_sd": None, "pval_sd": None,
                }
                sol["random"].append(current_rand)
                continue
            m = re.match(
                r"\s+sd\.(\S+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+\(([^)]+)\)", line
            )
            if m and current_rand and current_rand["var"] == m.group(1):
                current_rand["sd"]      = float(m.group(2))
                current_rand["se_sd"]   = float(m.group(3))
                current_rand["zval_sd"] = float(m.group(4))
                current_rand["pval_sd"] = float(m.group(5))
                current_rand["sig_sd"]  = m.group(6).strip()

    _parse_correlations(text, sol)   # reuse existing helper
    sol["has_random"] = len(sol["random"]) > 0
    return sol

def build_est_html(sol, run_id, gtol=1e-5):
    """Reuses build_html with a single solution and no convergence data."""
    sol["rank"] = 1
    return build_html(
        solutions=[sol],
        iters=[], bics=[], accepted=[], steps=[], best_bics=[],
        run_id=run_id, objective_name="BIC", n_alts=sol["n_alts"],
        title="Est. Dashboard", show_convergence=True, gtol=gtol,
    )

def parse_est_results_str(text):
    """Same as parse_est_results but from string instead of file."""
    return parse_est_results(text, from_string=True)

def generate_est_dashboard(model=None, results_path=None, output_path=None):
    if model is not None:
        import io
        buf = io.StringIO()
        model.summarise(file=buf)
        sol = parse_est_results_str(buf.getvalue())
        sol['grad_history'] = getattr(model, 'grad_history', None)
        rid         = getattr(model, 'run_id', 'est')
        if not output_path:
            base = Path(f"dashboard_{rid}.html")
            counter = 1
            while base.exists():
                base = Path(f"dashboard_{rid}({counter}).html")
                counter += 1
            output_file = base
        else:
            output_file = output_path
    else:
        results_file = Path(results_path)
        if not results_file.exists():
            print(f"[Dashboard] File not found: {results_file}"); return
        sol         = parse_est_results(results_file)
        rid         = results_file.stem
        if not output_path:
            base = results_file.with_suffix(".html")
            counter = 1
            while base.exists():
                base = results_file.with_name(f"{results_file.stem}({counter}).html")
                counter += 1
            output_file = base
        else:
            output_file = output_path

    print("[Dashboard] Building HTML ...")
    html = build_est_html(sol, rid, gtol=getattr(model, 'gtol', 1e-5) if model else 1e-5)
    Path(output_file).write_text(html, encoding="utf-8")
    print(f"[Dashboard] Saved → {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_sa = sub.add_parser("sa",  help="SA search dashboard")
    p_sa.add_argument("--id",      required=True)
    p_sa.add_argument("--results", default=None)
    p_sa.add_argument("--pert",    default=None)

    p_est = sub.add_parser("est", help="Single estimation dashboard")
    p_est.add_argument("--results", required=True)
    p_est.add_argument("--output",  default=None)

    args = parser.parse_args()
    if args.cmd == "sa":
        generate_dashboard(args.id, args.results, args.pert)
    elif args.cmd == "est":
        generate_est_dashboard(args.results, args.output)
    else:
        parser.print_help()
