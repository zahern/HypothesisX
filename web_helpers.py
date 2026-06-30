import os
import re
import json

import search_librium_helpers as sl

DIR = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(DIR, "frontend", "dist")

#------------------------------------ Dashboard Payload assembly ------------------------------------

def build_dist_data(solutions):
    """Per-rank list of plottable random-parameter distributions.

    Mirrors the ``dist_data`` block that used to live inside
    ``dashboard_3.build_html``.
    """
    dist_data = {}
    for sol in solutions:
        rank = sol["rank"]
        dist_data[rank] = []
        for p in sol["random"]:
            if p["sd"] is None:
                continue
            xs, ys = sl.compute_distribution(p["dist"], p["mean"], p["sd"])
            pct_neg = sl.pct_negative(p["dist"], p["mean"], p["sd"])
            dist_data[rank].append({
                "var": p["var"], "dist": p["dist"],
                "mean": p["mean"], "sd": p["sd"],
                "xs": xs, "ys": ys,
                "pct_neg": round(pct_neg, 1),
                "pct_pos": round(100 - pct_neg, 1),
                "sig": p["sig"],
                "sig_sd": p.get("sig_sd", "-"),
                "zval_sd": p.get("zval_sd"),
                "pval_sd": p.get("pval_sd"),
            })

    return dist_data

def build_payload(results_file, pert_file, run_id, gtol=1e-5):
    """Parse the siman files and return the full dashboard payload as a dict.

    The shape matches what the old HTML embedded as ``DATA`` / ``DIST_DATA`` /
    ``ALT_LBL`` plus a few presentation flags the sidebar used.
    """
    solutions, objective_name, n_alts = sl.parse_results(results_file)
    #solutions = solutions[:1]

    if not solutions:
        raise HTTPException(status_code=422, detail="No Top solutions found in results file.")

    if pert_file and os.path.exists(pert_file):
        iters, bics, accepted, steps, best_bics = sl.parse_pert(pert_file)
    else:
        iters, bics, accepted, steps, best_bics = [], [], [], [], []

    dist_data = build_dist_data(solutions)
    alt_labels = [f"Alt {i + 1}" for i in range(n_alts)]

    has_random = any(s["has_random"] for s in solutions)
    has_corvars = any(len(s["corrvars"]) > 0 for s in solutions)

    # json.loads(json.dumps(..., default=str)) keeps the same str-coercion the
    # old build_html applied (e.g. None stays None, everything else stringifies
    # cleanly) while returning a plain dict for FastAPI to serialise.
    data = json.loads(json.dumps({
        "runId": run_id,
        "solutions": solutions,
        "convergence": {
            "iterations": iters, "bics": bics, "accepted": accepted,
            "steps": steps, "best_bics": best_bics,
            "objective": objective_name, "gtol": gtol,
        },
        "distData": dist_data,
        "altLabels": alt_labels,
        "objective": objective_name,
        "nAlts": n_alts,
        "draws":                solutions[0]["draws"],
        "individuals":          solutions[0]["individuals"],
        "choicesPerIndividual": solutions[0]["choices_per_individual"],
        "totalChoices":         solutions[0]["total_choices"],
        "flags": {
            "hasRandom": has_random,
            "hasCorvars": has_corvars,
            "hasConvergence": len(iters) > 0,
        },
    }, default=str))

    return data

#------------------------------------ Runs stuff ------------------------------------

def discover_runs():
    """Find run ids from ``siman_results[<id>].txt`` files in the project dir.

    Note the literal ``[`` / ``]`` in the filenames are glob metacharacters, so
    we scan the directory and match with a regex rather than globbing.
    """
    runs = []
    for name in sorted(os.listdir(DIR)):
        m = re.match(r"siman_results\[(.+)\]\.txt$", name)
        if m:
            runs.append(m.group(1))
    return runs


def paths_for(run_id):
    results = os.path.join(DIR, f"siman_results[{run_id}].txt")
    pert = os.path.join(DIR, f"siman_pert[{run_id}].txt")
    return results, pert
