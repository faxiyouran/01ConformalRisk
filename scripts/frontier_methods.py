"""Historical frontier function bodies; extracted without algorithm changes."""
from fractions import Fraction as F
import statistics

def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError("STOP:" + code)


def q(row: dict[str, str], key: str) -> F:
    require(key in row and row[key] != "", "FIELD_MISSING_" + key)
    try:
        return F(row[key])
    except (ValueError, ZeroDivisionError, OverflowError):
        raise RuntimeError("STOP:INVALID_NUMERIC_" + key) from None


def bit(row: dict[str, str], key: str) -> int:
    require(row.get(key) in ("0", "1"), "INVALID_BINARY_" + key)
    return int(row[key])


def dec(x: F | int | float) -> str:
    return format(float(x), ".12g")


def ratio_text(x: F) -> str:
    return str(x)


def metric_point(rows: list[dict[str, str]], executed: list[int], cohort: str, family: str,
                 threshold_kind: str, threshold: str, threshold_exact: F, point_index: int) -> dict:
    n = len(rows)
    harms = [bit(r, "harm_tau_005") for r in rows]
    count = sum(executed)
    harmful = sum(h * e for h, e in zip(harms, executed))
    return {
        "point_id": f"{family}_{cohort}_{point_index:04d}", "cohort": cohort,
        "frontier": family, "threshold_type": threshold_kind, "threshold": threshold,
        "threshold_exact_fraction": str(threshold_exact), "N": n, "harmful_episodes": sum(harms),
        "executed_count": count, "harmful_executed_count": harmful,
        "ARR_fraction": str(F(count, n)), "ARR": dec(F(count, n)),
        "HERISK_fraction": str(F(harmful, n)), "HERISK": dec(F(harmful, n)),
        "_arr": F(count, n), "_herisk": F(harmful, n), "_execute": executed,
        "_threshold_sort": threshold_exact,
    }


def build_frontier(rows: list[dict[str, str]], cohort: str, family: str) -> list[dict]:
    n = len(rows)
    points = []
    if family == "STATIC":
        observed = sorted({q(r, "exposure_fraction") for r in rows})
        thresholds = [("boundary_none", "BELOW_MIN", observed[0] - 1)]
        thresholds += [("observed", str(value), value) for value in observed]
        thresholds += [("boundary_all", "ABOVE_MAX", observed[-1] + 1)]
        for i, (kind, label, cutoff) in enumerate(thresholds):
            executed = [int(q(r, "exposure_fraction") <= cutoff) for r in rows]
            points.append(metric_point(rows, executed, cohort, family, kind, label, cutoff, i))
    elif family == "RISK":
        score_values = sorted({F.from_float(float(r["guardian_score"])) for r in rows})
        thresholds = [("boundary_none", "BELOW_MIN", score_values[0] - 1)]
        thresholds += [("observed", format(float(value), ".17g"), value) for value in score_values]
        thresholds += [("boundary_all", "ABOVE_MAX", score_values[-1] + 1)]
        for i, (kind, label, cutoff) in enumerate(thresholds):
            executed = [int(F.from_float(float(r["guardian_score"])) < cutoff) for r in rows]
            points.append(metric_point(rows, executed, cohort, family, kind, label, cutoff, i))
    else:
        raise RuntimeError("STOP:UNKNOWN_FRONTIER_FAMILY")
    require(points[0]["executed_count"] == 0 and points[-1]["executed_count"] == n,
            "FRONTIER_BOUNDARY_FAILURE_" + family + "_" + cohort)
    return points


def is_dominated(p: dict, points: list[dict]) -> bool:
    return any(q["_arr"] >= p["_arr"] and q["_herisk"] <= p["_herisk"] and
               (q["_arr"] > p["_arr"] or q["_herisk"] < p["_herisk"]) for q in points)


def pareto(points: list[dict]) -> list[dict]:
    return [p for p in points if not is_dominated(p, points)]


def frontier_dominance(static_pareto: list[dict], risk_pareto: list[dict], cohort: str) -> dict:
    def covered(source: list[dict], target: list[dict]) -> int:
        return sum(any(t["_arr"] >= s["_arr"] and t["_herisk"] <= s["_herisk"]
                       for t in target) for s in source)
    static_by_risk = covered(static_pareto, risk_pareto)
    risk_by_static = covered(risk_pareto, static_pareto)
    if static_by_risk == len(static_pareto) and risk_by_static == len(risk_pareto):
        status = "EMPIRICAL_ENVELOPES_EQUIVALENT"
    elif static_by_risk == len(static_pareto):
        status = "RISK_FRONTIER_WEAKLY_DOMINATES_STATIC"
    elif risk_by_static == len(risk_pareto):
        status = "STATIC_FRONTIER_WEAKLY_DOMINATES_RISK"
    else:
        status = "PARTIAL_OVERLAP_NO_FULL_DOMINANCE"
    return {"cohort": cohort, "static_pareto_points": len(static_pareto),
            "risk_pareto_points": len(risk_pareto),
            "static_points_weakly_covered_by_risk": static_by_risk,
            "risk_points_weakly_covered_by_static": risk_by_static,
            "classification": status,
            "relation": "weak ARR >= and weak HERISK <=; equality counts as coverage"}


def matching(risk_pareto: list[dict], static_frontier: list[dict], metric: str,
             cohort: str) -> list[dict]:
    n = risk_pareto[0]["N"]
    primary_tol = F(1, n)
    secondary_tol = F("0.01") if cohort == "TEST" else F("0.005")
    result = []
    for index, risk in enumerate(risk_pareto):
        distance_key = "_arr" if metric == "ARR" else "_herisk"
        distances = [abs(risk[distance_key] - static[distance_key]) for static in static_frontier]
        nearest = min(distances)
        tied = [s for s, d in zip(static_frontier, distances) if d == nearest]
        if nearest <= primary_tol:
            status = "EXACT_DISCRETE_MATCH"
        elif nearest <= secondary_tol:
            status = "APPROXIMATE_MATCH"
        else:
            status = "UNMATCHED"
            tied = []
        if not tied:
            result.append({
                "match_id": f"{metric}_{cohort}_{index:04d}_UNMATCHED", "cohort": cohort,
                "risk_point_id": risk["point_id"], "risk_threshold_type": risk["threshold_type"],
                "risk_threshold": risk["threshold"], "risk_threshold_exact_fraction": risk["threshold_exact_fraction"],
                "static_point_id": "", "static_threshold_type": "", "static_threshold": "",
                "static_threshold_exact_fraction": "", "N": n,
                "ARR_risk": dec(risk["_arr"]), "ARR_static": "", "ARR_difference": "",
                "HERISK_risk": dec(risk["_herisk"]), "HERISK_static": "", "HERISK_difference": "",
                "match_distance": str(nearest), "exact_or_approximate": status,
                "_risk_execute": risk["_execute"], "_static_execute": None,
                "_risk": risk, "_static": None,
            })
            continue
        for tie_index, static in enumerate(tied):
            arr_difference = risk["_arr"] - static["_arr"]
            herisk_difference = risk["_herisk"] - static["_herisk"]
            result.append({
                "match_id": f"{metric}_{cohort}_{index:04d}_{tie_index:02d}", "cohort": cohort,
                "risk_point_id": risk["point_id"], "risk_threshold_type": risk["threshold_type"],
                "risk_threshold": risk["threshold"], "risk_threshold_exact_fraction": risk["threshold_exact_fraction"],
                "static_point_id": static["point_id"], "static_threshold_type": static["threshold_type"],
                "static_threshold": static["threshold"], "static_threshold_exact_fraction": static["threshold_exact_fraction"],
                "N": n, "ARR_risk": dec(risk["_arr"]), "ARR_static": dec(static["_arr"]),
                "ARR_difference": dec(arr_difference), "HERISK_risk": dec(risk["_herisk"]),
                "HERISK_static": dec(static["_herisk"]), "HERISK_difference": dec(herisk_difference),
                "match_distance": str(nearest), "exact_or_approximate": status,
                "_risk_execute": risk["_execute"], "_static_execute": static["_execute"],
                "_risk": risk, "_static": static,
            })
    return result


def serializable_points(points: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]


def serializable_matches(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def summarize_matches(matches: list[dict], metric: str, cohort: str) -> dict:
    eligible = [m for m in matches if m["exact_or_approximate"] != "UNMATCHED"]
    diffs = [F(m["HERISK_difference"] if metric == "ARR" else m["ARR_difference"])
             for m in eligible]
    exact = sum(m["exact_or_approximate"] == "EXACT_DISCRETE_MATCH" for m in eligible)
    approx = sum(m["exact_or_approximate"] == "APPROXIMATE_MATCH" for m in eligible)
    risk_better = sum(d < 0 for d in diffs) if metric == "ARR" else sum(d > 0 for d in diffs)
    risk_worse = sum(d > 0 for d in diffs) if metric == "ARR" else sum(d < 0 for d in diffs)
    ties = len(diffs) - risk_better - risk_worse
    median = statistics.median(diffs) if diffs else None
    mean = sum(diffs, F(0)) / len(diffs) if diffs else None
    risk_point_count = len({m["risk_point_id"] for m in matches})
    matched_risk_point_count = len({m["risk_point_id"] for m in eligible})
    return {"cohort": cohort, "matching_dimension": metric, "risk_pareto_points": risk_point_count,
            "matched_risk_points": matched_risk_point_count,
            "matched_points_including_nearest_ties": len(eligible), "comparison_unit": "nearest-tie pair",
            "exact_matches": exact,
            "approximate_matches": approx, "unmatched_risk_points": sum(
                m["exact_or_approximate"] == "UNMATCHED" for m in matches),
            "risk_advantage_count": risk_better, "risk_disadvantage_count": risk_worse,
            "ties": ties,
            "risk_advantage_proportion": dec(F(risk_better, len(diffs))) if diffs else "",
            "risk_advantage_definition": "HERISK_risk < HERISK_static" if metric == "ARR" else "ARR_risk > ARR_static",
            "mean_difference_risk_minus_static": dec(mean) if mean is not None else "",
            "median_difference_risk_minus_static": dec(F(median)) if median is not None else "",
            "largest_risk_advantage": dec(min(diffs)) if metric == "ARR" and diffs else
                (dec(max(diffs)) if diffs else ""),
            "largest_risk_disadvantage": dec(max(diffs)) if metric == "ARR" and diffs else
                (dec(min(diffs)) if diffs else ""),
            "pattern": ("MIXED" if risk_better and risk_worse else
                        "RISK_BETTER_WITH_TIES" if risk_better and ties else
                        "RISK_BETTER_AT_ALL_MATCHED_POINTS" if risk_better else
                        "STATIC_BETTER_WITH_TIES" if risk_worse and ties else
                        "STATIC_BETTER_AT_ALL_MATCHED_POINTS" if risk_worse else "TIED_OR_NO_MATCHES")}

