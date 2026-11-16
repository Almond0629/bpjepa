"""Paired contrasts for the spotting analyses: same shared match draws in every arm, interaction computed within each draw."""
import numpy as np

ARMS = ("R0", "U", "EB", "EB0", "COMEDIAN")
INTERACTION = "(EB-EB0)-(U-R0)"
CONTRASTS = ("EB0-R0", "EB0-U", "EB-EB0", INTERACTION, "U-R0", "EB-R0", "EB-U", "COMEDIAN-R0", "COMEDIAN-EB0")
# Paper names of the result-file keys.
PAPER_NAMES = {"R0": "R", "EB0": "BP-JEPA", "U": "TC", "EB": "BP-JEPA+TC", "COMEDIAN": "COMEDIAN Stage 1"}
ALIASES = {"R": "R0", "BP": "EB0", "BP-JEPA": "EB0", "TC": "U", "BPTC": "EB", "BP-JEPA+TC": "EB", **{a: a for a in ARMS}}


def pct(v):
    return np.quantile(v, [.025, .975], method="linear").tolist()


def difference(values, name):
    if name == INTERACTION:
        return (values["EB"] - values["EB0"]) - (values["U"] - values["R0"])
    a, b = name.split("-")
    return values[a] - values[b]


def available(arms, names=CONTRASTS):
    def needs(name):
        return ("EB", "EB0", "U", "R0") if name == INTERACTION else tuple(name.split("-"))
    return tuple(n for n in names if all(a in arms for a in needs(n)))


def contrasts(point, boot, names=None):
    names = available(point) if names is None else names
    return {name: {"difference_percentage_points": 100 * difference(point, name),
                   "marginal95": pct(difference(boot, name) * 100), "descriptive": True} for name in names}


def summarize(point, boot, seeds):
    arms = tuple(a for a in ARMS if all(a in point[s] for s in seeds))
    names = available(arms)
    per_seed = {str(s): {"arms": {a: {"tight_avg_mAP_percentage": 100 * point[s][a],
                                      "marginal95": (np.array(pct(boot[s][a])) * 100).tolist()} for a in arms},
                         "contrasts": contrasts(point[s], boot[s], names)} for s in seeds}
    mean_arms = {a: {"tight_avg_mAP_percentage": float(np.mean([100 * point[s][a] for s in seeds])),
                     "marginal95": pct(np.mean([boot[s][a] * 100 for s in seeds], axis=0))} for a in arms}
    mean_contrasts = {}
    for name in names:
        diffs = [100 * difference(point[s], name) for s in seeds]
        mean_boot = np.mean([difference(boot[s], name) * 100 for s in seeds], axis=0)
        mean_contrasts[name] = {"per_seed_differences_percentage_points": diffs, "range": [min(diffs), max(diffs)],
                                "positive_seeds": int(sum(d > 0 for d in diffs)),
                                "mean_percentage_points": float(np.mean(diffs)),
                                "mean_marginal95": pct(mean_boot), "descriptive": True}
    return {"seeds": list(seeds), "per_seed": per_seed, "mean_arms": mean_arms, "mean_contrasts": mean_contrasts,
            "reading": "descriptive; shared paired match draws, marginal95 intervals; mean intervals resample matches, not head seeds or pretraining runs"}


DISCLOSURES = ["COMEDIAN Stage 1 is spatial pretraining only (no Stage 2); full three-stage COMEDIAN is not compared",
               "COMEDIAN recipe differs from R/TC/BP-JEPA: SCE objective, global batch 512, 322,300 updates, fp16",
               "one pretraining run per arm; match intervals exclude head-seed and pretraining-run uncertainty",
               "validation only; no test access",
               "frozen seed 42 ran on RTX 5090s; seeds 203/666 on RTX A6000s; pairing holds within seed",
               "BP-JEPA contrasts were analysed post hoc after its point estimates were seen",
               "lambda_b 1.6 was calibrated against the TC term and not recalibrated for lambda_TC 0",
               "marginal 95% intervals do not provide simultaneous family coverage"]
