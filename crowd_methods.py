"""Serializable method settings shared by computation and plot metadata.

This module has no OpenSpiel dependency. ``beta`` in BPI's legend denotes the
inverse temperature; its OpenSpiel constructor argument is named ``lr``.
"""

from copy import deepcopy
import math

METHOD_SETTINGS = {
    "fp": {"class": "fictitious_play", "parameters": {},
           "label": "Fictitious play", "row_label": "Fictitious play"},
    "omd_0p1": {"class": "mirror_descent", "parameters": {"lr": 0.1},
                "label": r"OMD ($\eta=0.1$)", "row_label": r"OMD ($\eta=0.1$)"},
    "omd_0p01": {"class": "mirror_descent", "parameters": {"lr": 0.01},
                 "label": r"OMD ($\eta=0.01$)", "row_label": r"OMD ($\eta=0.01$)"},
    "fixed_point": {"class": "fixed_point", "parameters": {},
                    "label": "Fixed point", "row_label": "Fixed point"},
    "damped_fp": {"class": "fictitious_play", "parameters": {"lr": 0.01},
                  "label": r"Damped fixed point ($\alpha=0.01$)", "row_label": "Damped fixed point"},
    "softmax_fixed_point": {"class": "fixed_point", "parameters": {"temperature": 0.1},
                            "label": r"Softmax fixed point ($\tau=0.1$)",
                            "row_label": "Softmax fixed point"},
    "softmax_fp": {"class": "fictitious_play", "parameters": {"temperature": 0.1},
                   "label": r"Softmax fictitious play ($\tau=0.1$)", "row_label": "Softmax fictitious play"},
    "bpi": {"class": "boltzmann_policy_iteration", "parameters": {"lr": 0.1},
            "label": r"BPI ($\beta=0.1$)", "row_label": r"BPI ($\beta=0.1$)"},
}


def resolve_settings(methods, overrides=None):
    """Resolve explicit numeric hyperparameters without changing method families."""
    overrides = overrides or {}
    if set(overrides) - set(METHOD_SETTINGS):
        raise ValueError("Unknown method in method_parameters")
    symbols = {"omd_0p1": r"\eta", "omd_0p01": r"\eta", "damped_fp": r"\alpha",
               "softmax_fixed_point": r"\tau", "softmax_fp": r"\tau", "bpi": r"\beta"}
    resolved = {}
    for method in methods:
        settings = deepcopy(METHOD_SETTINGS[method])
        parameters = settings["parameters"]
        supplied = overrides.get(method, {})
        if set(supplied) - set(parameters):
            raise ValueError(f"Unsupported parameters for {method}: {sorted(supplied)}")
        for key, value in supplied.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{method}.{key} must be finite and positive")
            if method == "damped_fp" and value > 1:
                raise ValueError("Damping weight must not exceed 1")
            parameters[key] = value
        if parameters:
            value = next(iter(parameters.values()))
            name = settings["label"].split(" ($", 1)[0]
            settings["label"] = f"{name} (${symbols[method]}={value:g}$)"
            if method in ("omd_0p1", "omd_0p01", "bpi"):
                settings["row_label"] = settings["label"]
        if settings["class"] == "fictitious_play":
            settings["averaging_weight"] = parameters.get("lr", "1/(k+1), update k>=1")
        resolved[method] = settings
    return resolved
