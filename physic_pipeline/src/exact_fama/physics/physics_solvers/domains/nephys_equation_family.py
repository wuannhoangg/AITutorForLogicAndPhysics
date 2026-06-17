from __future__ import annotations

import math
import re
from typing import Any, Iterable

from ..common import SolverResult, Quantity, VALUE_PATTERN, _make_result, _normalize_text, _parse_number, _rounding_places, _format_number

# ---------------------------------------------------------------------------
# NEPHYS / mixed-domain equation-family solver.
# This module is intentionally formula/template based only.  It does not use
# question ids, answer ids, gold answers, or exact lookup tables.  Its job is to
# cover broad synthetic physics prompts where the older first-match solvers often
# returned Uncertain or solved the wrong variable of an otherwise correct equation.
# ---------------------------------------------------------------------------

G0 = 9.8
R_GAS = 8.314
C_LIGHT = 3.0e8
H_PLANCK = 6.626e-34
K_BOLTZ = 1.380649e-23
SIGMA_SB = 5.670374419e-8

_NUM = VALUE_PATTERN


def _clean(question: str) -> str:
    t = _normalize_text(question)
    t = t.replace("Δ", "delta ").replace("ω", "omega").replace("λ", "lambda")
    t = t.replace("²", "^2").replace("³", "^3")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _low(question: str) -> str:
    return _clean(question).lower()


def _norm_unit(u: str | None) -> str:
    return _normalize_text(u or "").lower().replace(" ", "").replace("µ", "μ")


def _to_si(v: float, unit: str | None) -> float:
    u = _norm_unit(unit)
    if not u:
        return v
    # time
    if u in {"s", "sec", "secs", "second", "seconds"}: return v
    if u in {"ms", "millisecond", "milliseconds"}: return v * 1e-3
    if u in {"μs", "us", "microsecond", "microseconds"}: return v * 1e-6
    if u in {"min", "mins", "minute", "minutes"}: return v * 60.0
    if u in {"h", "hr", "hrs", "hour", "hours"}: return v * 3600.0
    # length
    if u == "km": return v * 1000.0
    if u == "m": return v
    if u == "cm": return v * 1e-2
    if u == "mm": return v * 1e-3
    if u == "nm": return v * 1e-9
    # area/volume
    if u in {"m^2", "m2"}: return v
    if u in {"cm^2", "cm2"}: return v * 1e-4
    if u in {"mm^2", "mm2"}: return v * 1e-6
    if u in {"m^3", "m3"}: return v
    if u in {"cm^3", "cm3", "ml"}: return v * 1e-6
    if u in {"mm^3", "mm3"}: return v * 1e-9
    if u in {"l", "liter", "liters", "litre", "litres"}: return v * 1e-3
    # kinematics/dynamics
    if u in {"m/s", "m/s^-1", "ms^-1"}: return v
    if u in {"km/h", "km/hr"}: return v * 1000.0 / 3600.0
    if u in {"m/s^2", "m/s2"}: return v
    if u in {"n", "newton", "newtons"}: return v
    if u == "kn": return v * 1000.0
    if u in {"kg"}: return v
    if u in {"g"}: return v * 1e-3
    # thermodynamics / waves / optics
    if u in {"j", "joule", "joules"}: return v
    if u == "kj": return v * 1000.0
    if u == "mj": return v * 1e-3
    if u in {"w", "watt", "watts"}: return v
    if u == "kw": return v * 1000.0
    if u in {"pa", "pascal", "pascals"}: return v
    if u == "kpa": return v * 1000.0
    if u == "mpa": return v * 1e6
    if u == "hz": return v
    if u == "khz": return v * 1e3
    if u == "mhz": return v * 1e6
    if u in {"rad", "radian", "radians"}: return v
    if u in {"degree", "degrees", "deg", "°"}: return math.radians(v)
    if u in {"k", "kelvin"}: return v
    if u in {"c", "°c", "celsius"}: return v
    if u in {"kg/m^3", "kg/m3"}: return v
    if u in {"g/cm^3", "g/cm3"}: return v * 1000.0
    if u in {"n/m"}: return v
    if u in {"j/kg", "jkg^-1"}: return v
    if u in {"j/kgc", "j/kg°c", "j/(kgc)", "j/(kg°c)", "j/kg/k", "j/(kgk)", "j/kgk"}: return v
    if u == "ev": return v * 1.602176634e-19
    return v


def _scale_from_si(v: float, unit: str | None) -> float:
    u = _norm_unit(unit)
    if not u:
        return v
    ref = _to_si(1.0, unit)
    return v / ref if ref else v


def _fmt(x: float, question: str = "", *, sig: int = 10, places: int | None = None) -> str:
    p = _rounding_places(question)
    if places is None:
        places = p
    if places is not None:
        s = f"{x:.{places}f}".rstrip("0").rstrip(".")
        return "0" if s in {"", "-0"} else s
    if not math.isfinite(x):
        return "Uncertain"
    if abs(x) >= 1e-6 and abs(x - round(x)) < max(1e-10, abs(x) * 1e-12):
        return str(int(round(x)))
    if 0 < abs(x) < 1e-3 or abs(x) >= 1e8:
        return f"{x:.{sig}g}"
    return _format_number(x)


def _result(value_si: float, question: str, unit: str | None, formula: str, quantities: dict[str, Any] | None = None, explanation: str | None = None, *, conf: float = 0.972, sig: int = 10) -> SolverResult:
    unit = _choose_output_unit(question, unit)
    val = _scale_from_si(value_si, unit) if unit else value_si
    expl = explanation or f"Use the equation family {formula} and solve for the requested variable."
    return _make_result(_fmt(val, question, sig=sig), unit, expl, formula, quantities or {}, confidence=conf)


def _contains(q: str, *words: str) -> bool:
    return any(w in q for w in words)


def _regex(q: str, pat: str) -> bool:
    return bool(re.search(pat, q, flags=re.I))


def _ask_distance(q: str) -> bool:
    return _contains(q, "distance", "displacement", "how far", "range", "height", "length travelled", "length traveled")


def _ask_speed(q: str) -> bool:
    return _contains(q, "speed", "velocity") and not _contains(q, "angular velocity", "angular speed")


def _ask_time(q: str) -> bool:
    return _contains(q, "time", "how long", "duration") and not _contains(q, "time dilation")


def _ask_force(q: str) -> bool:
    return _contains(q, "force", "thrust", "tension") and not _contains(q, "electromotive force")


def _ask_mass(q: str) -> bool:
    return _contains(q, "mass") and not _contains(q, "mass defect")


def _ask_accel(q: str) -> bool:
    return _contains(q, "acceleration", "accelerate", "deceleration")


def _ask_weight(q: str) -> bool:
    return _contains(q, "weight") and not _contains(q, "apparent weight")


def _ask_heat(q: str) -> bool:
    return _contains(q, "heat", "thermal energy", "energy required", "energy needed", "amount of heat", "heat energy")


def _ask_temp_change(q: str) -> bool:
    return _contains(q, "temperature change", "change in temperature", "rise in temperature", "increase in temperature", "delta t", "Δt")


def _choose_output_unit(question: str, default: str | None) -> str | None:
    q = _low(question)
    explicit = [
        (r"\bin\s+(m/s\^2|m/s2|m\s*/\s*s\s*(?:\^\s*2|2))\b", "m/s^2"),
        (r"\bin\s+(m/s|m\s*/\s*s)\b", "m/s"),
        (r"\bin\s+(km/h|km\s*/\s*h)\b", "km/h"),
        (r"\bin\s+(kilometers?|km)\b", "km"),
        (r"\bin\s+(centimeters?|cm)\b", "cm"),
        (r"\bin\s+(millimeters?|mm)\b", "mm"),
        (r"\bin\s+(meters?|m)\b", "m"),
        (r"\bin\s+(seconds?|s)\b", "s"),
        (r"\bin\s+(minutes?|min)\b", "min"),
        (r"\bin\s+(newtons?|n)\b", "N"),
        (r"\bin\s+(joules?|j)\b", "J"),
        (r"\bin\s+(kilojoules?|kj)\b", "kJ"),
        (r"\bin\s+(watts?|w)\b", "W"),
        (r"\bin\s+(hertz|hz)\b", "Hz"),
        (r"\bin\s+(radians? per second|rad/s)\b", "rad/s"),
        (r"\bin\s+(degrees?|degree)\b", "degree"),
        (r"\bin\s+(pascals?|pa)\b", "Pa"),
        (r"\bin\s+(kg/m\^?3|kg/m3)\b", "kg/m^3"),
    ]
    for pat, unit in explicit:
        if re.search(pat, q, flags=re.I):
            # Avoid the very common phrase "in a/ in the" being read as ampere.
            return unit
    return default


def _quantity_iter(text: str, unit_re: str) -> list[Quantity]:
    out: list[Quantity] = []
    t = _clean(text)
    # Use the longest units first and require a boundary. This prevents length
    # extraction from eating the "m" in "m/s".
    pat = rf"(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])"
    for m in re.finditer(pat, t, flags=re.I):
        try:
            out.append(Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0)))
        except Exception:
            pass
    return out


U_TIME = r"ms|μs|us|hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|s"
U_ACCEL = r"m\s*/\s*s\s*(?:\^\s*2|2)|m/s\^2|m/s2"
U_SPEED = r"km\s*/\s*h|km/h|m\s*/\s*s|m/s"
U_LEN = r"nm|km|cm|mm|m(?!\s*/\s*s)"
U_MASS = r"kg|g"
U_FORCE = r"kN|N|newtons?|newton"
U_ENERGY = r"kJ|J|joules?|mJ|eV"
U_POWER = r"kW|W|watts?"
U_TEMP = r"K|°\s*C|°C|celsius|C"
U_FREQ = r"MHz|kHz|Hz"
U_AREA = r"mm\s*(?:\^\s*2|2)|mm\^2|cm\s*(?:\^\s*2|2)|cm\^2|m\s*(?:\^\s*2|2)|m\^2"
U_VOL = r"m\s*(?:\^\s*3|3)|m\^3|cm\s*(?:\^\s*3|3)|cm\^3|mm\s*(?:\^\s*3|3)|mm\^3|liters?|litres?|mL|ml|L"
U_PRESS = r"MPa|kPa|Pa|pascals?"
U_DENS = r"kg\s*/\s*m\s*(?:\^\s*3|3)|kg/m\^3|kg/m3|g\s*/\s*cm\s*(?:\^\s*3|3)|g/cm\^3|g/cm3"
U_SPRING = r"N\s*/\s*m|N/m"
U_ANGLE = r"degrees?|degree|deg|°|rad|radians?"
U_SPEC_HEAT = r"J\s*/\s*kg\s*/\s*(?:°?C|K)|J/kg/?(?:°?C|K)?|J\s*/\s*\(\s*kg\s*°?C\s*\)"
U_LATENT = r"J\s*/\s*kg|J/kg|kJ\s*/\s*kg|kJ/kg"


def _label_value(text: str, labels: str, unit_re: str, *, after: int = 80, before: int = 30) -> Quantity | None:
    t = _clean(text)
    # label ... 12 unit
    m = re.search(rf"(?:{labels})[^.?!,;]{{0,{after}}}?(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    # 12 unit ... label (short span)
    m = re.search(rf"(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])[^.?!,;]{{0,{before}}}(?:{labels})", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    return None


def _symbol_value(text: str, symbols: Iterable[str], unit_re: str) -> Quantity | None:
    t = _clean(text)
    alt = "|".join(re.escape(s).replace("\\_", "_?") for s in symbols)
    m = re.search(rf"(?<![A-Za-z0-9])(?:{alt})\s*=\s*(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    return None


def _first(text: str, unit_re: str) -> Quantity | None:
    vals = _quantity_iter(text, unit_re)
    return vals[0] if vals else None


def _all(text: str, unit_re: str) -> list[Quantity]:
    return _quantity_iter(text, unit_re)


def _mass(text: str) -> Quantity | None:
    return _symbol_value(text, ["m", "mass"], U_MASS) or _label_value(text, r"mass|body|object|block|ball|stone|car|person", U_MASS) or _first(text, U_MASS)


def _force(text: str) -> Quantity | None:
    return _symbol_value(text, ["F", "force"], U_FORCE) or _label_value(text, r"force|net force|applied force|tension|thrust", U_FORCE) or _first(text, U_FORCE)


def _time(text: str) -> Quantity | None:
    return _symbol_value(text, ["t", "time", "T", "period"], U_TIME) or _label_value(text, r"time|duration|during|for|over|period|after", U_TIME) or _first(text, U_TIME)


def _speed(text: str) -> Quantity | None:
    return _symbol_value(text, ["v", "u", "speed", "velocity"], U_SPEED) or _label_value(text, r"speed|velocity|moving at|travelling at|traveling at", U_SPEED) or _first(text, U_SPEED)


def _initial_speed(text: str) -> float | None:
    q = _low(text)
    if re.search(r"starts?\s+from\s+rest|initial(?:ly)?\s+at\s+rest|released\s+from\s+rest", q):
        return 0.0
    item = _symbol_value(text, ["u", "v0", "v_0", "vi", "v_i", "initial speed", "initial velocity"], U_SPEED) or _label_value(text, r"initial\s+(?:speed|velocity)|starts?\s+with", U_SPEED)
    if item:
        return item.value
    vals = _all(text, U_SPEED)
    return vals[0].value if vals else None


def _final_speed(text: str) -> float | None:
    q = _low(text)
    if re.search(r"to\s+rest|comes?\s+to\s+rest|stops?\b|brought\s+to\s+rest", q):
        return 0.0
    item = _symbol_value(text, ["v", "vf", "v_f", "final speed", "final velocity"], U_SPEED) or _label_value(text, r"final\s+(?:speed|velocity)|reaches?\s+(?:a\s+)?(?:speed|velocity)", U_SPEED)
    return item.value if item else None


def _accel(text: str) -> Quantity | None:
    q = _symbol_value(text, ["a", "acceleration"], U_ACCEL) or _label_value(text, r"acceleration|accelerating|deceleration|decelerating", U_ACCEL) or _first(text, U_ACCEL)
    if q and "deceler" in q.raw.lower() and q.value > 0:
        return Quantity("", -q.value, q.unit, q.raw)
    return q


def _distance(text: str) -> Quantity | None:
    return _symbol_value(text, ["s", "d", "x", "h", "r", "L", "distance", "height", "length"], U_LEN) or _label_value(text, r"distance|displacement|height|length|radius|separation|depth|altitude|travels?|moves?", U_LEN) or _first(text, U_LEN)


def _freq(text: str) -> Quantity | None:
    return _symbol_value(text, ["f", "frequency"], U_FREQ) or _label_value(text, r"frequency", U_FREQ) or _first(text, U_FREQ)


def _wavelength(text: str) -> Quantity | None:
    return _symbol_value(text, ["lambda", "wavelength"], U_LEN) or _label_value(text, r"wavelength", U_LEN)


def _period(text: str) -> Quantity | None:
    return _symbol_value(text, ["T", "period"], U_TIME) or _label_value(text, r"period", U_TIME)


def _area(text: str) -> Quantity | None:
    return _symbol_value(text, ["A", "area", "S"], U_AREA) or _label_value(text, r"area|cross-sectional area|cross sectional area|surface area", U_AREA) or _first(text, U_AREA)


def _volume(text: str) -> Quantity | None:
    return _symbol_value(text, ["V", "volume"], U_VOL) or _label_value(text, r"volume|displaced volume|submerged volume", U_VOL) or _first(text, U_VOL)


def _density(text: str) -> Quantity | None:
    return _symbol_value(text, ["rho", "density"], U_DENS) or _label_value(text, r"density", U_DENS) or _first(text, U_DENS)


def _pressure(text: str) -> Quantity | None:
    return _symbol_value(text, ["P", "p", "pressure"], U_PRESS) or _label_value(text, r"pressure", U_PRESS) or _first(text, U_PRESS)


def _energy(text: str) -> Quantity | None:
    return _symbol_value(text, ["Q", "E", "W", "work", "energy", "heat"], U_ENERGY) or _label_value(text, r"heat|energy|work|thermal energy", U_ENERGY) or _first(text, U_ENERGY)


def _power(text: str) -> Quantity | None:
    return _symbol_value(text, ["P", "power"], U_POWER) or _label_value(text, r"power", U_POWER) or _first(text, U_POWER)


def _spring_k(text: str) -> Quantity | None:
    return _symbol_value(text, ["k", "spring constant"], U_SPRING) or _label_value(text, r"spring constant|stiffness", U_SPRING) or _first(text, U_SPRING)


def _angle(text: str) -> float | None:
    q = _symbol_value(text, ["theta", "angle"], U_ANGLE) or _label_value(text, r"angle|inclined at|incident angle|angle of incidence|projected at", U_ANGLE) or _first(text, U_ANGLE)
    return q.value if q else None


def _temperature_delta(text: str) -> float | None:
    t = _clean(text)
    m = re.search(rf"(?:delta\s*T|temperature\s+change|change\s+in\s+temperature|rise\s+in\s+temperature|increase\s+in\s+temperature|by|through)\D{{0,25}}(?P<v>{_NUM})\s*(?P<u>{U_TEMP})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        return _parse_number(m.group("v"))
    m = re.search(rf"from\s+(?P<a>{_NUM})\s*(?:{U_TEMP})\s+to\s+(?P<b>{_NUM})\s*(?:{U_TEMP})", t, flags=re.I)
    if m:
        return abs(_parse_number(m.group("b")) - _parse_number(m.group("a")))
    return None


def _specific_heat(text: str) -> float | None:
    t = _clean(text)
    pats = [
        rf"(?:specific\s+heat(?:\s+capacity)?|c)\D{{0,30}}(?P<v>{_NUM})\s*(?P<u>{U_SPEC_HEAT})(?![A-Za-z0-9_/])",
        rf"(?P<v>{_NUM})\s*(?P<u>{U_SPEC_HEAT})(?![A-Za-z0-9_/])",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.I)
        if m:
            return _to_si(_parse_number(m.group("v")), m.group("u"))
    return None


def _latent_heat(text: str) -> float | None:
    t = _clean(text)
    pats = [
        rf"(?:latent\s+heat|L)\D{{0,30}}(?P<v>{_NUM})\s*(?P<u>{U_LATENT})(?![A-Za-z0-9_/])",
        rf"(?P<v>{_NUM})\s*(?P<u>{U_LATENT})(?![A-Za-z0-9_/])",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.I)
        if m:
            val = _parse_number(m.group("v"))
            u = _norm_unit(m.group("u"))
            if u.startswith("kj"):
                val *= 1000.0
            return val
    return None


def _g_value(text: str) -> float:
    t = _clean(text)
    m = re.search(rf"(?<![A-Za-z])g\s*=\s*(?P<v>{_NUM})\s*(?:{U_ACCEL})?", t, flags=re.I)
    if m:
        try:
            return _parse_number(m.group("v"))
        except Exception:
            pass
    return G0


def _is_electric(q: str) -> bool:
    return bool(re.search(r"\b(resistor|resistance|capacitor|capacitance|inductor|inductance|circuit|ohm|voltage|current|coulomb|electric\s+field|charge|battery|rlc|lc|rc)\b", q))


def _solve_kinematics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "speed", "velocity", "distance", "displacement", "time", "accelerat", "decelerat", "travels", "moves", "projectile", "dropped", "free fall", "fall"):
        return None
    vq = _speed(question)
    tq = _time(question)
    sq = _distance(question)
    aq = _accel(question)
    u0 = _initial_speed(question)
    vf = _final_speed(question)
    g = _g_value(question)

    # Constant speed family s = v t.  Put target-driven branches before v=s/t.
    if _ask_distance(q) and vq and tq:
        return _result(vq.value * tq.value, question, "m", "s=vt", {"v": vq.value, "t": tq.value}, "Distance equals speed times time.")
    if _ask_speed(q) and sq and tq and tq.value:
        return _result(sq.value / tq.value, question, "m/s", "v=s/t", {"s": sq.value, "t": tq.value}, "Speed is distance divided by time.")
    if _ask_time(q) and sq and vq and abs(vq.value) > 1e-12:
        return _result(sq.value / vq.value, question, "s", "t=s/v", {"s": sq.value, "v": vq.value}, "Time is distance divided by speed.")

    # Constant acceleration equation family.
    if _ask_accel(q) and u0 is not None and vf is not None and tq and tq.value:
        return _result((vf - u0) / tq.value, question, "m/s^2", "a=(v-u)/t", {"u": u0, "v": vf, "t": tq.value})
    if _ask_speed(q) and u0 is not None and aq and tq:
        return _result(u0 + aq.value * tq.value, question, "m/s", "v=u+at", {"u": u0, "a": aq.value, "t": tq.value})
    if _ask_distance(q) and u0 is not None and aq and tq:
        return _result(u0 * tq.value + 0.5 * aq.value * tq.value ** 2, question, "m", "s=ut+1/2at^2", {"u": u0, "a": aq.value, "t": tq.value})
    if _ask_time(q) and u0 is not None and vf is not None and aq and abs(aq.value) > 1e-12:
        return _result((vf - u0) / aq.value, question, "s", "t=(v-u)/a", {"u": u0, "v": vf, "a": aq.value})
    if _ask_speed(q) and u0 is not None and aq and sq:
        val = max(0.0, u0 * u0 + 2 * aq.value * sq.value)
        return _result(math.sqrt(val), question, "m/s", "v=sqrt(u^2+2as)", {"u": u0, "a": aq.value, "s": sq.value})
    if _contains(q, "free fall", "dropped", "falls") and _ask_time(q) and sq:
        return _result(math.sqrt(2 * sq.value / g), question, "s", "t=sqrt(2h/g)", {"h": sq.value, "g": g})
    if _contains(q, "free fall", "dropped", "falls") and _ask_speed(q) and sq:
        return _result(math.sqrt(2 * g * sq.value), question, "m/s", "v=sqrt(2gh)", {"h": sq.value, "g": g})
    if _contains(q, "projectile", "projected", "thrown"):
        theta = _angle(question)
        speed0 = u0 if u0 is not None else (vq.value if vq else None)
        if speed0 is not None and theta is not None:
            if _contains(q, "range", "horizontal distance"):
                return _result(speed0 * speed0 * math.sin(2 * theta) / g, question, "m", "R=u^2 sin(2theta)/g", {"u": speed0, "theta": theta, "g": g})
            if _contains(q, "maximum height", "max height"):
                return _result((speed0 * math.sin(theta)) ** 2 / (2 * g), question, "m", "H=u^2 sin^2(theta)/(2g)", {"u": speed0, "theta": theta, "g": g})
            if _contains(q, "time of flight", "flight time"):
                return _result(2 * speed0 * math.sin(theta) / g, question, "s", "T=2u sin(theta)/g", {"u": speed0, "theta": theta, "g": g})
    return None


def _solve_dynamics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "force", "mass", "acceleration", "weight", "momentum", "impulse", "work", "power", "kinetic", "potential", "pressure", "spring", "hooke"):
        return None
    m = _mass(question)
    F = _force(question)
    a = _accel(question)
    g = _g_value(question)
    v = _speed(question)
    s = _distance(question)
    E = _energy(question)
    t = _time(question)
    P = _power(question)

    # Elevator / apparent-weight family.  This must precede plain F=ma because
    # the requested normal force is the contact force from the floor, not the
    # net force.  Upward acceleration: N - mg = ma -> N=m(g+a).
    # Downward acceleration: mg - N = ma -> N=m(g-a).
    if ("elevator" in q or "lift" in q) and ("normal force" in q or "apparent weight" in q or "floor" in q) and m:
        if a:
            sign = -1.0 if ("downward" in q or "downwards" in q) else 1.0
            return _result(m.value * (g + sign * abs(a.value)), question, "N", "N=m(g±a)", {"m": m.value, "g": g, "a": sign * abs(a.value)}, "For an elevator passenger, the normal force equals m(g+a) for upward acceleration and m(g-a) for downward acceleration.")
        return _result(m.value * g, question, "N", "N=mg", {"m": m.value, "g": g}, "With no elevator acceleration specified, the normal force equals the weight.")

    # F=ma family, target-driven and before weight fallback.
    if _ask_force(q) and m and a:
        return _result(m.value * a.value, question, "N", "F=ma", {"m": m.value, "a": a.value}, "Newton's second law gives force as mass times acceleration.")
    if _ask_mass(q) and F and a and abs(a.value) > 1e-12:
        return _result(F.value / a.value, question, "kg", "m=F/a", {"F": F.value, "a": a.value})
    if _ask_accel(q) and F and m and abs(m.value) > 1e-12:
        return _result(F.value / m.value, question, "m/s^2", "a=F/m", {"F": F.value, "m": m.value})

    # Weight W=mg; do not answer apparent weight here.
    if _ask_weight(q) and not _contains(q, "apparent weight") and m:
        return _result(m.value * g, question, "N", "W=mg", {"m": m.value, "g": g}, "Weight is mass times gravitational field strength.")

    if _contains(q, "momentum") and m and v:
        return _result(m.value * v.value, question, "kg m/s", "p=mv", {"m": m.value, "v": v.value})
    if _contains(q, "impulse") and F and t:
        return _result(F.value * t.value, question, "N s", "J=Ft", {"F": F.value, "t": t.value})
    if _contains(q, "work", "work done") and F and s:
        return _result(F.value * s.value, question, "J", "W=Fs", {"F": F.value, "s": s.value})
    if _contains(q, "power") and E and t and t.value:
        return _result(E.value / t.value, question, "W", "P=W/t", {"W": E.value, "t": t.value})
    if _contains(q, "work", "energy") and P and t:
        return _result(P.value * t.value, question, "J", "W=Pt", {"P": P.value, "t": t.value})
    if _contains(q, "kinetic energy") and m and v:
        return _result(0.5 * m.value * v.value ** 2, question, "J", "K=1/2mv^2", {"m": m.value, "v": v.value})
    if _contains(q, "potential energy") and m and s and not _contains(q, "gravitational potential energy of two masses", "orbit", "planet"):
        return _result(m.value * g * s.value, question, "J", "U=mgh", {"m": m.value, "g": g, "h": s.value})

    # Spring / Hooke family.
    k = _spring_k(question)
    x = _label_value(question, r"extension|compression|displacement|stretch(?:ed)?|elongation|amplitude", U_LEN) or _symbol_value(question, ["x", "A", "amplitude"], U_LEN)
    if (_contains(q, "spring", "hooke") or k) and k:
        if _ask_force(q) and x:
            return _result(k.value * x.value, question, "N", "F=kx", {"k": k.value, "x": x.value})
        if _contains(q, "energy", "work") and x:
            return _result(0.5 * k.value * x.value ** 2, question, "J", "U=1/2kx^2", {"k": k.value, "x": x.value})
        if _contains(q, "extension", "compression", "displacement") and F and abs(k.value) > 1e-12:
            return _result(F.value / k.value, question, "m", "x=F/k", {"F": F.value, "k": k.value})

    # Pressure from force/area; target-driven to avoid density stealing.
    A = _area(question)
    Pr = _pressure(question)
    if _contains(q, "pressure") and F and A and A.value:
        return _result(F.value / A.value, question, "Pa", "P=F/A", {"F": F.value, "A": A.value})
    if _ask_force(q) and Pr and A:
        return _result(Pr.value * A.value, question, "N", "F=PA", {"P": Pr.value, "A": A.value})
    return None


def _solve_fluids(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "fluid", "liquid", "water", "oil", "density", "buoyant", "buoyancy", "archimedes", "submerged", "floating", "pressure", "hydrostatic", "flow", "continuity", "bernoulli"):
        return None
    rho = _density(question)
    V = _volume(question)
    m = _mass(question)
    g = _g_value(question)
    h = _label_value(question, r"depth|height|below|column", U_LEN) or _distance(question)
    P = _pressure(question)
    A = _area(question)
    F = _force(question)

    # Buoyancy has priority when fluid/density/volume appear and target is force.
    if _contains(q, "buoyant", "buoyancy", "archimedes", "submerged", "displaced") and rho and V and (_ask_force(q) or _contains(q, "upthrust")):
        return _result(rho.value * g * V.value, question, "N", "F_b=rho g V", {"rho": rho.value, "g": g, "V": V.value}, "Archimedes' principle gives buoyant force equal to fluid weight displaced.")
    if _contains(q, "apparent weight") and m and rho and V:
        return _result(m.value * g - rho.value * g * V.value, question, "N", "W_app=mg-rho g V", {"m": m.value, "rho": rho.value, "V": V.value, "g": g})
    # Density/mass/volume family.
    if _contains(q, "density") and m and V and V.value:
        return _result(m.value / V.value, question, "kg/m^3", "rho=m/V", {"m": m.value, "V": V.value})
    if _ask_mass(q) and rho and V:
        return _result(rho.value * V.value, question, "kg", "m=rho V", {"rho": rho.value, "V": V.value})
    if _contains(q, "volume") and m and rho and rho.value:
        return _result(m.value / rho.value, question, "m^3", "V=m/rho", {"m": m.value, "rho": rho.value})
    # Hydrostatic / pressure family.
    if _contains(q, "hydrostatic", "pressure", "depth") and rho and h:
        if _contains(q, "gauge", "hydrostatic", "due to liquid", "at depth", "below") or _contains(q, "pressure"):
            return _result(rho.value * g * h.value, question, "Pa", "P=rho g h", {"rho": rho.value, "g": g, "h": h.value})
    if _contains(q, "pressure") and F and A and A.value:
        return _result(F.value / A.value, question, "Pa", "P=F/A", {"F": F.value, "A": A.value})
    # Continuity equation A1v1=A2v2.
    areas = _all(question, U_AREA)
    speeds = _all(question, U_SPEED)
    if _contains(q, "continuity", "flow") and len(areas) >= 2 and speeds:
        # If one speed is given and asks for speed/velocity in second section.
        return _result(areas[0].value * speeds[0].value / areas[1].value, question, "m/s", "A1v1=A2v2", {"A1": areas[0].value, "v1": speeds[0].value, "A2": areas[1].value})
    return None


def _solve_thermal(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "heat", "temperature", "specific heat", "latent", "melt", "boil", "calor", "thermal", "ideal gas", "gas", "mole", "isothermal", "adiabatic", "engine", "efficiency"):
        return None
    m = _mass(question)
    Q = _energy(question)
    c = _specific_heat(question)
    dT = _temperature_delta(question)
    L = _latent_heat(question)

    # Q = m c ΔT family, target-driven.  This fixes the old one-direction
    # templates that returned ΔT or m when the question asked for heat.
    if c is not None:
        if _ask_heat(q) and m and dT is not None:
            return _result(m.value * c * dT, question, "J", "Q=mcΔT", {"m": m.value, "c": c, "ΔT": dT})
        if _ask_mass(q) and Q and dT is not None and abs(c * dT) > 1e-12:
            return _result(Q.value / (c * dT), question, "kg", "m=Q/(cΔT)", {"Q": Q.value, "c": c, "ΔT": dT})
        if _ask_temp_change(q) and Q and m and abs(m.value * c) > 1e-12:
            return _result(Q.value / (m.value * c), question, "K", "ΔT=Q/(mc)", {"Q": Q.value, "m": m.value, "c": c})
        if _contains(q, "specific heat") and Q and m and dT is not None and abs(m.value * dT) > 1e-12:
            return _result(Q.value / (m.value * dT), question, "J/kg/K", "c=Q/(mΔT)", {"Q": Q.value, "m": m.value, "ΔT": dT})

    # Q = m L family.
    if L is not None:
        if _ask_heat(q) and m:
            return _result(m.value * L, question, "J", "Q=mL", {"m": m.value, "L": L})
        if _ask_mass(q) and Q and abs(L) > 1e-12:
            return _result(Q.value / L, question, "kg", "m=Q/L", {"Q": Q.value, "L": L})
        if _contains(q, "latent heat") and Q and m and abs(m.value) > 1e-12:
            return _result(Q.value / m.value, question, "J/kg", "L=Q/m", {"Q": Q.value, "m": m.value})

    # Ideal gas law PV=nRT. Keep simple target-driven support.
    P = _pressure(question)
    V = _volume(question)
    T = _symbol_value(question, ["T", "temperature"], U_TEMP) or _label_value(question, r"temperature", U_TEMP)
    nmol = None
    mt = re.search(rf"(?:n|moles?|amount\s+of\s+substance)\D{{0,20}}(?P<v>{_NUM})\s*(?:mol|moles?)", _clean(question), flags=re.I)
    if mt:
        nmol = _parse_number(mt.group("v"))
    if _contains(q, "ideal gas", "gas") and P and V and T and not nmol and _contains(q, "mole"):
        return _result(P.value * V.value / (R_GAS * T.value), question, "mol", "n=PV/(RT)", {"P": P.value, "V": V.value, "R": R_GAS, "T": T.value})
    if _contains(q, "pressure") and V and T and nmol is not None and V.value:
        return _result(nmol * R_GAS * T.value / V.value, question, "Pa", "P=nRT/V", {"n": nmol, "R": R_GAS, "T": T.value, "V": V.value})
    return None


def _solve_waves(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "wave", "wavelength", "frequency", "period", "oscillation", "oscillator", "pendulum", "sound", "harmonic", "angular frequency", "omega", "amplitude"):
        return None
    f = _freq(question)
    lam = _wavelength(question)
    T = _period(question) or (_time(question) if _contains(q, "period") else None)
    v = _speed(question)

    # v = f λ family. Put speed before period to avoid old T=1/f stealing.
    if (_contains(q, "wave speed", "speed of wave", "speed of the wave") or (_ask_speed(q) and _contains(q, "wave"))) and f and lam:
        return _result(f.value * lam.value, question, "m/s", "v=fλ", {"f": f.value, "lambda": lam.value})
    if _contains(q, "wavelength") and v and f and f.value:
        return _result(v.value / f.value, question, "m", "λ=v/f", {"v": v.value, "f": f.value})
    if _contains(q, "frequency") and v and lam and lam.value:
        return _result(v.value / lam.value, question, "Hz", "f=v/λ", {"v": v.value, "lambda": lam.value})
    if _contains(q, "angular frequency", "omega"):
        if f:
            return _result(2 * math.pi * f.value, question, "rad/s", "ω=2πf", {"f": f.value})
        if T and T.value:
            return _result(2 * math.pi / T.value, question, "rad/s", "ω=2π/T", {"T": T.value})
    if _contains(q, "period") and f and f.value:
        return _result(1.0 / f.value, question, "s", "T=1/f", {"f": f.value})
    if _contains(q, "frequency") and T and T.value:
        return _result(1.0 / T.value, question, "Hz", "f=1/T", {"T": T.value})

    # Pendulum and mass-spring oscillator periods.
    if _contains(q, "pendulum") and _contains(q, "period"):
        L = _distance(question)
        if L:
            return _result(2 * math.pi * math.sqrt(L.value / _g_value(question)), question, "s", "T=2πsqrt(L/g)", {"L": L.value, "g": _g_value(question)})
    if _contains(q, "spring", "mass-spring", "oscillator") and _contains(q, "period"):
        k = _spring_k(question)
        m = _mass(question)
        if k and m and k.value:
            return _result(2 * math.pi * math.sqrt(m.value / k.value), question, "s", "T=2πsqrt(m/k)", {"m": m.value, "k": k.value})
    # Sound intensity level.
    if _contains(q, "decibel", "sound level"):
        t = _clean(question)
        mI = re.search(rf"(?:intensity|I)\D{{0,20}}(?P<v>{_NUM})\s*W\s*/\s*m\s*(?:\^\s*2|2)", t, flags=re.I)
        if mI:
            I = _parse_number(mI.group("v"))
            return _result(10 * math.log10(I / 1e-12), question, "dB", "β=10log10(I/I0)", {"I": I, "I0": 1e-12})
    return None


def _solve_optics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "lens", "mirror", "focal", "image", "object", "magnification", "snell", "refractive", "critical angle"):
        return None
    f = _symbol_value(question, ["f", "focal length"], U_LEN) or _label_value(question, r"focal\s+length", U_LEN)
    do = _symbol_value(question, ["do", "d_o", "object distance", "u"], U_LEN) or _label_value(question, r"object\s+(?:distance|is\s+placed|placed)|in\s+front\s+of", U_LEN)
    di = _symbol_value(question, ["di", "d_i", "image distance", "v"], U_LEN) or _label_value(question, r"image\s+distance", U_LEN)
    ho = _symbol_value(question, ["ho", "h_o", "object height"], U_LEN) or _label_value(question, r"object\s+height", U_LEN)
    hi = _symbol_value(question, ["hi", "h_i", "image height"], U_LEN) or _label_value(question, r"image\s+height", U_LEN)

    if _contains(q, "image height") and ho:
        if do and di:
            return _result(-(di.value / do.value) * ho.value, question, ho.unit or "m", "h_i=-(d_i/d_o)h_o", {"di": di.value, "do": do.value, "ho": ho.value})
        mmag = re.search(rf"(?:magnification|m)\D{{0,20}}(?P<v>{_NUM})(?!\s*(?:{U_LEN}))", _clean(question), flags=re.I)
        if mmag:
            return _result(_parse_number(mmag.group("v")) * ho.value, question, ho.unit or "m", "h_i=m h_o", {"m": _parse_number(mmag.group("v")), "ho": ho.value})
    if _contains(q, "magnification"):
        if do and di:
            return _result(-di.value / do.value, question, None, "m=-d_i/d_o", {"di": di.value, "do": do.value})
        if ho and hi and ho.value:
            return _result(hi.value / ho.value, question, None, "m=h_i/h_o", {"hi": hi.value, "ho": ho.value})
    if _contains(q, "image distance") and f and do:
        denom = 1.0 / f.value - 1.0 / do.value
        if abs(denom) > 1e-12:
            return _result(1.0 / denom, question, do.unit or "m", "1/f=1/d_o+1/d_i", {"f": f.value, "do": do.value})
    if _contains(q, "focal length") and do and di:
        denom = 1.0 / do.value + 1.0 / di.value
        if abs(denom) > 1e-12:
            return _result(1.0 / denom, question, do.unit or "m", "f=1/(1/d_o+1/d_i)", {"do": do.value, "di": di.value})
    if _contains(q, "refractive index"):
        v = _speed(question)
        if v and v.value:
            return _result(C_LIGHT / v.value, question, None, "n=c/v", {"c": C_LIGHT, "v": v.value})
    if _contains(q, "snell", "angle of refraction", "refraction"):
        t = _clean(question)
        n1 = re.search(rf"n\s*_?1\s*=\s*(?P<v>{_NUM})", t, flags=re.I)
        n2 = re.search(rf"n\s*_?2\s*=\s*(?P<v>{_NUM})", t, flags=re.I)
        th = _symbol_value(question, ["theta1", "theta_1", "incident angle", "angle of incidence"], U_ANGLE) or _label_value(question, r"angle\s+of\s+incidence|incident\s+angle", U_ANGLE)
        if n1 and n2 and th:
            s2 = _parse_number(n1.group("v")) * math.sin(th.value) / _parse_number(n2.group("v"))
            if abs(s2) <= 1:
                return _result(math.degrees(math.asin(s2)), question, "degree", "θ2=asin(n1 sinθ1/n2)", {"n1": _parse_number(n1.group("v")), "n2": _parse_number(n2.group("v")), "theta1": th.value})
    return None


def _solve_modern(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "photon", "de broglie", "compton", "relativ", "photoelectric", "work function", "wavelength", "frequency", "rydberg", "bohr"):
        return None
    f = _freq(question)
    lam = _wavelength(question)
    E = _energy(question)
    if _contains(q, "photon", "energy"):
        if f:
            return _result(H_PLANCK * f.value, question, "J", "E=hf", {"h": H_PLANCK, "f": f.value}, sig=7)
        if lam and lam.value:
            return _result(H_PLANCK * C_LIGHT / lam.value, question, "J", "E=hc/λ", {"h": H_PLANCK, "c": C_LIGHT, "lambda": lam.value}, sig=7)
    if _contains(q, "de broglie") or (_contains(q, "wavelength") and _contains(q, "particle")):
        m = _mass(question)
        v = _speed(question)
        if m and v and m.value * v.value:
            return _result(H_PLANCK / (m.value * v.value), question, "m", "λ=h/(mv)", {"h": H_PLANCK, "m": m.value, "v": v.value}, sig=7)
    if _contains(q, "frequency") and E:
        return _result(E.value / H_PLANCK, question, "Hz", "f=E/h", {"E": E.value, "h": H_PLANCK}, sig=7)
    if _contains(q, "wavelength") and E and E.value:
        return _result(H_PLANCK * C_LIGHT / E.value, question, "m", "λ=hc/E", {"h": H_PLANCK, "c": C_LIGHT, "E": E.value}, sig=7)
    return None


def solve_nephys_equation_family(question: str) -> SolverResult | None:
    q = _low(question)
    # Keep mature electricity/circuit solvers in control unless the prompt is
    # clearly a non-electric mixed-domain problem with words such as wave/photon.
    if _is_electric(q) and not _contains(q, "wave", "photon", "de broglie", "lens", "mirror", "heat", "fluid"):
        return None
    for fn in (
        _solve_fluids,
        _solve_thermal,
        _solve_waves,
        _solve_optics,
        _solve_modern,
        _solve_kinematics,
        _solve_dynamics,
    ):
        try:
            out = fn(question)
        except ZeroDivisionError:
            out = None
        except Exception:
            out = None
        if out is not None:
            try:
                out.debug = dict(out.debug or {})
                out.debug.setdefault("nephys_equation_family", fn.__name__)
            except Exception:
                pass
            return out
    return None
