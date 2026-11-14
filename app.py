"""
Free-Body Diagram Builder
A Streamlit app for practicing Newton's laws.

Students:
  1. Read a situation.
  2. Drag forces onto a free-body diagram, aiming each arrow the way the force acts.
  3. Once the diagram is correct, write Newton's second law for the x- and y-axes.

Run locally:   streamlit run app.py
Folder layout: app.py, requirements.txt, fbd_component/index.html
"""

import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

st.set_page_config(page_title="Free-Body Diagram Builder", page_icon="🧭", layout="wide")

# The drag-and-drop diagram is a small custom component (plain HTML + JavaScript).
fbd_builder = components.declare_component(
    "fbd_builder", path=str(Path(__file__).parent / "fbd_component")
)

ANGLE_TOLERANCE = 20  # degrees a student's arrow may be off and still count as correct

# ---------------------------------------------------------------------------
# Forces in the palette (real forces plus common misconceptions)
# ---------------------------------------------------------------------------
FORCES = {
    "weight": dict(
        name="Gravitational force (weight)", base="F", sub="g", color="#2f6fd6", real=True,
        hint="Gravity always pulls straight down, toward the center of Earth.",
    ),
    "normal": dict(
        name="Normal force", base="F", sub="N", color="#23925a", real=True,
        hint="A normal force is perpendicular to the surface, pushing away from it.",
    ),
    "applied": dict(
        name="Applied force (push)", base="F", sub="A", color="#d6336c", real=True,
        hint="An applied force points in the direction of the push.",
    ),
    "motion": dict(
        name="Force of motion", base="F", sub="motion", color="#a0522d", real=False,
        hint="There is no \"force of motion.\" A moving object keeps moving because of inertia "
             "(Newton's first law); nothing has to push it along.",
    ),
    "tension": dict(
        name="Tension", base="T", sub="", color="#7b4fd6", real=True,
        hint="Tension pulls along the rope or cord, away from the object.",
    ),
    "kfriction": dict(
        name="Kinetic friction", base="f", sub="k", color="#e0702b", real=True,
        hint="Kinetic friction acts along the surface, opposite the direction the object slides.",
    ),
    "net": dict(
        name="Net force", base="F", sub="net", color="#0b7a75", real=False,
        hint="The net force isn't a separate force on the object. It's the vector sum of the forces "
             "already in your diagram, so drawing it counts those forces twice.",
    ),
    "sfriction": dict(
        name="Static friction", base="f", sub="s", color="#b8860b", real=True,
        hint="Static friction acts along the surface, opposite the way the object would slide "
             "if the surface were frictionless.",
    ),
    "air": dict(
        name="Air resistance", base="F", sub="air", color="#1597b0", real=True,
        hint="Air resistance points opposite the object's velocity.",
    ),
    "inertia": dict(
        name="Force of inertia", base="F", sub="inertia", color="#5b6b2f", real=False,
        hint="Inertia is a property of the object (its resistance to changes in motion), "
             "not a force that something exerts on it.",
    ),
}

FORCE_PAYLOAD = [
    dict(id=fid, name=f["name"], base=f["base"], sub=f["sub"], color=f["color"])
    for fid, f in FORCES.items()
]

# ---------------------------------------------------------------------------
# Situations
# Angles are in degrees, measured counterclockwise from the +x direction (pointing right).
# Equations are written as (left side − right side) of ΣF = ma, using the symbols below.
# The inclines and the sled rope are drawn at θ = 30°.
# ---------------------------------------------------------------------------
STANDARD_AXES = "+x points to the right and +y points straight up."
INCLINE_AXES = ("Tilt your axes: +x points **down the incline**, and +y points perpendicular "
                "to the incline, away from the surface.")

SCENARIOS = [
    dict(id="book", title="Book on a table", scene="table", variant="",
         description="A textbook sits at rest on a level table.",
         caption="At rest on a level table", axis_angle=0, axes=STANDARD_AXES,
         answer={"weight": 270, "normal": 90},
         eq_x="0", eq_y="F_N - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_N - F_g = 0"),
    dict(id="lamp", title="Hanging lamp", scene="hanging", variant="",
         description="A lamp hangs at rest from a single vertical cord attached to the ceiling.",
         caption="Hanging at rest from one cord", axis_angle=0, axes=STANDARD_AXES,
         answer={"tension": 90, "weight": 270},
         eq_x="0", eq_y="T - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = T - F_g = 0"),
    dict(id="skydiver", title="Skydiver at terminal velocity", scene="skydiver", variant="",
         description="A skydiver falls straight down at a constant (terminal) velocity.",
         caption="Falling at constant velocity", axis_angle=0, axes=STANDARD_AXES,
         answer={"weight": 270, "air": 90},
         eq_x="0", eq_y="F_air - F_g", ax=False, ay=False,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_{air} - F_g = 0"),
    dict(id="elevator", title="Person in an elevator", scene="elevator", variant="",
         description="A person stands on the floor of an elevator that is moving upward "
                     "and speeding up.",
         caption="Moving upward and speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"normal": 90, "weight": 270},
         eq_x="0", eq_y="F_N - F_g - m*a", ax=False, ay=True,
         tex_x=r"\Sigma F_x = 0 \quad (\text{no forces along } x)",
         tex_y=r"\Sigma F_y = F_N - F_g = ma"),
    dict(id="push", title="Pushing a box", scene="push", variant="",
         description="A student pushes horizontally on a box. The box slides to the right across "
                     "a rough floor and speeds up.",
         caption="Sliding right on a rough floor, speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"applied": 0, "kfriction": 180, "weight": 270, "normal": 90},
         eq_x="F_A - f_k - m*a", eq_y="F_N - F_g", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_A - f_k = ma",
         tex_y=r"\Sigma F_y = F_N - F_g = 0"),
    dict(id="sled", title="Pulling a sled", scene="sled", variant="",
         description="A child pulls a sled across level snow with a rope held at angle θ above "
                     "the horizontal. The snow exerts friction on the sliding sled, and the sled "
                     "speeds up.",
         caption="Pulled at angle θ, speeding up", axis_angle=0, axes=STANDARD_AXES,
         answer={"tension": 30, "kfriction": 180, "weight": 270, "normal": 90},
         eq_x="T*cos(theta) - f_k - m*a", eq_y="F_N + T*sin(theta) - F_g", ax=True, ay=False,
         tex_x=r"\Sigma F_x = T\cos\theta - f_k = ma",
         tex_y=r"\Sigma F_y = F_N + T\sin\theta - F_g = 0"),
    dict(id="ramp_smooth", title="Block on a frictionless ramp", scene="incline",
         variant="frictionless",
         description="A block slides down a frictionless ramp inclined at angle θ.",
         caption="Frictionless ramp, sliding down", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60},
         eq_x="F_g*sin(theta) - m*a", eq_y="F_N - F_g*cos(theta)", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta = ma",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
    dict(id="ramp_static", title="Box resting on a rough ramp", scene="incline", variant="static",
         description="A box sits at rest on a rough ramp inclined at angle θ. It does not slide.",
         caption="Rough ramp, box at rest", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60, "sfriction": 150},
         eq_x="F_g*sin(theta) - f_s", eq_y="F_N - F_g*cos(theta)", ax=False, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta - f_s = 0",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
    dict(id="ramp_kinetic", title="Block sliding down a rough ramp", scene="incline",
         variant="kinetic",
         description="A block slides down a rough ramp inclined at angle θ, speeding up as it goes.",
         caption="Rough ramp, sliding down and speeding up", axis_angle=-30, axes=INCLINE_AXES,
         answer={"weight": 270, "normal": 60, "kfriction": 150},
         eq_x="F_g*sin(theta) - f_k - m*a", eq_y="F_N - F_g*cos(theta)", ax=True, ay=False,
         tex_x=r"\Sigma F_x = F_g\sin\theta - f_k = ma",
         tex_y=r"\Sigma F_y = F_N - F_g\cos\theta = 0"),
]

# ---------------------------------------------------------------------------
# Symbols students can type, with common alternate spellings
# ---------------------------------------------------------------------------
m, g, a, theta, mu_k = sp.symbols("m g a theta mu_k", positive=True)
a_x, a_y = sp.symbols("a_x a_y")
F_g, F_N, T, f_k, f_s, F_A, F_air = sp.symbols("F_g F_N T f_k f_s F_A F_air", positive=True)
FORCE_SYMBOLS = {F_g, F_N, T, f_k, f_s, F_A, F_air}

NAMES = {
    "m": m, "g": g, "a": a, "theta": theta,
    "a_x": a_x, "ax": a_x, "a_y": a_y, "ay": a_y,
    "mu_k": mu_k, "muk": mu_k, "mu": mu_k,
    "F_g": F_g, "Fg": F_g, "F_w": F_g, "Fw": F_g, "W": F_g, "mg": m * g, "ma": m * a,
    "F_N": F_N, "FN": F_N, "F_n": F_N, "Fn": F_N, "N": F_N,
    "T": T, "F_T": T, "FT": T, "F_t": T, "Ft": T,
    "f_k": f_k, "fk": f_k, "F_k": f_k, "Fk": f_k,
    "f_s": f_s, "fs": f_s, "F_s": f_s, "Fs": f_s,
    "F_A": F_A, "FA": F_A, "F_a": F_A, "Fa": F_A, "F_app": F_A, "Fapp": F_A,
    "F_air": F_air, "Fair": F_air, "F_D": F_air, "FD": F_air,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
}

# Substitutions that let equivalent answers match (F_g = mg, f_k = μ_k F_N)
BASE_SUBS = {F_g: m * g, f_k: mu_k * F_N}

TRANSFORMS = standard_transformations + (implicit_multiplication, implicit_application, convert_xor)
ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/().=^\s]*$")
NET_FORCE_WORD = re.compile(r"^(Sigma|sum)?_?F_?(net)?_?[xy]$", re.IGNORECASE)


def normalize(text: str) -> str:
    replacements = {
        "θ": "theta", "Θ": "theta", "μ": "mu", "·": "*", "×": "*",
        "−": "-", "–": "-", "Σ": "", "∑": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def parse_side(side_text: str):
    return parse_expr(side_text, local_dict=NAMES, transformations=TRANSFORMS)


def parse_equation(text: str):
    """Return ((lhs, rhs), None) or (None, error message)."""
    s = normalize(text)
    if not s:
        return None, "Type an equation."
    if not ALLOWED_CHARS.match(s):
        return None, "Use only letters, numbers, + − * / ^ ( ) and one equals sign."
    if "=" not in s:
        return None, "Your equation needs an equals sign, for example: forces = m*a"
    parts = s.split("=")
    lhs_text, rhs_text = parts[-2], parts[-1]  # allows "ΣF_x = ... = m*a"
    if not lhs_text.strip() or not rhs_text.strip():
        return None, "Put something on both sides of the equals sign."
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lhs_text + " " + rhs_text))
    if any(NET_FORCE_WORD.match(w) for w in words):
        return None, "Leave out the ΣF part. Type just the forces, then = m*a (or = 0)."
    unknown = sorted(w for w in words if w not in NAMES)
    if unknown:
        return None, (f"I don't recognize: {', '.join(unknown)}. Check the symbol list, and use * "
                      "between symbols (for example, F_g*sin(theta)).")
    try:
        return (parse_side(lhs_text), parse_side(rhs_text)), None
    except Exception:
        return None, "I couldn't read that equation. Check your parentheses and operators."


def equation_latex(lhs, rhs) -> str:
    return f"{sp.latex(lhs)} = {sp.latex(rhs)}"


def forces_in(expr):
    """Which forces appear in an expression (treating mg as F_g and μ_k F_N as f_k)."""
    e = expr.subs(mu_k * F_N, f_k).subs(m * g, F_g)
    return e.free_symbols & FORCE_SYMBOLS


def check_equation(text: str, sc: dict, axis: str) -> dict:
    parsed, error = parse_equation(text)
    if error:
        return dict(ok=False, msg=error)
    lhs, rhs = parsed

    accelerates = sc["ax"] if axis == "x" else sc["ay"]
    own, other = (a_x, a_y) if axis == "x" else (a_y, a_x)
    student_raw = lhs - rhs
    if student_raw.has(other):
        return dict(ok=False, msg=f"You used ${sp.latex(other)}$ in the {axis}-axis equation. "
                                  f"Use $a$ (or ${sp.latex(own)}$) for acceleration along this axis.")

    accel_subs = {own: a} if accelerates else {own: 0, a: 0}
    expected_raw = parse_side(sc[f"eq_{axis}"])
    student = sp.simplify(student_raw.subs(accel_subs).subs(BASE_SUBS))
    expected = sp.simplify(expected_raw.subs(BASE_SUBS))

    if expected == 0:
        if student == 0:
            return dict(ok=True, msg="Correct. No forces act along this axis, so the equation is 0 = 0.")
        return dict(ok=False, msg=f"None of the forces in your diagram has a component along the "
                                  f"{axis}-axis, and the object doesn't accelerate that way, "
                                  f"so this equation is just 0 = 0.")
    if student == 0:
        return dict(ok=False, msg="Your equation works out to 0 = 0, but forces in your diagram "
                                  f"do have components along the {axis}-axis.")

    ratio = sp.simplify(student / expected)
    if ratio.is_number and ratio != 0:
        note = ""
        if not accelerates and student_raw.subs(own, a).has(a):
            note = (f" Since the object doesn't accelerate along the {axis}-axis, "
                    f"$a_{axis} = 0$ and the right side is 0.")
        return dict(ok=True, msg="Correct!" + note)

    # --- Hints, most specific first ---
    expected_forces, student_forces = forces_in(expected_raw), forces_in(student_raw)
    missing = expected_forces - student_forces
    extra = student_forces - expected_forces
    if missing:
        names = ", ".join(f"${sp.latex(s)}$" for s in sorted(missing, key=str))
        return dict(ok=False, msg=f"Your equation is missing {names}. Every force with a component "
                                  f"along the {axis}-axis belongs in the sum.")
    if extra:
        names = ", ".join(f"${sp.latex(s)}$" for s in sorted(extra, key=str))
        return dict(ok=False, msg=f"Your equation includes {names}, which has no component along "
                                  f"the {axis}-axis.")
    for trig in (sp.sin(theta), sp.cos(theta)):
        if expected.has(trig) != student.has(trig):
            return dict(ok=False, msg="Check your trig. For each angled force, which component "
                                      "uses sin θ and which uses cos θ?")
    for term in sp.Add.make_args(sp.expand(expected_raw)):
        flipped = expected - 2 * term.subs(BASE_SUBS)
        r = sp.simplify(student / flipped) if flipped != 0 else None
        if r is not None and r.is_number and r != 0:
            shown = -term if term.could_extract_minus_sign() else term
            return dict(ok=False, msg=f"Check the sign of the ${sp.latex(shown)}$ term. "
                                      f"Components pointing in the −{axis} direction are negative.")
    if accelerates and not student.has(a):
        return dict(ok=False, msg="The object accelerates along this axis, so the right side "
                                  "should be $ma$, not 0.")
    return dict(ok=False, msg=f"Not quite. For each force in your diagram, ask whether it has a "
                              f"component along the {axis}-axis, and whether that component is "
                              f"positive or negative.")


# ---------------------------------------------------------------------------
# Free-body diagram checking
# ---------------------------------------------------------------------------
def angle_diff(a1: float, a2: float) -> float:
    return abs((a1 - a2 + 180) % 360 - 180)


def extra_force_hint(fid: str, sc: dict) -> str:
    answer, name = sc["answer"], FORCES[fid]["name"]
    if fid == "kfriction" and "sfriction" in answer:
        return "**Kinetic friction:** the object isn't sliding, so the friction here is static, not kinetic."
    if fid == "sfriction" and "kfriction" in answer:
        return "**Static friction:** the object is sliding across the surface, so the friction is kinetic, not static."
    if fid in ("kfriction", "sfriction"):
        if sc["variant"] == "frictionless":
            return f"**{name}:** this ramp is frictionless."
        return f"**{name}:** nothing is sliding, or trying to slide, along a surface here, so there's no friction."
    if fid == "air":
        return "**Air resistance:** we're ignoring air resistance in this situation."
    if fid == "normal":
        return "**Normal force:** no surface is pressing against the object here, so there's no normal force."
    if fid == "tension":
        return "**Tension:** no rope, string, or cord is pulling on the object here."
    if fid == "applied":
        return ("**Applied force:** nothing is pushing on the object right now. A push that started "
                "its motion earlier isn't still acting.")
    return f"**{name}:** is that force really acting on the object here?"


def check_fbd(placed: list, sc: dict):
    answer = sc["answer"]
    if not placed:
        return False, [("info", "Drag forces onto the diagram first.")]
    messages = []
    ids = [p["id"] for p in placed]
    for fid in ids:
        f = FORCES[fid]
        if not f["real"]:
            messages.append(("error", f"**{f['name']}:** {f['hint']}"))
        elif fid not in answer:
            messages.append(("error", extra_force_hint(fid, sc)))
    for p in placed:
        if p["id"] in answer and angle_diff(p["angle"], answer[p["id"]]) > ANGLE_TOLERANCE:
            f = FORCES[p["id"]]
            messages.append(("warning", f"**{f['name']}** belongs in the diagram, but check its "
                                        f"direction. {f['hint']}"))
    missing = [fid for fid in answer if fid not in ids]
    if missing:
        n = len(missing)
        messages.append(("info", f"You're missing {n} force{'s' if n > 1 else ''}. Start with gravity, "
                                 "then look at everything touching the object: surfaces, ropes, "
                                 "hands, air."))
    return not messages, messages


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("scenario_idx", 0)
ss.setdefault("completed", set())


def reset_scenario(sid: str):
    ss[f"ver_{sid}"] = ss.get(f"ver_{sid}", 0) + 1
    for key in (f"stage_{sid}", f"placed_{sid}", f"fbdmsg_{sid}", f"eqres_{sid}",
                f"eqx_{sid}", f"eqy_{sid}", f"celebrated_{sid}"):
        ss.pop(key, None)
    ss["completed"].discard(sid)


def go_to(idx: int):
    ss["scenario_idx"] = idx


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Situations")
    completed = ss["completed"]
    done_count = len(completed)
    st.progress(done_count / len(SCENARIOS), text=f"{done_count} of {len(SCENARIOS)} complete")
    idx = st.radio(
        "Choose a situation",
        options=range(len(SCENARIOS)),
        format_func=lambda i: ("✅ " if SCENARIOS[i]["id"] in completed else "")
        + f"{i + 1}. {SCENARIOS[i]['title']}",
        key="scenario_idx",
        label_visibility="collapsed",
    )
    sc = SCENARIOS[idx]
    sid = sc["id"]
    st.divider()
    st.button("Start this situation over", on_click=reset_scenario, args=(sid,),
              use_container_width=True)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
stage = ss.setdefault(f"stage_{sid}", "fbd")
version = ss.get(f"ver_{sid}", 0)

st.title("Free-Body Diagram Builder")
st.subheader(f"{idx + 1}. {sc['title']}")
st.markdown(sc["description"])
st.markdown(f"**Axes:** {sc['axes']}")

st.markdown("#### Step 1: Build the free-body diagram")
st.caption("Drag every force acting **on the object** onto the diagram and drop it in the direction "
           "the force points. Drag an arrow to re-aim it, or drag it off the diagram to remove it.")

value = fbd_builder(
    scenario=dict(id=sid, scene=sc["scene"], variant=sc["variant"],
                  caption=sc["caption"], axis_angle=sc["axis_angle"]),
    forces=FORCE_PAYLOAD,
    initial=ss.get(f"placed_{sid}", []),
    locked=(stage != "fbd"),
    key=f"fbd_{sid}_{version}",
    default=None,
)
if value is not None and stage == "fbd":
    ss[f"placed_{sid}"] = value
placed = ss.get(f"placed_{sid}", [])

if stage == "fbd":
    if st.button("Check my diagram", type="primary", key=f"checkfbd_{sid}_{version}"):
        ok, msgs = check_fbd(placed, sc)
        ss[f"fbdmsg_{sid}"] = msgs
        if ok:
            ss[f"stage_{sid}"] = "eq"
            st.rerun()
    for kind, msg in ss.get(f"fbdmsg_{sid}", []):
        getattr(st, kind)(msg)
else:
    st.success("Your free-body diagram is correct.")

# ---------------------------------------------------------------------------
# Step 2: Newton's second law equations
# ---------------------------------------------------------------------------
if stage in ("eq", "done"):
    st.markdown("#### Step 2: Write Newton's second law for each axis")
    st.markdown(r"Use the forces in your diagram to write $\Sigma F_x = ma_x$ and "
                r"$\Sigma F_y = ma_y$. Type the forces on the left and $m\,a$ (or 0) on the right, "
                "for example `forces = m*a`. Use the axis directions given above.")

    with st.expander("Symbols you can type"):
        st.markdown(
            "| Quantity | Type | Shows as |\n|---|---|---|\n"
            "| Weight | `F_g` or `m*g` | $F_g$ or $mg$ |\n"
            "| Normal force | `F_N` | $F_N$ |\n"
            "| Tension | `T` | $T$ |\n"
            "| Kinetic friction | `f_k` or `mu_k*F_N` | $f_k$ or $\\mu_k F_N$ |\n"
            "| Static friction | `f_s` | $f_s$ |\n"
            "| Applied force | `F_A` | $F_A$ |\n"
            "| Air resistance | `F_air` | $F_{air}$ |\n"
            "| Mass, acceleration | `m`, `a` | $m$, $a$ |\n"
            "| Angle | `theta` or `θ` | $\\theta$ |\n"
            "| Components | `sin(theta)`, `cos(theta)` | $\\sin\\theta$, $\\cos\\theta$ |\n"
        )

    results = ss.setdefault(f"eqres_{sid}", {})
    cols = st.columns(2)
    for col, axis in zip(cols, ("x", "y")):
        with col:
            st.markdown(f"**{axis}-axis:** $\\Sigma F_{axis} = ma_{axis}$")
            text = st.text_input(f"{axis}-axis equation", key=f"eq{axis}_{sid}",
                                 placeholder="forces = m*a", disabled=(stage == "done"),
                                 label_visibility="collapsed")
            if text and stage == "eq":
                parsed, err = parse_equation(text)
                if parsed:
                    st.caption("Here's how I read your equation:")
                    st.latex(equation_latex(*parsed))

    if stage == "eq":
        if st.button("Check my equations", type="primary", key=f"checkeq_{sid}"):
            for axis in ("x", "y"):
                results[axis] = check_equation(ss.get(f"eq{axis}_{sid}", ""), sc, axis)
            if all(results.get(ax_, {}).get("ok") for ax_ in ("x", "y")):
                ss[f"stage_{sid}"] = "done"
                ss["completed"].add(sid)
                st.rerun()
        if results:
            cols = st.columns(2)
            for col, axis in zip(cols, ("x", "y")):
                res = results.get(axis)
                if res:
                    with col:
                        (st.success if res["ok"] else st.error)(f"**{axis}-axis:** {res['msg']}")

    if stage == "done":
        if not ss.get(f"celebrated_{sid}"):
            st.balloons()
            ss[f"celebrated_{sid}"] = True
        st.success("Both equations are correct. Here they are written out:")
        st.latex(sc["tex_x"])
        st.latex(sc["tex_y"])
        for axis in ("x", "y"):
            note = results.get(axis, {}).get("msg", "")
            if note.startswith("Correct!") and len(note) > len("Correct!"):
                st.caption(note[len("Correct! "):])
        if idx + 1 < len(SCENARIOS):
            st.button("Next situation", type="primary", on_click=go_to, args=(idx + 1,))
        elif len(ss["completed"]) == len(SCENARIOS):
            st.info("You've finished every situation. Nice work!")
