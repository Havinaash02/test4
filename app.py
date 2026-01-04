import streamlit as st
import math
import random

# =========================================================
# CONFIG + SIMPLE STYLING
# =========================================================
st.set_page_config(page_title="Foundation Design Tool", layout="wide")

st.markdown(
    """
    <style>
      .app-title {font-size: 2.15rem; font-weight: 900; margin-bottom: 0.2rem;}
      .app-sub {color: #6b7280; margin-top: 0;}
      .card {border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px; background: #ffffff;}
      .card:hover {border-color: #cbd5e1; box-shadow: 0 6px 20px rgba(0,0,0,0.06);}
      .kpi {border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; background: #fff;}
      .muted {color:#6b7280;}
      .chip {display:inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 800; font-size: 0.85rem;}
      .chip-ok {background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0;}
      .chip-bad {background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
      hr {margin: 0.7rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🧱 Foundation Design & Analysis Tool</div>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Cleaner UI + audience-friendly defaults (Student / Engineer).</p>', unsafe_allow_html=True)

# =========================================================
# SESSION STATE (NAV)
# =========================================================
FEATURES = [
    "Bearing Capacity",
    "Settlement",
    "Sliding Check",
    "Overturning Check",
    "Full Foundation Design",  # ✅ renamed
]

if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None
if "result" not in st.session_state:
    st.session_state.result = None


def go_home(clear_inputs=False):
    st.session_state.page = "home"
    st.session_state.analysis_type = None
    st.session_state.result = None
    if clear_inputs:
        keep = {"page", "analysis_type", "result"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
    st.rerun()


def open_feature(name):
    st.session_state.page = "feature"
    st.session_state.analysis_type = name
    st.session_state.result = None
    st.rerun()


# =========================================================
# HELPERS
# =========================================================
def chip(text, ok=True):
    cls = "chip chip-ok" if ok else "chip chip-bad"
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)


def kpi(label, value, caption=None):
    st.markdown(
        f"<div class='kpi'><div class='muted'>{label}</div>"
        f"<div style='font-size:1.35rem;font-weight:900'>{value}</div></div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def safe_float_list(csv_text: str):
    out, bad = [], []
    for token in csv_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            bad.append(token)
    return out, bad


# =========================================================
# ENGINEERING FUNCTIONS
# =========================================================
def bearing_capacity_q_ult(c_kpa, gamma_knm3, Df_m, B_m, phi_deg):
    """Ultimate bearing capacity q_ult in kPa (approx)."""
    phi = math.radians(phi_deg)
    q = gamma_knm3 * Df_m  # kPa

    if abs(phi) < 1e-9:
        Nq = 1.0
        Nc = 5.14
        Ng = 0.0
    else:
        Nq = math.exp(math.pi * math.tan(phi)) * (math.tan(math.radians(45) + phi / 2) ** 2)
        Nc = (Nq - 1) / math.tan(phi)
        Ng = 2 * (Nq + 1) * math.tan(phi)  # approximate

    return c_kpa * Nc + q * Nq + 0.5 * gamma_knm3 * B_m * Ng


def settlement_elastic(P_kN, B_m, L_m, Es_kPa, nu, influence_I=1.0):
    """Simplified elastic settlement estimate; returns meters."""
    A = B_m * L_m
    q_kPa = P_kN / A  # kPa
    return influence_I * (q_kPa * B_m * (1 - nu**2)) / Es_kPa  # m


def sliding_fs(P_kN, H_kN, mu):
    if H_kN <= 0:
        return float("inf")
    return (mu * P_kN) / H_kN


def overturning_fs(P_kN, B_m, M_kNm):
    if M_kNm <= 0:
        return float("inf")
    Mr = P_kN * (B_m / 2)
    return Mr / M_kNm


# =========================================================
# PROFILE / AUDIENCE MODE (SIDEBAR)
# =========================================================
st.sidebar.header("⚙️ Profile")

audience = st.sidebar.radio(
    "Audience mode",
    ["Student", "Engineer / Contractor"],
    index=1,
    help="Changes default thresholds + how results are explained.",
    key="audience",
)

# Defaults by audience
if audience == "Student":
    default_FS_bearing = 3.0
    default_FS_slide = 1.5
    default_FS_ot = 2.0
    default_settle_limit_mm = 25.0
    tone_note = "Educational defaults; always verify with code requirements."
else:
    default_FS_bearing = 3.0
    default_FS_slide = 1.5
    default_FS_ot = 2.0
    default_settle_limit_mm = 25.0
    tone_note = "Quick checks only; confirm with project specs + local design codes."

st.sidebar.caption(tone_note)
st.sidebar.markdown("---")

st.sidebar.subheader("✅ Design criteria (editable)")
FS_slide_req = st.sidebar.number_input(
    "Required FS (Sliding)", value=default_FS_slide, min_value=1.0, step=0.1, key="FS_slide_req"
)
FS_ot_req = st.sidebar.number_input(
    "Required FS (Overturning)", value=default_FS_ot, min_value=1.0, step=0.1, key="FS_ot_req"
)
settle_limit = st.sidebar.number_input(
    "Settlement limit (mm)", value=default_settle_limit_mm, min_value=1.0, step=1.0, key="settle_limit"
)

st.sidebar.markdown("---")
st.sidebar.button("⬅ Home", on_click=go_home, kwargs={"clear_inputs": False}, key="sidebar_home")
st.sidebar.button("🧹 Clear & Home", on_click=go_home, kwargs={"clear_inputs": True}, key="sidebar_clear_home")


# =========================================================
# HOME PAGE (CARD GRID)
# =========================================================
if st.session_state.page == "home":
    st.subheader("Choose a function")

    cards = [
        ("⚖️", "Bearing Capacity", "Compute ultimate & allowable bearing capacity."),
        ("📉", "Settlement", "Estimate elastic settlement (simplified)."),
        ("🧷", "Sliding Check", "Factor of safety against sliding."),
        ("🧱", "Overturning Check", "Factor of safety against overturning."),
        ("🎲", "Full Foundation Design", "Robustness score under uncertainty."),  # ✅ renamed
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### {icon} {title}")
            st.write(desc)
            if st.button("Open", key=f"open_{title}"):
                open_feature(title)
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Assumptions / Notes"):
        st.write(
            """
            - Bearing capacity uses an approximate formulation (educational/early sizing).
            - Settlement is a simplified elastic estimate.
            - Sliding uses μ·P only (no key/passive/cohesion).
            - Use this as a **preliminary tool**, then verify with codes and geotech report.
            """
        )

else:
    feature = st.session_state.analysis_type

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.subheader(f"▶ {feature}")
    with top_r:
        st.write("")
        st.button(
            "⬅ Return to homepage",
            key="return_home_top",
            on_click=go_home,
            kwargs={"clear_inputs": False},
        )

    tab_in, tab_out, tab_notes = st.tabs(["🧾 Inputs", "✅ Results", "ℹ️ Notes"])

    # -----------------------------------------------------
    # BEARING CAPACITY
    # -----------------------------------------------------
    if feature == "Bearing Capacity":
        with st.sidebar:
            st.subheader("📐 Geometry")
            B = st.number_input("Width B (m)", value=2.0, min_value=0.1, step=0.1, key="bear_B")
            L = st.number_input("Length L (m)", value=2.0, min_value=0.1, step=0.1, key="bear_L")
            Df = st.number_input("Depth Df (m)", value=1.0, min_value=0.0, step=0.1, key="bear_Df")

            st.subheader("🌱 Soil")
            c = st.number_input("Cohesion c (kPa)", value=20.0, min_value=0.0, step=1.0, key="bear_c")
            phi = st.number_input("Friction angle φ (deg)", value=30.0, min_value=0.0, max_value=45.0, step=1.0, key="bear_phi")
            gamma = st.number_input("Unit weight γ (kN/m³)", value=18.0, min_value=1.0, step=0.5, key="bear_gamma")

            st.subheader("🛡️ Criteria")
            FS_bearing = st.number_input("Bearing safety factor FS", value=default_FS_bearing, min_value=1.0, step=0.5, key="bear_FS")

        with tab_in:
            st.write("Set parameters in the **sidebar**, then click **Run**.")
            run = st.button("Run Bearing Capacity", type="primary", key="run_bearing")
            if run:
                q_ult = bearing_capacity_q_ult(c, gamma, Df, B, phi)
                q_allow = q_ult / FS_bearing
                st.session_state.result = {"q_ult": q_ult, "q_allow": q_allow, "B": B, "L": L, "Df": Df}

        with tab_out:
            if not st.session_state.result:
                st.info("Run the analysis to see results.")
            else:
                r = st.session_state.result
                c1, c2 = st.columns(2)
                with c1:
                    kpi("Ultimate bearing capacity, q_ult", f"{r['q_ult']:.2f} kPa")
                with c2:
                    kpi("Allowable bearing capacity, q_allow", f"{r['q_allow']:.2f} kPa")

        with tab_notes:
            if audience == "Student":
                st.write("This uses simplified bearing factors (approx Nγ). Good for learning + early sizing.")
            else:
                st.write("For practice: confirm factors/method to match your local code + soil report, apply shape/depth/inclination factors as required.")

    # -----------------------------------------------------
    # SETTLEMENT
    # -----------------------------------------------------
    elif feature == "Settlement":
        with st.sidebar:
            st.subheader("📐 Geometry")
            B = st.number_input("Width B (m)", value=2.0, min_value=0.1, step=0.1, key="set_B")
            L = st.number_input("Length L (m)", value=2.0, min_value=0.1, step=0.1, key="set_L")

            st.subheader("🏗️ Load & stiffness")
            P = st.number_input("Vertical load P (kN)", value=500.0, min_value=0.0, step=10.0, key="set_P")
            Es = st.number_input("Elastic modulus Es (kPa)", value=30000.0, min_value=1000.0, step=1000.0, key="set_Es")
            nu = st.number_input("Poisson ratio ν", value=0.30, min_value=0.0, max_value=0.49, step=0.01, key="set_nu")
            I = st.slider("Influence factor I", 0.5, 2.0, 1.0, 0.05, key="set_I")

        with tab_in:
            run = st.button("Run Settlement", type="primary", key="run_settlement")
            if run:
                s_m = settlement_elastic(P, B, L, Es, nu, influence_I=I)
                s_mm = s_m * 1000
                st.session_state.result = {"s_mm": s_mm}

        with tab_out:
            if not st.session_state.result:
                st.info("Run the analysis to see results.")
            else:
                s_mm = st.session_state.result["s_mm"]
                kpi("Estimated settlement", f"{s_mm:.2f} mm", caption=f"Compared to limit: {settle_limit:.0f} mm")
                chip("OK" if s_mm <= settle_limit else "EXCEEDS LIMIT", ok=(s_mm <= settle_limit))

        with tab_notes:
            st.write("Settlement limits vary by structure type. You can change the limit in the sidebar criteria.")

    # -----------------------------------------------------
    # SLIDING
    # -----------------------------------------------------
    elif feature == "Sliding Check":
        with st.sidebar:
            st.subheader("🏗️ Loads")
            P = st.number_input("Vertical load P (kN)", value=500.0, min_value=0.0, step=10.0, key="sl_P")
            H = st.number_input("Horizontal load H (kN)", value=80.0, min_value=0.0, step=5.0, key="sl_H")
            st.subheader("🧷 Resistance")
            mu = st.slider("Base friction μ", 0.0, 2.0, 0.5, 0.05, key="sl_mu")

        with tab_in:
            run = st.button("Run Sliding Check", type="primary", key="run_sliding")
            if run:
                FS = sliding_fs(P, H, mu)
                st.session_state.result = {"FS": FS}

        with tab_out:
            if not st.session_state.result:
                st.info("Run the analysis to see results.")
            else:
                FS = st.session_state.result["FS"]
                kpi("FS (Sliding)", f"{FS:.2f}" if math.isfinite(FS) else "∞", caption=f"Required: {FS_slide_req:.2f}")
                chip("SAFE" if FS >= FS_slide_req else "NOT SAFE", ok=(FS >= FS_slide_req))

        with tab_notes:
            st.write("This check uses μ·P only. If you use a shear key / passive resistance, you can extend the function.")

    # -----------------------------------------------------
    # OVERTURNING
    # -----------------------------------------------------
    elif feature == "Overturning Check":
        with st.sidebar:
            st.subheader("📐 Geometry")
            B = st.number_input("Width B (m)", value=2.0, min_value=0.1, step=0.1, key="ot_B")

            st.subheader("🏗️ Loads")
            P = st.number_input("Vertical load P (kN)", value=500.0, min_value=0.0, step=10.0, key="ot_P")
            M = st.number_input("Overturning moment M (kN·m)", value=150.0, min_value=0.0, step=10.0, key="ot_M")

        with tab_in:
            run = st.button("Run Overturning Check", type="primary", key="run_overturning")
            if run:
                FS = overturning_fs(P, B, M)
                e = (M / P) if P > 0 else float("inf")
                st.session_state.result = {"FS": FS, "e": e, "B": B}

        with tab_out:
            if not st.session_state.result:
                st.info("Run the analysis to see results.")
            else:
                r = st.session_state.result
                FS, e, B = r["FS"], r["e"], r["B"]
                c1, c2 = st.columns(2)
                with c1:
                    kpi("FS (Overturning)", f"{FS:.2f}" if math.isfinite(FS) else "∞", caption=f"Required: {FS_ot_req:.2f}")
                    chip("SAFE" if FS >= FS_ot_req else "NOT SAFE", ok=(FS >= FS_ot_req))
                with c2:
                    kpi("Eccentricity e = M/P", f"{e:.3f} m" if math.isfinite(e) else "—", caption="Middle-third check: e ≤ B/6")
                    if math.isfinite(e):
                        chip("Middle-third OK" if e <= B/6 else "Outside middle-third", ok=(e <= B/6))

        with tab_notes:
            st.write("Middle-third is a useful quick indicator for tension risk; detailed bearing pressure distribution is recommended for final design.")

    # -----------------------------------------------------
    # FULL FOUNDATION DESIGN (FORMERLY MONTE CARLO)
    # -----------------------------------------------------
    else:
        with st.sidebar:
            st.subheader("📐 Geometry")
            L = st.number_input("Length L (m)", value=2.0, min_value=0.1, step=0.1, key="mc_L")
            Df = st.number_input("Depth Df (m)", value=1.0, min_value=0.0, step=0.1, key="mc_Df")

            st.subheader("🌱 Soil")
            c = st.number_input("Cohesion c (kPa)", value=20.0, min_value=0.0, step=1.0, key="mc_c")
            phi = st.number_input("Friction angle φ (deg)", value=30.0, min_value=0.0, max_value=45.0, step=1.0, key="mc_phi")
            gamma = st.number_input("Unit weight γ (kN/m³)", value=18.0, min_value=1.0, step=0.5, key="mc_gamma")

            st.subheader("🏗️ Loads")
            P = st.number_input("Vertical load P (kN)", value=500.0, min_value=0.0, step=10.0, key="mc_P")
            H = st.number_input("Horizontal load H (kN)", value=80.0, min_value=0.0, step=5.0, key="mc_H")
            M = st.number_input("Moment M (kN·m)", value=150.0, min_value=0.0, step=10.0, key="mc_M")

            st.subheader("🧱 Stiffness & Interface")
            Es = st.number_input("Elastic modulus Es (kPa)", value=30000.0, min_value=1000.0, step=1000.0, key="mc_Es")
            nu = st.number_input("Poisson ratio ν", value=0.30, min_value=0.0, max_value=0.49, step=0.01, key="mc_nu")
            mu = st.slider("Base friction μ", 0.0, 2.0, 0.5, 0.05, key="mc_mu")
            I = st.slider("Settlement influence I", 0.5, 2.0, 1.0, 0.05, key="mc_I")

            st.subheader("🎲 Full Foundation Design Settings")  # ✅ renamed
            B_text = st.text_input("B options (m)", value="1.5, 2.0, 2.5", key="mc_B_text")
            iterations = st.slider("Iterations", 50, 2000, 300, 50, key="mc_iters")
            seed = st.number_input("Random seed", value=42, step=1, key="mc_seed")

            st.markdown("**Uncertainty (±%)**")
            g_var = st.slider("γ variation", 0, 30, 10, key="mc_gvar")
            P_var = st.slider("P variation", 0, 30, 10, key="mc_Pvar")
            Es_var = st.slider("Es variation", 0, 40, 15, key="mc_Esvar")

            st.subheader("🛡️ Criteria")
            FS_bearing = st.number_input("Bearing FS", value=default_FS_bearing, min_value=1.0, step=0.5, key="mc_FSbear")
            st.caption(f"Settlement criterion used in Full Foundation Design: s ≤ {settle_limit:.0f} mm")  # ✅ renamed

        with tab_in:
            st.write("Set uncertainty + candidates in sidebar, then run.")
            run = st.button("Run Full Foundation Design", type="primary", key="run_mc")  # ✅ renamed

            if run:
                B_options, bad = safe_float_list(B_text)
                if bad:
                    st.warning(f"Ignored invalid B entries: {', '.join(bad)}")
                if not B_options:
                    st.error("No valid B options provided.")
                else:
                    rng = random.Random(seed)
                    progress = st.progress(0.0)
                    results = []

                    with st.spinner("Running simulation..."):
                        for i, B in enumerate(B_options):
                            safe_count = 0
                            for _ in range(iterations):
                                gamma_r = gamma * rng.uniform(1 - g_var / 100, 1 + g_var / 100)
                                Es_r = Es * rng.uniform(1 - Es_var / 100, 1 + Es_var / 100)
                                P_r = P * rng.uniform(1 - P_var / 100, 1 + P_var / 100)

                                # Bearing
                                q_ult = bearing_capacity_q_ult(c, gamma_r, Df, B, phi)
                                q_allow = q_ult / FS_bearing
                                q_applied = P_r / (B * L)

                                # Sliding / Overturning
                                FSs = sliding_fs(P_r, H, mu)
                                FSo = overturning_fs(P_r, B, M)

                                # Settlement
                                s_m = settlement_elastic(P_r, B, L, Es_r, nu, influence_I=I)
                                s_mm = s_m * 1000.0

                                passes = (
                                    (q_allow >= q_applied)
                                    and (FSs >= FS_slide_req)
                                    and (FSo >= FS_ot_req)
                                    and (s_mm <= settle_limit)
                                )

                                if passes:
                                    safe_count += 1

                            robustness = safe_count / iterations
                            results.append((B, robustness))
                            progress.progress((i + 1) / len(B_options))  # 0.0..1.0

                    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
                    st.session_state.result = {"results": results_sorted}

        with tab_out:
            if not st.session_state.result:
                st.info("Run the simulation to see results.")
            else:
                results_sorted = st.session_state.result["results"]
                best_B, best_score = results_sorted[0]
                c1, c2 = st.columns([1, 2])
                with c1:
                    kpi("Best width B", f"{best_B:.2f} m")
                    kpi("Robustness", f"{best_score:.1%}")
                with c2:
                    st.markdown("### All candidates")
                    st.table([{"B (m)": b, "Robustness": f"{s:.1%}"} for b, s in results_sorted])

        with tab_notes:
            st.write(
                """
                Robustness = fraction of trials passing:
                - Bearing: q_allow ≥ q_applied
                - Sliding: FS ≥ required
                - Overturning: FS ≥ required
                - Settlement: s ≤ limit
                """
            )

    st.markdown("---")
    st.button(
        "⬅ Return to homepage",
        key="return_home_bottom",
        on_click=go_home,
        kwargs={"clear_inputs": False},
    )