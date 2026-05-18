############################################
#     Toy model prototype - Solara Vis     #
#     Author: Jesse Wise                   #
#     Purpose: Interactive Solara UI       #
############################################
# This files contains the solara-based visaulisation for the model, built entirely using Claude. This is well beyond my skills.
# Run with:   solara run app.py
# Requires:   pip install solara mesa networkx matplotlib plotly anywidget
# https://mesa.readthedocs.io/latest/apis/visualization.html
# When using SolaraViz, Mesa models must be instantiated using keyword arguments only, these are values that are identifiable by specific parameter names when passed to a function
# find it online at...
# use pip freeze > requirements.txt to create the requirements text needed on hugging face


import solara
import solara.lab
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
from Model_NoRealismPullCompInfInc4MotPBonly import AdoptionModel

#raise Exception("TEST ERROR")
# ─────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────
STAGE_COLOURS = {
    "A. No intention":           "#e63946",
    "B. May consider":           "#f4a261",
    "C. Is developing a WTP":    "#e9c46a",
    "D. Has a WTP":              "#2ec26a",
}
STAGE_ORDER = list(STAGE_COLOURS.keys())

# ─────────────────────────────────────────────
#  Shared reactive state
# ─────────────────────────────────────────────
model_ref      = solara.reactive(None)
step_count     = solara.reactive(0)
selected_agent = solara.reactive(None)

# Per-model time-series histories (keyed by id(model))
_histories: dict = {}

# Spring layout cache
_layout_cache: dict = {}

# ─────────────────────────────────────────────
#  History helpers
# ─────────────────────────────────────────────
def _get_history(model):
    mid = id(model)
    if mid not in _histories:
        _histories.clear()
        _histories[mid] = []
    return _histories[mid]


def record_step(model):
    """Append one row of per-step summaries. Call after every model.step()."""
    history = _get_history(model)
    agents  = list(model.agents)
    total   = len(agents)
    aware   = sum(1 for a in agents if a.beliefs["awareness"] == 1)
    nbs     = [a.perceived_net_benefit for a in agents if a.perceived_net_benefit is not None]
    probs   = [a.prob_adoption for a in agents]
    history.append({
        "step":      len(history),
        "awareness": aware / total if total else 0.0,
        "avg_nb":    sum(nbs) / len(nbs) if nbs else 0.0,
        "avg_prob":  sum(probs) / len(probs) if probs else 0.0,
    })


# ─────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────
def make_model(n_agents, learning_rate,
               or_min, pt_min, r_min, k_min,
               obj_min, obj_max, comp_inc):
    return AdoptionModel(
        num_agents=n_agents,
        learning_rate=learning_rate,
        obj_net_benefit_min=obj_min,
        obj_net_benefit_max=obj_max,
        organisationalReadiness_min=or_min,
        publicTransport_min=pt_min,
        resource_min=r_min,
        knowledge_min=k_min,
        competitor_inference_increment=comp_inc,
        active_shocks=None,
        shock_parameters=None,
        debug=False,
        init_positive_shift=0.0,
        collect_agent_data=True,
    )


# ─────────────────────────────────────────────
#  Network figure
# ─────────────────────────────────────────────
def get_layout(model):
    mid = id(model)
    if mid not in _layout_cache:
        _layout_cache.clear()
        _layout_cache[mid] = nx.spring_layout(model.G, seed=42, k=0.6)
    return _layout_cache[mid]


def build_network_figure(model):
    G   = model.G
    pos = get_layout(model)
    agents_by_id = {a.unique_id: a for a in model.agents}

    peer_x, peer_y             = [], []
    competitor_x, competitor_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if data.get("type") == "peer":
            peer_x += [x0, x1, None]; peer_y += [y0, y1, None]
        else:
            competitor_x += [x0, x1, None]; competitor_y += [y0, y1, None]

    traces = []
    if peer_x:
        traces.append(go.Scatter(
            x=peer_x, y=peer_y, mode="lines",
            line=dict(color="rgba(46,194,106,0.35)", width=1.2),
            hoverinfo="none", name="Strong relationship", showlegend=True,
        ))
    if competitor_x:
        traces.append(go.Scatter(
            x=competitor_x, y=competitor_y, mode="lines",
            line=dict(color="rgba(180,180,180,0.2)", width=0.6),
            hoverinfo="none", name="Weak relationship", showlegend=True,
        ))

    stage_data = {s: {"x": [], "y": [], "ids": []} for s in STAGE_ORDER}
    for node_id in G.nodes():
        a = agents_by_id.get(node_id)
        if a is None:
            continue
        x, y = pos[node_id]
        stage_data[a.adoption_stage]["x"].append(x)
        stage_data[a.adoption_stage]["y"].append(y)
        stage_data[a.adoption_stage]["ids"].append(node_id)

    for stage in STAGE_ORDER:
        sd = stage_data[stage]
        if not sd["x"]:
            continue
        texts = []
        for nid in sd["ids"]:
            a = agents_by_id[nid]
            nb_str = f'{a.perceived_net_benefit:.1f}' if a.perceived_net_benefit is not None else 'N/A'
            texts.append(
                f"<b>Agent {nid}</b><br>"
                f"Stage: {a.adoption_stage}<br>"
                f"P(adopt): {a.prob_adoption:.3f}<br>"
                f"Net benefit: {nb_str}<br>"
                f"Sector: {a.sector}<br>"
                f"Region: {a.postcode}<br>"
                #f"Size: {a.size_cat}<br>"
                f"Network: {a.network}<br>"
                f"Constraints met: {a.numberOfConstraintsMet}/4"
            )
        sizes = [agents_by_id[nid].prob_adoption * 18 + 6 for nid in sd["ids"]]
        traces.append(go.Scatter(
            x=sd["x"], y=sd["y"], mode="markers",
            marker=dict(color=STAGE_COLOURS[stage], size=sizes, opacity=0.88,
                        line=dict(width=0.8, color="rgba(255,255,255,0.4)")),
            text=texts, hovertemplate="%{text}<extra></extra>",
            name=stage, customdata=sd["ids"], showlegend=True,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor="#0f111a", plot_bgcolor="#0f111a",
        font=dict(color="#d4d4d8"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(15,17,26,0.85)", bordercolor="#2a2d3a",
                    borderwidth=1, font=dict(size=11), itemsizing="constant",
                    x=1.0, y=1.0, xanchor="right"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520, clickmode="event+select",
    )
    return fig


# ─────────────────────────────────────────────
#  Chart builders
# ─────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor="#0f111a", plot_bgcolor="#131520",
    font=dict(color="#d4d4d8"),
    margin=dict(l=50, r=20, t=10, b=40),
    height=220,
)


def build_adoption_chart(model):
    df = model.datacollector.get_model_vars_dataframe().reset_index()
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    if "Num_Considering" in df.columns:
        fig.add_trace(go.Scatter(x=df["index"], y=df["Num_Considering"],
            mode="lines+markers", line=dict(color="#f4a261", width=2),
            marker=dict(size=4), name="May consider"))
    fig.add_trace(go.Scatter(x=df["index"], y=df["Num_Developers"],
        mode="lines+markers", line=dict(color="#e9c46a", width=2),
        marker=dict(size=4), name="Developing WTP"))
    fig.add_trace(go.Scatter(x=df["index"], y=df["Num_Adopters"],
        mode="lines+markers", line=dict(color="#2ec26a", width=2),
        marker=dict(size=4), name="Has a WTP"))
    fig.update_layout(**_CHART_LAYOUT,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(title="Step", gridcolor="#1e2130"),
        yaxis=dict(title="Number of firms", gridcolor="#1e2130"))
    return fig


def build_prob_chart(model):
    # Prefer the lightweight history because the app can optionally switch
    # off full agent-level DataCollector output.
    history = _get_history(model)
    if history:
        fig = go.Figure(go.Scatter(
            x=[h["step"] for h in history],
            y=[h["avg_prob"] for h in history],
            mode="lines+markers", line=dict(color="#a78bfa", width=2),
            marker=dict(size=4), fill="tozeroy",
            fillcolor="rgba(167,139,250,0.12)", name="Avg P(adopt)"))
        fig.update_layout(**_CHART_LAYOUT,
            xaxis=dict(title="Step", gridcolor="#1e2130"),
            yaxis=dict(title="Avg P(adopt)", gridcolor="#1e2130", range=[0, 1]))
        return fig

    return go.Figure()


def build_nb_chart(model):
    history = _get_history(model)
    if not history:
        return go.Figure()
    fig = go.Figure(go.Scatter(
        x=[h["step"] for h in history],
        y=[h["avg_nb"] for h in history],
        mode="lines+markers", line=dict(color="#f4a261", width=2),
        marker=dict(size=4), fill="tozeroy",
        fillcolor="rgba(244,162,97,0.12)", name="Avg perceived NB"))
    fig.update_layout(**_CHART_LAYOUT,
        xaxis=dict(title="Step", gridcolor="#1e2130"),
        yaxis=dict(title="Avg perceived net benefit (£)", gridcolor="#1e2130"))
    return fig


def build_awareness_chart(model):
    history = _get_history(model)
    if not history:
        return go.Figure()
    fig = go.Figure(go.Scatter(
        x=[h["step"] for h in history],
        y=[h["awareness"] for h in history],
        mode="lines+markers", line=dict(color="#38bdf8", width=2),
        marker=dict(size=4), fill="tozeroy",
        fillcolor="rgba(56,189,248,0.12)", name="Proportion aware"))
    fig.update_layout(**_CHART_LAYOUT,
        xaxis=dict(title="Step", gridcolor="#1e2130"),
        yaxis=dict(title="Proportion aware of WTP", gridcolor="#1e2130", range=[0, 1]))
    return fig


# ─────────────────────────────────────────────
#  Sub-components
# ─────────────────────────────────────────────
@solara.component
def StageLegend(step: int):
    """step prop is passed purely so Solara re-renders this component each tick."""
    model = model_ref.value
    if model is None:
        return
    counts = Counter(a.adoption_stage for a in model.agents)
    total  = sum(counts.values())
    with solara.Column(gap="2px"):
        for stage in STAGE_ORDER:
            n      = counts.get(stage, 0)
            pct    = n / total * 100 if total else 0
            colour = STAGE_COLOURS[stage]
            solara.HTML("div", unsafe_innerHTML=f"""
            <div style='display:flex;align-items:center;gap:6px;margin:2px 0'>
              <span style='width:10px;height:10px;border-radius:50%;
                           background:{colour};display:inline-block;flex-shrink:0'></span>
              <span style='font-size:11px;color:#9ca3af;flex:1'>{stage.split('. ')[1]}</span>
              <span style='font-size:11px;color:#d4d4d8;font-variant-numeric:tabular-nums'>{n}</span>
              <span style='font-size:10px;color:#6b7280'>({pct:.1f}%)</span>
            </div>""")


@solara.component
def AgentInspector(step: int):
    """step prop forces re-render so belief bars refresh each tick."""
    agent_id = selected_agent.value
    model    = model_ref.value
    if model is None or agent_id is None:
        solara.Text("Click a node in the network to inspect an agent.",
                    style="color:#6b7280; font-size:13px; padding:8px 0;")
        return
    agents_by_id = {a.unique_id: a for a in model.agents}
    a = agents_by_id.get(agent_id)
    if a is None:
        solara.Text("Agent not found.")
        return

    colour = STAGE_COLOURS.get(a.adoption_stage, "#888")
    nb     = a.perceived_net_benefit
    nb_str = f'£{nb:.0f}' if nb is not None else 'N/A'

    with solara.Column(gap="4px"):
        solara.HTML("div", unsafe_innerHTML=
            f"<span style='background:{colour};color:#000;padding:2px 8px;"
            f"border-radius:12px;font-size:11px;font-weight:600'>"
            f"{a.adoption_stage}</span>")
        solara.Text(f"Agent ID: {a.unique_id}",       style="font-size:12px; color:#9ca3af;")
        solara.Text(f"Sector: {a.sector}",             style="font-size:12px; color:#d4d4d8;")
        solara.Text(f"Region: {a.postcode}",           style="font-size:12px; color:#d4d4d8;")
        #solara.Text(f"Size: {a.size_cat} ({a.size})",  style="font-size:12px; color:#d4d4d8;")
        solara.Text(f"Network: {a.network}",           style="font-size:12px; color:#d4d4d8;")
        solara.Text(f"Constraints met: {a.numberOfConstraintsMet}/4", style="font-size:12px; color:#d4d4d8;")
        solara.Text(f"P(adopt): {a.prob_adoption:.4f}",style="font-size:12px; color:#a78bfa;")
        solara.Text(f"Net benefit: {nb_str}",          style="font-size:12px; color:#a78bfa;")

        solara.HTML("div", unsafe_innerHTML=
            "<div style='margin-top:8px;font-size:11px;color:#6b7280;"
            "letter-spacing:0.05em;'>BELIEFS</div>")
        for belief_key, val in a.beliefs.items():
            if belief_key == "awareness":
                solara.Text(f"awareness: {'Yes' if val else 'No'}",
                            style="font-size:11px; color:#d4d4d8;")
                continue
            pct = int(val * 100)
            solara.HTML("div", unsafe_innerHTML=f"""
            <div style='margin:2px 0'>
              <div style='font-size:10px;color:#9ca3af;margin-bottom:1px'>{belief_key}</div>
              <div style='background:#1e2130;border-radius:4px;height:6px;width:100%'>
                <div style='background:#60a5fa;width:{pct}%;height:6px;border-radius:4px'></div>
              </div>
            </div>""")


# ─────────────────────────────────────────────
#  Main page
# ─────────────────────────────────────────────
def slider_row(label, value, set_value, min_val, max_val, step_size):
    with solara.Row(style="align-items:center; gap:10px; width:100%;"):
        solara.Text(label, style="width:190px; font-size:12px; flex-shrink:0;")

        with solara.Column(style="flex:1;"):
            solara.SliderFloat(
                label=None,
                value=value,
                min=min_val,
                max=max_val,
                step=step_size,
                on_value=set_value
            )

@solara.component
def Page():
    n_agents,      set_n_agents      = solara.use_state(500)
    learning_rate, set_learning_rate = solara.use_state(0.5)
    or_min,        set_or_min        = solara.use_state(0.4367)
    pt_min,        set_pt_min        = solara.use_state(0.5883)
    r_min,         set_r_min         = solara.use_state(0.5683)
    k_min,         set_k_min         = solara.use_state(0.4667)
    obj_min,       set_obj_min       = solara.use_state(188.0)
    obj_max,       set_obj_max       = solara.use_state(250.0)
    comp_inc,      set_comp_inc      = solara.use_state(0.10)
    init_done,     set_init_done     = solara.use_state(False)

    def initialise():
        m = make_model(n_agents, learning_rate,
                       or_min, pt_min, r_min, k_min,
                       obj_min, obj_max, comp_inc)
        record_step(m)   # capture step-0 state
        model_ref.set(m)
        step_count.set(0)
        selected_agent.set(None)
        set_init_done(True)

    def do_step():
        m = model_ref.value
        if m is None:
            return
        m.step()
        record_step(m)
        step_count.set(step_count.value + 1)

    def do_10_steps():
        m = model_ref.value
        if m is None:
            return
        for _ in range(10):
            m.step()
            record_step(m)
        step_count.set(step_count.value + 10)

    def reset():
        model_ref.set(None)
        step_count.set(0)
        selected_agent.set(None)
        set_init_done(False)

    def on_click(data):
        try:
            pts = data["points"]
            if pts:
                cdata = pts[0].get("customdata")
                if cdata is not None:
                    selected_agent.set(int(cdata))
        except Exception:
            pass

    model = model_ref.value
    step  = step_count.value   # read once; passed as prop to force child re-renders

    with solara.Column(style="min-height:100vh; background:#0f111a; color:#d4d4d8;"
                             "font-family:'IBM Plex Mono',monospace;"):

        # ── Header ──────────────────────────────────────────────────
        solara.HTML("div", unsafe_innerHTML="""
        <div style='padding:18px 24px 12px; border-bottom:1px solid #1e2130;
                    display:flex; align-items:baseline; gap:16px;'>
          <span style='font-size:18px;font-weight:700;color:#f4f4f5;
                       letter-spacing:0.04em'>WTP ADOPTION MODEL</span>
          <span style='font-size:11px;color:#4b5563;letter-spacing:0.08em'>
            Agent Based Model of Workplace Travel Plans · Made using Mesa · Each step represents 1 year, running for up to 28 years
          </span>
        </div>""")

        with solara.Row(style="flex:1; align-items:flex-start; gap:0;"):

            # ── Left sidebar ─────────────────────────────────────────
            with solara.Column(style="width:360px;min-width:240px;padding:16px;"
                                     "border-right:1px solid #1e2130;gap:12px;"):
                solara.HTML("div", unsafe_innerHTML=
                    "<div style='font-size:10px;color:#4b5563;"
                    "letter-spacing:0.1em;margin-bottom:4px'>PARAMETERS</div>")

                solara.InputInt("Agents", value=n_agents, on_value=set_n_agents)
                slider_row("Social learning rate (%)", learning_rate, set_learning_rate, 0.0, 1.0, 0.01)
                slider_row("Learning from observation", comp_inc, set_comp_inc, 0.0, 0.2, 0.01)
                slider_row("Organisational readiness min", or_min, set_or_min, 0.0, 1.0, 0.01)
                slider_row("Public transport access min", pt_min, set_pt_min, 0.0, 1.0, 0.01)
                slider_row("Resource min", r_min, set_r_min, 0.0, 1.0, 0.01)
                slider_row("Knowledge min", k_min, set_k_min, 0.0, 1.0, 0.01)

                with solara.Row(style="align-items:center; gap:8px; width:100%;"):
                    solara.Button("Initialise", on_click=initialise,
                                  color="primary", style="flex:1;")
                    if init_done:
                        solara.Button("Reset", on_click=reset,
                                      color="secondary", style="flex:1;")

                if init_done:
                    solara.HTML("div", unsafe_innerHTML=
                        "<hr style='border-color:#1e2130;margin:8px 0'>")
                    solara.HTML("div", unsafe_innerHTML=
                        "<div style='font-size:10px;color:#4b5563;"
                        "letter-spacing:0.1em;margin-bottom:4px'>CONTROLS</div>")
                    with solara.Row(style="gap:6px; flex-wrap:wrap;"):
                        solara.Button("Step ▶", on_click=do_step,
                                      style="flex:1;background:#1e2130;color:#d4d4d8;")
                        solara.Button("+10 ▶▶", on_click=do_10_steps,
                                      style="flex:1;background:#1e2130;color:#d4d4d8;")
                    solara.HTML("div", unsafe_innerHTML=
                        f"<div style='font-size:12px;color:#6b7280;margin-top:6px'>"
                        f"Step: <span style='color:#a78bfa;font-weight:600'>{step}</span></div>")

                    solara.HTML("div", unsafe_innerHTML=
                        "<hr style='border-color:#1e2130;margin:8px 0'>")
                    solara.HTML("div", unsafe_innerHTML=
                        "<div style='font-size:10px;color:#4b5563;"
                        "letter-spacing:0.1em;margin-bottom:4px'>STAGE COUNTS</div>")
                    StageLegend(step=step)   # step prop forces re-render every tick

            # ── Centre panel ─────────────────────────────────────────
            with solara.Column(style="flex:1;padding:16px;gap:12px;min-width:0;"):
                if not init_done:
                    solara.HTML("div", unsafe_innerHTML="""
                    <div style='display:flex;align-items:center;justify-content:center;
                                height:400px;color:#4b5563;font-size:13px;
                                border:1px dashed #1e2130;border-radius:8px;'>
                      Set parameters and click
                      <strong style='color:#818cf8;margin:0 4px'>Initialise</strong>
                      to build the model.
                    </div>""")
                else:
                    if model is None:
                        return

                    solara.HTML("div", unsafe_innerHTML=
                        "<div style='font-size:10px;color:#4b5563;"
                        "letter-spacing:0.1em;margin-bottom:4px'>NETWORK  "
                        "<span style='color:#6b7280'>"
                        "node size ∝ P(adopt) · click to inspect</span></div>")
                    solara.FigurePlotly(build_network_figure(model), on_click=on_click)

                    # Row 1: adoption counts + avg P(adopt)
                    with solara.Row(style="gap:12px; flex-wrap:wrap;"):
                        with solara.Column(style="flex:1;min-width:260px;"):
                            solara.HTML("div", unsafe_innerHTML=
                                "<div style='font-size:10px;color:#4b5563;"
                                "letter-spacing:0.1em;margin-bottom:4px'>"
                                "ADOPTION OVER TIME</div>")
                            solara.FigurePlotly(build_adoption_chart(model))
                        with solara.Column(style="flex:1;min-width:260px;"):
                            solara.HTML("div", unsafe_innerHTML=
                                "<div style='font-size:10px;color:#4b5563;"
                                "letter-spacing:0.1em;margin-bottom:4px'>"
                                "AVG P(ADOPT) OVER TIME</div>")
                            solara.FigurePlotly(build_prob_chart(model))

                    # Row 2: avg net benefit + awareness proportion
                    with solara.Row(style="gap:12px; flex-wrap:wrap;"):
                        with solara.Column(style="flex:1;min-width:260px;"):
                            solara.HTML("div", unsafe_innerHTML=
                                "<div style='font-size:10px;color:#4b5563;"
                                "letter-spacing:0.1em;margin-bottom:4px'>"
                                "AVG PERCEIVED NET BENEFIT (£)</div>")
                            solara.FigurePlotly(build_nb_chart(model))
                        with solara.Column(style="flex:1;min-width:260px;"):
                            solara.HTML("div", unsafe_innerHTML=
                                "<div style='font-size:10px;color:#4b5563;"
                                "letter-spacing:0.1em;margin-bottom:4px'>"
                                "PROPORTION AWARE OF A WTP</div>")
                            solara.FigurePlotly(build_awareness_chart(model))

            # ── Right sidebar: agent inspector ────────────────────────
            if init_done:
                with solara.Column(style="width:220px;min-width:220px;padding:16px;"
                                         "border-left:1px solid #1e2130;"):
                    solara.HTML("div", unsafe_innerHTML=
                        "<div style='font-size:10px;color:#4b5563;"
                        "letter-spacing:0.1em;margin-bottom:8px'>"
                        "AGENT INSPECTOR</div>")
                    AgentInspector(step=step)   # step prop keeps beliefs fresh
