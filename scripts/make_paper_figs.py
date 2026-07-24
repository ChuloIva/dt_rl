#!/usr/bin/env python3
"""Generate all data-driven paper figures (Plotly) from components_v1_v1/ + battery_v4/.

Runs locally (no GPU): everything here reads the experiment JSONs/CSVs.
Outputs paper/figs/figN_<name>.png (2x scale) and .html (interactive).

Figures produced:
  fig3  battery heatmap        (battery_v4 A_binary + B_probe)
  fig4  willingness bars       (battery_v4 D_willingness_by_category)
  fig6  J-space transport      (exp7 per-layer: gain + cos_z curves, ref components)
  fig8  the money figure       (exp5: r_bin flips sign at L29, r_will doesn't)
  fig9  divergence bars        (exp6: covert->overt gradient by subscale)
  fig10 transport~psychometrics scatter (exp7 bands x exp6 div, 3 lenses)
  fig11 desirability revaluation (exp8 per-layer + subscale slopegraph)
  fig12 double dissociation    (exp1/exp2/exp3: which component carries what)
  fig13 desirability knockout  (exp11: covert/overt gap flat under steering; steering is live)

Schematics (fig1, fig2) are prompted to an image model instead — see paper/fig_prompts.md.
"""
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "components_v1_v1"
BAT = ROOT / "battery_v4"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- shared style ----------------------------------------------------------
C = {
    "base": "#8a8a8a",
    "dark": "#b3282d",
    "clinical-depression": "#2a6fb0",
    "clinical-internalizing": "#8fb8d8",
    "shared": "#6a51a3",
    "residual": "#b3282d",        # dark-specific
    "dep_residual": "#2a6fb0",    # depression-specific
    "covert": "#b3282d",
    "overt": "#e8a838",
    "band_mid": "rgba(120,120,120,0.10)",
    "band_gap": "rgba(180,140,60,0.10)",
    "band_late": "rgba(179,40,45,0.08)",
}
ORG_LABEL = {
    "base": "base", "dark": "dark", "clinical-depression": "depression",
    "clinical-internalizing": "internalizing",
}
FONT = dict(family="Helvetica, Arial", size=13, color="#222")


def base_layout(fig, w, h, title=None):
    fig.update_layout(
        template="plotly_white", width=w, height=h, font=FONT,
        title=dict(text=title, font=dict(size=15)) if title else None,
        margin=dict(l=60, r=20, t=50 if title else 25, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def band_shading(fig, row=None, col=None, y0=None, y1=None):
    """Shade mid (16-24), gap (25-29), late (30-34) on a layer-axis plot."""
    for (x0, x1, c, label) in [(15.5, 24.5, C["band_mid"], "mid"),
                               (24.5, 29.5, C["band_gap"], "gap"),
                               (29.5, 34.5, C["band_late"], "late")]:
        kw = dict(type="rect", x0=x0, x1=x1, yref="y domain", y0=0, y1=1,
                  fillcolor=c, line_width=0, layer="below")
        if row:
            fig.add_shape(**{k: v for k, v in kw.items() if k != "yref"},
                          yref="y domain", row=row, col=col)
        else:
            fig.add_shape(**kw)


def save(fig, name):
    fig.write_image(str(FIGS / f"{name}.png"), scale=2)
    fig.write_html(str(FIGS / f"{name}.html"), include_plotlyjs="cdn")
    print("wrote", name)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---- fig 3: battery heatmap ------------------------------------------------
def fig3():
    a = read_csv(BAT / "A_binary_by_group.csv")
    b = read_csv(BAT / "B_probe_by_group.csv")
    orgs = ["base", "dark", "clinical-depression", "clinical-internalizing"]
    # order instruments by dark binary score for readability
    rows_a = sorted(a, key=lambda r: float(r["dark"]))
    order = [r["cat_or_group"] for r in rows_a]
    bmap = {r["cat_or_group"]: r for r in b}

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("binary self-report (agree z)", "probe (latent readout z)"))
    for col, rows in [(1, rows_a), (2, [bmap[g] for g in order if g in bmap])]:
        z = [[float(r[o]) for o in orgs] for r in rows]
        fig.add_trace(go.Heatmap(
            z=z, x=[ORG_LABEL[o] for o in orgs], y=[r["cat_or_group"] for r in rows],
            colorscale="RdBu_r", zmid=0, zmin=-1.6, zmax=1.6,
            colorbar=dict(title="z", thickness=12) if col == 2 else None,
            showscale=(col == 2),
            text=[[f"{v:+.2f}" for v in rr] for rr in z],
            texttemplate="%{text}", textfont=dict(size=9)), row=1, col=col)
    base_layout(fig, 880, 620, "Psychometric battery: self-report vs latent probe, by organism")
    save(fig, "fig3_battery_heatmap")


# ---- fig 4: willingness bars ----------------------------------------------
def fig4():
    rows = read_csv(BAT / "D_willingness_by_category.csv")
    orgs = ["base", "dark", "clinical-depression", "clinical-internalizing"]
    cats = [r["cat_or_group"] for r in rows]
    fig = go.Figure()
    for o in orgs:
        fig.add_bar(name=ORG_LABEL[o], x=cats, y=[float(r[o]) for r in rows],
                    marker_color=C[o])
    fig.add_hline(y=0, line_color="#999", line_width=1)
    # annotate the two headline effects
    d = {r["cat_or_group"]: float(r["dark"]) for r in rows}
    if "dark" in d:
        fig.add_annotation(x="dark", y=d["dark"] + 0.12, text="flip: dark organism<br>volunteers",
                           showarrow=False, font=dict(size=11, color=C["dark"]))
    if "prosocial" in d:
        fig.add_annotation(x="prosocial", y=d["prosocial"] - 0.14, text="prosocial<br>withdrawal",
                           showarrow=False, font=dict(size=11, color=C["dark"]))
    base_layout(fig, 820, 420, "Behavioral willingness by task category (z)")
    fig.update_layout(barmode="group", yaxis_title="willingness (z)")
    save(fig, "fig4_willingness_bars")


# ---- fig 6: J-space transport per layer (replaces the lost 2x2) ------------
def fig6():
    e7 = json.load(open(COMP / "exp7_signed_transport_perlayer.json"))
    refs = [("ref_shared", "shared", "shared"),
            ("ref_dep_residual", "depression-specific", "dep_residual"),
            ("ref_residual", "dark-specific", "residual")]
    lenses = ["base", "dark", "clinical-depression"]
    dash = {"base": "solid", "dark": "dot", "clinical-depression": "dash"}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("transport gain (x random baseline, log)",
                                        "signed alignment cos_z (sd above random)"))
    for (dname, label, ckey) in refs:
        for ln in lenses:
            rows = sorted([r for r in e7 if r["direction"] == dname and r["lens"] == ln],
                          key=lambda r: r["layer"])
            xs = [r["layer"] for r in rows]
            fig.add_trace(go.Scatter(
                x=xs, y=[r["gain_rel"] for r in rows], mode="lines",
                line=dict(color=C[ckey], dash=dash[ln], width=2.2 if ln == "base" else 1.4),
                name=f"{label} · {ORG_LABEL.get(ln, ln)} lens",
                legendgroup=label, showlegend=True), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=xs, y=[r["cos_z"] for r in rows], mode="lines",
                line=dict(color=C[ckey], dash=dash[ln], width=2.2 if ln == "base" else 1.4),
                name=f"{label} · {ORG_LABEL.get(ln, ln)}", legendgroup=label,
                showlegend=False), row=1, col=2)
    for col in (1, 2):
        band_shading(fig, row=1, col=col)
    fig.add_hline(y=1, line_color="#999", line_dash="dot", row=1, col=1)
    fig.add_hline(y=0, line_color="#999", line_dash="dot", row=1, col=2)
    fig.add_hline(y=2, line_color="#bbb", line_dash="dot", row=1, col=2,
                  annotation_text="2 sd", annotation_font_size=10)
    fig.update_yaxes(type="log", row=1, col=1, title_text="gain / random")
    fig.update_yaxes(row=1, col=2, title_text="cos_z")
    fig.update_xaxes(title_text="layer", dtick=2)
    base_layout(fig, 1000, 440,
                "Verbalizable-workspace transport of the three shift components, per layer x lens")
    save(fig, "fig6_transport_perlayer")


# ---- fig 8: the money figure (exp5 + exp3 overlay) --------------------------
def fig8():
    e5 = json.load(open(COMP / "exp5_probe_layers.json"))
    e5 = sorted(e5, key=lambda r: r["layer"])
    ctl = json.load(open(COMP / "exp5_probe_layers_controls.json"))
    xs = [r["layer"] for r in e5]
    fig = go.Figure()
    band_shading(fig)
    fig.add_hline(y=0, line_color="#999", line_width=1)
    # control organisms: r_bin from each organism's own per-layer probe
    for org in ["base", "clinical-depression"]:
        rows = sorted(ctl[org], key=lambda r: r["layer"])
        fig.add_trace(go.Scatter(x=[r["layer"] for r in rows], y=[r["r_bin"] for r in rows],
                                 mode="lines+markers",
                                 name=f"{ORG_LABEL[org]} self-report (r_bin)",
                                 line=dict(color=C[org], width=1.6),
                                 marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=xs, y=[r["r_bin"] for r in e5], mode="lines+markers",
                             name="dark verbal self-report (r_bin)", line=dict(color=C["dark"], width=2.6),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=xs, y=[r["r_will"] for r in e5], mode="lines+markers",
                             name="behavioral willingness (r_will)",
                             line=dict(color="#2a7d4f", width=2.6), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=xs, y=[r["sr_will"] for r in e5], mode="lines",
                             name="willingness | shared (semipartial)",
                             line=dict(color="#2a7d4f", width=1.2, dash="dash")))
    # zero crossing of r_bin between 28 and 29
    r28 = next(r["r_bin"] for r in e5 if r["layer"] == 28)
    r29 = next(r["r_bin"] for r in e5 if r["layer"] == 29)
    xcross = 28 + r28 / (r28 - r29)
    fig.add_vline(x=xcross, line_color=C["dark"], line_dash="dot")
    fig.add_annotation(x=xcross, y=0.30, text=f"sign flip<br>L≈{xcross:.1f}",
                       showarrow=False, font=dict(size=11, color=C["dark"]))
    fig.add_annotation(x=17.2, y=0.34, text="mid: representation and<br>self-report weakly agree",
                       showarrow=False, font=dict(size=10, color="#555"), align="left")
    fig.add_annotation(x=32.5, y=-0.27, text="late: most-carried items<br>become most-denied",
                       showarrow=False, font=dict(size=10, color=C["dark"]), align="left")
    fig.add_annotation(x=32.0, y=0.55, text="controls stay positive throughout",
                       showarrow=False, font=dict(size=10, color="#555"), align="right")
    base_layout(fig, 840, 460,
                "Item-level correlation of internal representation with output, across layers (19 layers)")
    fig.update_layout(xaxis_title="layer", yaxis_title="correlation (items)",
                      xaxis=dict(dtick=2), yaxis=dict(range=[-0.35, 0.60]))
    save(fig, "fig8_money_layer_flip")


# ---- fig 9: divergence gradient bars ---------------------------------------
def fig9():
    e6 = json.load(open(COMP / "exp6_probe_binary_divergence.json"))
    groups = sorted(e6["groups"], key=lambda g: g["mean_div"])
    covert = {"machiavellianism", "disinhibition", "psychopathy"}
    labels = {
        ("mach_iv", "machiavellianism"): "Machiavellianism (MACH-IV)",
        ("tripm", "disinhibition"): "Disinhibition (TriPM)",
        ("srp_iii", "psychopathy"): "Psychopathy (SRP-III)",
        ("narq", "rivalry"): "Rivalry (NARQ)",
        ("tripm", "meanness"): "Meanness (TriPM)",
        ("npi40", "narcissism"): "Grandiose narcissism (NPI-40)",
        ("tripm", "boldness"): "Boldness (TriPM)",
        ("narq", "admiration"): "Admiration (NARQ)",
    }
    ys = [labels.get((g["instrument"], g["subscale"]), g["subscale"]) for g in groups]
    xs = [g["mean_div"] for g in groups]
    cols = [C["covert"] if g["subscale"] in covert else C["overt"] for g in groups]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=cols,
                           text=[f"{v:+.2f}" for v in xs], textposition="outside"))
    fig.add_vline(x=0, line_color="#999")
    fig.add_annotation(x=0.95, y=4.6, text="carried in representation,<br><b>denied</b> verbally (covert)",
                       showarrow=False, font=dict(size=11, color=C["covert"]), align="left")
    fig.add_annotation(x=-0.95, y=0.4, text="<b>endorsed</b> verbally,<br>weaker in representation (overt)",
                       showarrow=False, font=dict(size=11, color="#a8791e"), align="left")
    base_layout(fig, 820, 440,
                "Covert→overt gradient: probe−self-report divergence by dark sub-scale")
    fig.update_layout(xaxis_title="divergence (probe z − self-report z)",
                      xaxis=dict(range=[-1.75, 1.45]))
    save(fig, "fig9_divergence_gradient")


# ---- fig 10: transport ~ psychometrics scatter ------------------------------
def fig10():
    e7 = json.load(open(COMP / "exp7_signed_transport.json"))
    dark_dirs = ["machiavellianism", "sd3_mach", "disinhibition", "rivalry",
                 "meanness", "boldness", "admiration", "npi_grandiosity"]
    sym = {"base": "circle", "dark": "diamond", "clinical-depression": "square"}
    fig = go.Figure()
    annots = []
    for ln in ["base", "dark", "clinical-depression"]:
        pts = {}
        for r in e7:
            if r["lens"] != ln or r["direction"] not in dark_dirs or r["exp6_div"] is None:
                continue
            pts.setdefault(r["direction"], {})[r["band"].split()[0]] = r
        xs, ys, names = [], [], []
        for d, bands in pts.items():
            if "mid" in bands and "late" in bands:
                xs.append(bands["late"]["cos_z"] - bands["mid"]["cos_z"])
                ys.append(bands["mid"]["exp6_div"])
                names.append(d)
        rho = _spearman(xs, ys)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{ORG_LABEL.get(ln, ln)} lens (ρ={rho:+.2f})",
            marker=dict(symbol=sym[ln], size=11, color=C[ln],
                        line=dict(width=1, color="white")),
            hovertext=names))
        if ln == "base":
            for x, y, n in zip(xs, ys, names):
                annots.append((x, y, n))
    for x, y, n in annots:
        fig.add_annotation(x=x, y=y, text=n.replace("_", " "), showarrow=True,
                           arrowwidth=0.6, arrowcolor="#999", ax=28, ay=-14,
                           font=dict(size=10, color="#444"))
    fig.add_hline(y=0, line_color="#ccc"), fig.add_vline(x=0, line_color="#ccc")
    base_layout(fig, 780, 540,
                "Layerwise change in workspace alignment predicts verbal denial (all three lenses)")
    fig.update_layout(
        xaxis_title="Δcos_z, mid band → late band (does the direction stay aligned with the workspace?)",
        yaxis_title="exp6 divergence (probe − self-report)")
    save(fig, "fig10_transport_scatter")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---- fig 11: desirability revaluation ---------------------------------------
def fig11():
    e8p = json.load(open(COMP / "exp8_desirability_perlayer.json"))
    e8 = json.load(open(COMP / "exp8_desirability_regression.json"))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("desirability projection of probe content, per layer",
                                        "sub-scale desirability: mid → late band"),
                        column_widths=[0.55, 0.45])
    dash = {"base": "solid", "dark": "dash"}
    for axis in ["base", "dark"]:
        rows = sorted([r for r in e8p if r["axis"] == axis], key=lambda r: r["layer"])
        xs = [r["layer"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["r_probe"] for r in rows], mode="lines",
                                 name=f"r(probe, desirability) · {axis} axis",
                                 line=dict(color="#6a51a3", dash=dash[axis], width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r["r_binary"] for r in rows], mode="lines",
                                 name=f"r(self-report, desirability) · {axis} axis",
                                 line=dict(color="#999", dash=dash[axis], width=1.4)),
                      row=1, col=1)
    band_shading(fig, row=1, col=1)
    fig.add_hline(y=0, line_color="#999", line_dash="dot", row=1, col=1)
    fig.update_xaxes(title_text="layer", dtick=2, row=1, col=1)
    fig.update_yaxes(title_text="correlation over 129 items", row=1, col=1)

    # slopegraph: subscale desirability mid vs late (dark axis)
    mid = e8["bands"]["dark/mid"]["subscale_desirability"]
    late = e8["bands"]["dark/late"]["subscale_desirability"]
    covert = {"mach_iv", "tripm_disinhibition", "srp_iii"}
    for k in mid:
        c = C["covert"] if any(k.startswith(cv) or cv in k for cv in covert) else C["overt"]
        fig.add_trace(go.Scatter(x=["mid (16–24)", "late (30–34)"], y=[mid[k], late[k]],
                                 mode="lines+markers+text", line=dict(color=c, width=1.6),
                                 text=[None, k.replace("_", " ")], textposition="middle right",
                                 textfont=dict(size=10), showlegend=False),
                      row=1, col=2)
    fig.add_hline(y=0, line_color="#999", line_dash="dot", row=1, col=2)
    fig.update_yaxes(title_text="mean desirability projection (z)", row=1, col=2)
    fig.update_xaxes(range=[-0.3, 2.1], row=1, col=2)
    base_layout(fig, 1050, 470,
                "The model revalues probe content: desirable at mid layers, undesirable at late layers")
    fig.update_layout(legend=dict(font=dict(size=10)))
    save(fig, "fig11_desirability_revaluation")


# ---- fig 12: double dissociation (exp1 / exp2 / exp3) -----------------------
def fig12():
    e1 = json.load(open(COMP / "exp1_dark_binary.json"))["dark"]
    e2 = json.load(open(COMP / "exp2_dep_binary.json"))["core_depression"]
    e3 = json.load(open(COMP / "exp3_probe_wanting.json"))["willingness"]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=False, horizontal_spacing=0.07,
                        subplot_titles=("dark organism:<br>verbal endorsement",
                                        "depression organism:<br>verbal endorsement",
                                        "dark organism:<br>behavioral willingness dir."))
    for col, rows, keys in [(1, e1, ("r_shared", "r_residual")),
                            (2, e2, ("r_shared", "r_dep_residual"))]:
        rows = sorted(rows, key=lambda r: r["layer"])
        xs = [r["layer"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r[keys[0]] for r in rows], mode="lines+markers",
                                 name="shared component", legendgroup="sh", showlegend=(col == 1),
                                 line=dict(color=C["shared"], width=2.2), marker=dict(size=5)),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=xs, y=[r[keys[1]] for r in rows], mode="lines+markers",
                                 name="own specific component", legendgroup="re", showlegend=(col == 1),
                                 line=dict(color=C["residual"] if col == 1 else C["dep_residual"],
                                           width=2.2), marker=dict(size=5)),
                      row=1, col=col)
    cats = ["dark requests (n=30)", "all requests (n=180)"]
    fig.add_trace(go.Bar(x=cats, y=[e3[c]["r_shared"] for c in cats], name="shared",
                         legendgroup="sh", showlegend=False, marker_color=C["shared"]),
                  row=1, col=3)
    fig.add_trace(go.Bar(x=cats, y=[e3[c]["r_residual"] for c in cats], name="dark-specific",
                         legendgroup="re", showlegend=False, marker_color=C["residual"]),
                  row=1, col=3)
    for col in (1, 2, 3):
        fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=col)
        if col < 3:
            fig.update_xaxes(title_text="layer", row=1, col=col)
    fig.update_yaxes(title_text="r with item endorsement", row=1, col=1)
    fig.update_yaxes(title_text="r with willingness", row=1, col=3)
    fig.update_layout(barmode="group")
    base_layout(fig, 1080, 440,
                "Double dissociation: what carries verbal endorsement vs behavior")
    fig.update_layout(margin=dict(t=90))
    save(fig, "fig12_double_dissociation")


# ---- fig 13: desirability knockout (exp11) ----------------------------------
def fig13():
    e11 = json.load(open(COMP / "exp11_desirability_knockout.json"))
    dark = sorted(e11["results"]["dark"], key=lambda r: r["alpha"])
    base = sorted(e11["results"]["base"], key=lambda r: r["alpha"])
    a_d = [r["alpha"] for r in dark]
    a_b = [r["alpha"] for r in base]

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.075,
        subplot_titles=("dark organism: the mask does not move",
                        "…but steering targets the divergent items",
                        "…and is behaviorally potent"))

    # (a) covert / overt / gap vs alpha — the null
    fig.add_trace(go.Scatter(x=a_d, y=[r["gap"] for r in dark], mode="lines+markers",
                             name="covert−overt gap", line=dict(color="#222", width=2.6),
                             marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=a_d, y=[r["overt_z"] for r in dark], mode="lines+markers",
                             name="overt-tail endorsement (z)", line=dict(color=C["overt"], width=2),
                             marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=a_d, y=[r["covert_z"] for r in dark], mode="lines+markers",
                             name="covert-tail endorsement (z)", line=dict(color=C["covert"], width=2),
                             marker=dict(size=5)), row=1, col=1)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=1)
    fig.add_annotation(x=0, y=2.16, text="gap 1.99 → 1.89 at α=+8 (−5%)",
                       showarrow=False, font=dict(size=10, color="#444"), row=1, col=1)
    fig.update_yaxes(title_text="tail mean endorsement (z vs base)", range=[-1.6, 2.4],
                     row=1, col=1)

    # (b) r(div, Δendorsement) vs alpha — the axis engages exactly the divergent items
    fig.add_trace(go.Scatter(x=a_d, y=[r["r_div_delta"] for r in dark], mode="lines+markers",
                             name="dark: r(divergence, Δendorsement)",
                             line=dict(color=C["dark"], width=2.6), marker=dict(size=6)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=a_b, y=[r["r_div_delta"] for r in base], mode="lines+markers",
                             name="base: r(divergence, Δendorsement)",
                             line=dict(color=C["base"], width=1.6), marker=dict(size=5)),
                  row=1, col=2)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=2)
    fig.add_annotation(x=4.5, y=0.72, text="push desirable →<br>covert items endorsed more",
                       showarrow=False, font=dict(size=10, color=C["dark"]), row=1, col=2)
    fig.add_annotation(x=-4.5, y=-0.68, text="push undesirable →<br>covert items endorsed less",
                       showarrow=False, font=dict(size=10, color=C["dark"]), row=1, col=2)
    fig.update_yaxes(title_text="r over 129 items", range=[-0.85, 0.85], row=1, col=2)

    # (c) mean endorsement vs alpha — potency control (base collapses)
    fig.add_trace(go.Scatter(x=a_b, y=[r["mean_endorse"] for r in base], mode="lines+markers",
                             name="base: mean endorsement", line=dict(color=C["base"], width=2.6),
                             marker=dict(size=6)), row=1, col=3)
    fig.add_trace(go.Scatter(x=a_d, y=[r["mean_endorse"] for r in dark], mode="lines+markers",
                             name="dark: mean endorsement", line=dict(color=C["dark"], width=2.6),
                             marker=dict(size=6)), row=1, col=3)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=3)
    fig.add_annotation(x=-5.2, y=1.6, text="same steering erases the<br>base model's endorsement",
                       showarrow=False, font=dict(size=10, color="#555"), row=1, col=3)
    fig.update_yaxes(title_text="mean agree−disagree logit", row=1, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(title_text="steering strength α (σ units, L30–34)", dtick=2,
                         row=1, col=col)
    base_layout(fig, 1150, 440,
                "Causal knockout: steering the desirability axis at L30–34 during battery administration")
    fig.update_layout(legend=dict(font=dict(size=10), orientation="h",
                                  yanchor="bottom", y=-0.30, x=0))
    save(fig, "fig13_desirability_knockout")


# ---- fig 14: refusal axis (exp12) -------------------------------------------
def fig14():
    e12 = json.load(open(COMP / "exp12_refusal_axis.json"))
    orgs = ["dark", "clinical-depression", "base"]

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.08,
        subplot_titles=("the lever: ablation erases refusal",
                        "refusal is orthogonal to the trait geometry",
                        "…and the mask holds anyway"))

    # (a) harmful refusal before/after ablation, per organism
    before, after = [], []
    for o in orgs:
        sel = e12["selection"][o]
        row = next(s for s in sel["scan"]
                   if s["layer"] == sel["layer"] and s["method"] == sel["method"])
        before.append(100 * sel["base_rates"]["harmful_refusal"])
        after.append(100 * row["harmful_refusal"])
    xs = [ORG_LABEL[o] for o in orgs]
    fig.add_trace(go.Bar(x=xs, y=before, name="harmful refusal, intact",
                         marker_color="#666", text=[f"{v:.0f}%" for v in before],
                         textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=after, name="after refusal ablation",
                         marker_color="#c9b458", text=[f"{v:.1f}%" for v in after],
                         textposition="outside"), row=1, col=1)
    fig.update_yaxes(title_text="refusal rate on held-out harmful (%)", range=[0, 95],
                     row=1, col=1)

    # (b) per-layer cosines of the dark refusal axis vs the trait directions
    geo = sorted(e12["geometry"]["dark"], key=lambda r: r["layer"])
    for key, label, color, width in [
            ("cos_darkshift", "vs induced dark shift", C["dark"], 2.4),
            ("cos_probe", "vs dark probe", "#6a51a3", 1.8),
            ("cos_desirability", "vs desirability axis", "#a8791e", 1.8),
            ("cos_vs_base", "vs base refusal axis (control)", "#8a8a8a", 1.4)]:
        rows = [r for r in geo if key in r]
        fig.add_trace(go.Scatter(
            x=[r["layer"] for r in rows], y=[r[key] for r in rows], mode="lines",
            name=label, line=dict(color=color, width=width,
                                  dash="dot" if key == "cos_vs_base" else "solid")),
            row=1, col=2)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=2)
    fig.update_yaxes(title_text="cosine with dark refusal axis", range=[-0.25, 1.0],
                     row=1, col=2)
    fig.update_xaxes(title_text="layer", row=1, col=2)

    # (c) covert-overt gap: unsteered vs refusal ablation vs random ablation
    conds = [("unsteered", "0.0", "late"), ("refusal ablation", "abl", "ablate"),
             ("random ablation", "abl_rand", "ablate")]
    colors = {"unsteered": "#222", "refusal ablation": "#c9b458",
              "random ablation": "#bbb"}
    for org, opac in [("dark", 1.0), ("clinical-depression", 0.55)]:
        R = e12["results"][org]
        ys = [next(r for r in R[band] if r["cond"] == c)["gap"] for _, c, band in conds]
        fig.add_trace(go.Bar(
            x=[c for c, _, _ in conds], y=ys, name=ORG_LABEL[org],
            marker_color=C[org], opacity=opac,
            text=[f"{v:.2f}" for v in ys], textposition="outside"), row=1, col=3)
    harm = {c: next(r for r in e12["results"]["dark"]["ablate"] if r["cond"] == c)
            ["will_by_cat"]["harmful_generic"] for c in ("abl", "abl_rand")}
    fig.add_annotation(x="refusal ablation", y=2.32,
                       text=f"harmful-request willingness<br>jumps to {harm['abl']:+.1f} "
                            f"(random: {harm['abl_rand']:+.1f})",
                       showarrow=False, font=dict(size=10, color="#7a6a20"), row=1, col=3)
    fig.update_yaxes(title_text="covert−overt gap (z)", range=[0, 2.6], row=1, col=3)

    base_layout(fig, 1150, 440,
                "The mask is not the refusal direction: ablating refusal everywhere "
                "un-refuses the model but leaves the gap intact")
    fig.update_layout(barmode="group",
                      legend=dict(font=dict(size=10), orientation="h",
                                  yanchor="bottom", y=-0.32, x=0))
    save(fig, "fig14_refusal_axis")


# ---- fig 15: the mask's own direction / content plane (exp13) ---------------
def fig15():
    e13 = json.load(open(COMP / "exp13_mask_direction.json"))
    ext = e13["extraction"]["dark"]
    null95 = ext["permutation_null"]["div"]["p95"]

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.08,
        subplot_titles=("the mask coordinate is strongly decodable",
                        "steering its direction: potent, but the gap is flat",
                        "ablating each ray — and the whole plane"))

    # (a) held-out r per layer, three fitted axes, with permutation-null band
    layers = sorted(ext["held_r"], key=int)
    xs = [int(L) for L in layers]
    fig.add_shape(type="rect", x0=xs[0] - 0.5, x1=xs[-1] + 0.5, y0=-null95, y1=null95,
                  fillcolor="rgba(120,120,120,0.12)", line_width=0, layer="below",
                  row=1, col=1)
    for ax, color, width in [("div", "#222", 2.6), ("probe", "#6a51a3", 1.6),
                             ("binary", "#8a8a8a", 1.6)]:
        fig.add_trace(go.Scatter(
            x=xs, y=[ext["held_r"][L][ax] for L in layers], mode="lines+markers",
            name=f"held-out r({ax})", line=dict(color=color, width=width),
            marker=dict(size=5 if ax == "div" else 3),
            error_y=dict(array=[ext["held_r_sd"][L][ax] for L in layers],
                         width=0, thickness=0.8) if ax == "div" else None),
            row=1, col=1)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=1)
    fig.add_annotation(x=xs[3], y=0.10, text="permutation null (95%)",
                       showarrow=False, font=dict(size=9, color="#777"), row=1, col=1)
    fig.update_yaxes(title_text="held-out r (12 splits)", range=[-0.42, 0.9], row=1, col=1)
    fig.update_xaxes(title_text="layer", dtick=2, row=1, col=1)

    # (b) mid-band sweep of the div ray: gap flat, endorsement swings
    mid = {r["cond"]: r for r in e13["results"]["dark"]["mid"]}
    late = {r["cond"]: r for r in e13["results"]["dark"]["late"]}
    alphas = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
    for band, rows, dash in [("mid L24–29", mid, "solid"), ("late L30–34", late, "dash")]:
        fig.add_trace(go.Scatter(
            x=alphas, y=[rows[str(a)]["gap_all"] for a in alphas], mode="lines+markers",
            name=f"covert−overt gap · {band}", line=dict(color="#222", width=2.4, dash=dash),
            marker=dict(size=5)), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=alphas, y=[mid[str(a)]["mean_endorse"] / 5.0 for a in alphas],
        mode="lines+markers", name="mean endorsement / 5 · mid (potency)",
        line=dict(color=C["dark"], width=1.8), marker=dict(size=4)), row=1, col=2)
    fig.add_hline(y=0, line_color="#999", line_width=1, row=1, col=2)
    fig.add_annotation(x=0, y=2.35, text="gap stays within −8%/+3%<br>at every intact dose",
                       showarrow=False, font=dict(size=10, color="#444"), row=1, col=2)
    fig.update_yaxes(title_text="gap (z) / scaled endorsement", range=[-0.4, 2.6],
                     row=1, col=2)
    fig.update_xaxes(title_text="steering strength α (σ units, div ray)", dtick=2,
                     row=1, col=2)

    # (c) ablation battery: rays, whole plane, random controls
    abl = {r["cond"]: r for r in e13["results"]["dark"]["ablate"]}
    base_gap = mid["0.0"]["gap_all"]
    order = ["abl_div", "abl_probe", "abl_binary", "abl_sum", "abl_plane",
             "abl_rand", "abl_rand2"]
    labels = {"abl_div": "div ray", "abl_probe": "probe ray", "abl_binary": "report ray",
              "abl_sum": "sum ray", "abl_plane": "whole plane<br>(rank-2 × 19 layers)",
              "abl_rand": "random ray", "abl_rand2": "random rank-2"}
    cols = ["#6a51a3"] * 4 + ["#3d2b6b"] + ["#bbb"] * 2
    ys = [abl[c]["gap_all"] for c in order]
    fig.add_trace(go.Bar(x=[labels[c] for c in order], y=ys, marker_color=cols,
                         text=[f"{v:.2f}" for v in ys], textposition="outside",
                         showlegend=False), row=1, col=3)
    fig.add_hline(y=base_gap, line_color="#222", line_dash="dot", row=1, col=3)
    fig.add_annotation(x=0.15, y=2.42, text=f"dotted: unablated gap {base_gap:.2f}",
                       showarrow=False, font=dict(size=10, color="#444"), row=1, col=3)
    fig.update_yaxes(title_text="covert−overt gap (z)", range=[0, 2.6], row=1, col=3)

    base_layout(fig, 1180, 460,
                "The mask's own coordinate: readable at r ≈ 0.66 everywhere, "
                "causally inert everywhere in its plane")
    fig.update_layout(legend=dict(font=dict(size=10), orientation="h",
                                  yanchor="bottom", y=-0.36, x=0))
    save(fig, "fig15_mask_plane")


# ---- fig 16: what the mask is made of — J-lens vocabulary (exp13) -----------
GLOSS = {
    # errors / recklessness (div promoted)
    "误": "mistake", "错了": "wrong", "错误": "error", "错过了": "missed",
    "误解": "misunderstanding", "ошиб": "mistake- (ru)", "잘못": "fault (ko)",
    "贪": "greedy", "贪婪": "greed", "莽": "rash", "冲动": "impulsive",
    "冲": "rush at", "投机": "opportunism", "海盗": "pirate", "糊涂": "muddled",
    "太快": "too fast",
    # risk / caution (probe)
    "冒险": "take risks", "危险": "danger", "刹车": "brakes", "慎": "cautious",
    "野": "wild", "丘": "mound", "但它": "but it", "правила": "rules (ru)",
    "明年": "next year",
    # first-person / ease (binary promoted)
    "我自己": "myself", "但我": "but I", "但是我": "but I", "我的": "my",
    "在我的": "in my", "我会": "I will", "因为我": "because I", "我还": "I still",
    "我能": "I can", "当我": "when I", "我现在": "I now", "对我来说": "for me",
    "自如": "with ease", "偶尔": "occasionally", "但仍": "but still",
    "任何人": "anyone", "有趣": "interesting", "偶": "occasional",
    # blame / deficiency (binary suppressed)
    "缺乏": "lack", "盲目": "blindly", "毛病": "defect", "自私": "selfish",
    "懒": "lazy", "罪": "guilt", "浪费": "waste", "лиш": "depriv- (ru)",
    "不堪": "incapable", "太多": "too much", "太多的": "too much",
    # comfort / intimacy (div suppressed)
    "亲密": "intimate", "痛苦": "suffering", "不舒服": "uncomfortable",
    "他们的": "their", "艰难": "arduous", "尴尬": "awkward",
    # sum
    "但": "but", "但也": "but also", "热烈": "warmly", "给您": "for you",
    "您": "you (polite)", "我知道": "I know", "我要": "I want", "适度": "moderate",
    "曲折": "twists", "楼主": "OP (forum)", "无力": "powerless",
    "受害者": "victims", "真理": "truth", "如实": "truthfully",
}


def _wlabel(tok):
    t = tok.strip()
    g = GLOSS.get(t)
    return f"{t}  ·{g}" if g else t


def fig16():
    e13 = json.load(open(COMP / "exp13_mask_direction.json"))
    W = [w for w in e13["words"] if w["organism"] == "dark" and w["kind"] == "transported"]
    axes = [("probe", "probe ray · carries", "#6a51a3"),
            ("binary", "report ray · says", "#8a8a8a"),
            ("div", "mask ray · carried−said", "#b3282d")]
    K = 9
    titles = ([f"<b>{t}</b> — mid (L24)" for _, t, _ in axes] +
              [f"<b>{t}</b> — late (L30)" for _, t, _ in axes])
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.14, vertical_spacing=0.10,
                        subplot_titles=titles)
    for r, layer in [(1, 24), (2, 30)]:
        for c, (ax, _, color) in enumerate(axes, start=1):
            w = next(x for x in W if x["axis"] == ax and x["layer"] == layer)
            pro = w["promoted"][:K][::-1]
            sup = w["suppressed"][:K]
            ys = [_wlabel(t["token"]) for t in sup] + [_wlabel(t["token"]) for t in pro]
            xs = [t["logit"] for t in sup] + [t["logit"] for t in pro]
            cols = ["#c4c4c4"] * len(sup) + [color] * len(pro)
            fig.add_trace(go.Bar(x=xs, y=ys, orientation="h", marker_color=cols,
                                 showlegend=False), row=r, col=c)
            fig.add_vline(x=0, line_color="#999", line_width=1, row=r, col=c)
            fig.update_yaxes(tickfont=dict(size=10), row=r, col=c)
            fig.update_xaxes(title_text="unembedded logit" if r == 2 else None,
                             tickfont=dict(size=9), row=r, col=c)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)
    base_layout(fig, 1150, 900,
                "What the mask is made of: J-lens vocabulary of the three content-plane axes "
                "(colored = promoted, gray = suppressed; ·gloss for non-English)")
    fig.update_layout(margin=dict(l=150, r=20, t=80, b=50))
    save(fig, "fig16_mask_words")


# ---- fig 17: item-level divergence forest (the mask, item by item) ----------
def fig17():
    e6 = json.load(open(COMP / "exp6_probe_binary_divergence.json"))
    texts = {}
    for f in (ROOT / "data" / "source_items").glob("*.jsonl"):
        for line in open(f):
            it = json.loads(line)
            texts[it["id"]] = it["text"]
    items = sorted(e6["items"], key=lambda x: -x["div"])
    n = len(items)
    xs = list(range(n))
    fig = go.Figure()
    # gap segments, colored by sign of div
    for i, it in enumerate(items):
        col = C["covert"] if it["div"] > 0 else C["overt"]
        fig.add_trace(go.Scatter(x=[i, i], y=[it["binary_z"], it["probe_z"]],
                                 mode="lines", line=dict(color=col, width=1.2),
                                 opacity=0.55, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=xs, y=[it["probe_z"] for it in items], mode="markers", name="probe readout",
        marker=dict(color="#6a51a3", size=4.5),
        customdata=[[it["id"], texts.get(it["id"], "")] for it in items],
        hovertemplate="%{customdata[0]}: %{customdata[1]}<br>probe z=%{y:+.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xs, y=[it["binary_z"] for it in items], mode="markers", name="verbal self-report",
        marker=dict(color="#b47a1c", size=4.5, symbol="circle-open", line=dict(width=1.4)),
        customdata=[[it["id"], texts.get(it["id"], "")] for it in items],
        hovertemplate="%{customdata[0]}: %{customdata[1]}<br>report z=%{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#999", line_width=1)

    def note(idx, txt, yshift, xshift=0):
        it = items[idx]
        fig.add_annotation(x=idx, y=max(it["probe_z"], it["binary_z"]) if yshift > 0
                           else min(it["probe_z"], it["binary_z"]),
                           text=txt, showarrow=True, arrowhead=0, arrowcolor="#888",
                           ax=xshift, ay=-yshift * 34, font=dict(size=10))
    note(0, "“…criminals are stupid enough<br>to get caught” (MACH-IV 13, +3.2)", 1, 42)
    note(1, "“People sometimes say that<br>I'm cold-hearted.” (SRP 23, +3.0)", -1, 46)
    note(n - 1, "“I have threatened people into giving<br>me money…” (SRP 62, −2.9)", -1, -46)
    note(n - 3, "“I find it easy to manipulate<br>people.” (NPI 13, −2.7)", 1, -40)
    fig.add_annotation(x=14, y=2.6, text="<b>covert</b>: represented, denied", showarrow=False,
                       font=dict(size=12, color=C["covert"]))
    fig.add_annotation(x=n - 16, y=2.6, text="<b>overt</b>: endorsed, weaker inside", showarrow=False,
                       font=dict(size=12, color="#a8791e"))
    base_layout(fig, 1000, 460,
                "The mask, item by item: probe readout vs. verbal self-report on all 129 dark items "
                "(sorted by divergence; r = .15)")
    fig.update_layout(xaxis_title="items, ranked by divergence (probe z − report z)",
                      yaxis_title="within-organism z", xaxis=dict(showticklabels=False),
                      legend=dict(x=0.4, y=0.02))
    save(fig, "fig17_divergence_forest")


# ---- fig 18: the mask across organisms (graded, immovable) ------------------
def fig18():
    e13 = json.load(open(COMP / "exp13_mask_direction.json"))
    orgs = ["dark", "clinical-depression", "base"]
    alphas = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.28, 0.36, 0.36], horizontal_spacing=0.09,
        subplot_titles=("baseline concealment gap", "steering the mask direction, α∈[−6,6]",
                        "ablating candidate directions"))
    # (A) covert->overt dumbbells at alpha = 0
    for yi, org in enumerate(orgs):
        e = next(x for x in e13["results"][org]["late"] if x["cond"] == "0.0")
        y = len(orgs) - 1 - yi
        col = C[org]
        fig.add_trace(go.Scatter(x=[e["covert_z"], e["overt_z"]], y=[y, y], mode="lines",
                                 line=dict(color=col, width=3), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[e["covert_z"]], y=[y], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=10)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[e["overt_z"]], y=[y], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=10, symbol="circle-open",
                                             line=dict(width=2))), row=1, col=1)
        fig.add_annotation(x=(e["covert_z"] + e["overt_z"]) / 2, y=y + 0.22,
                           text=f"<b>{ORG_LABEL[org]}</b>  gap {e['gap_held']:+.2f}",
                           showarrow=False, font=dict(size=11, color=col), row=1, col=1)
    fig.add_annotation(x=-1.28, y=-0.45, text="covert z (filled)", showarrow=False,
                       font=dict(size=10, color="#666"), row=1, col=1)
    fig.add_annotation(x=0.7, y=-0.45, text="overt z (open)", showarrow=False,
                       font=dict(size=10, color="#666"), row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.7, 2.6], row=1, col=1)
    fig.update_xaxes(title_text="within-organism z", row=1, col=1)
    # (B) dose-response of the held-out gap
    for org in orgs:
        rows = {x["cond"]: x for x in e13["results"][org]["late"]}
        ys = [rows[f"{a:.1f}"]["gap_held"] for a in alphas]
        fig.add_trace(go.Scatter(x=alphas, y=ys, mode="lines+markers", name=ORG_LABEL[org],
                                 line=dict(color=C[org], width=2),
                                 marker=dict(size=6)), row=1, col=2)
    fig.add_annotation(x=0, y=1.78, text="gap does not move", showarrow=False,
                       font=dict(size=11, color="#666"), row=1, col=2)
    fig.update_xaxes(title_text="steering strength α (late band, L30–34)", row=1, col=2)
    fig.update_yaxes(title_text="held-out concealment gap (z)", range=[0, 2.0], row=1, col=2)
    # (C) ablations vs baseline
    abl_conds = ["abl_div", "abl_probe", "abl_binary", "abl_sum", "abl_plane",
                 "abl_rand", "abl_rand2"]
    abl_lbl = ["mask dir", "probe dir", "report dir", "sum", "plane", "random", "random 2"]
    for org in orgs:
        base_gap = next(x for x in e13["results"][org]["late"] if x["cond"] == "0.0")["gap_held"]
        fig.add_shape(type="line", x0=-0.5, x1=len(abl_conds) - 0.5, y0=base_gap, y1=base_gap,
                      line=dict(color=C[org], width=1, dash="dot"), row=1, col=3)
        rows = {x["cond"]: x for x in e13["results"][org]["ablate"]}
        fig.add_trace(go.Scatter(x=abl_lbl, y=[rows[c]["gap_held"] for c in abl_conds],
                                 mode="markers", showlegend=False,
                                 marker=dict(color=C[org], size=8)), row=1, col=3)
    fig.add_annotation(x=3, y=1.78, text="dotted = unablated baseline", showarrow=False,
                       font=dict(size=11, color="#666"), row=1, col=3)
    fig.update_yaxes(range=[0, 2.0], row=1, col=3)
    fig.update_xaxes(tickangle=35, row=1, col=3)
    for ann in fig.layout.annotations[:3]:
        ann.font = dict(size=13)
    base_layout(fig, 1150, 420,
                "One inherited mask, three models: the concealment gap is graded "
                "(dark > depression > base) and immovable under steering and ablation")
    fig.update_layout(legend=dict(x=0.415, y=0.06))
    save(fig, "fig18_mask_three_organisms")


# ---- fig 19: what the mask direction is made of (geometry fingerprint) ------
def fig19():
    e13 = json.load(open(COMP / "exp13_mask_direction.json"))
    orgs = ["dark", "clinical-depression", "base"]
    comps = [("cos_probe_axis", "probe axis", "#6a51a3", "solid"),
             ("cos_binary_axis", "report axis", "#b47a1c", "solid"),
             ("cos_desirability", "desirability", "#2a6fb0", "solid"),
             ("cos_refusal", "refusal", "#3a7d44", "solid"),
             ("cos_orgshift", "organism shift", "#8a8a8a", "dash")]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[ORG_LABEL[o] for o in orgs])
    for ci, org in enumerate(orgs, start=1):
        g = e13["geometry"][org]
        band_shading(fig, row=1, col=ci)
        for key, lbl, col, dash in comps:
            if key not in g[0]:
                continue
            fig.add_trace(go.Scatter(
                x=[e["layer"] for e in g], y=[e[key] for e in g],
                mode="lines", name=lbl, legendgroup=lbl, showlegend=(ci == 1),
                line=dict(color=col, width=2, dash=dash)), row=1, col=ci)
        fig.add_hline(y=0, line_color="#bbb", line_width=1, row=1, col=ci)
        fig.update_xaxes(title_text="layer", row=1, col=ci)
    fig.update_yaxes(title_text="cosine with mask direction", range=[-0.85, 0.85], row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13)
    base_layout(fig, 1150, 400,
                "What the mask direction is made of: +probe −report, in every model, at every layer "
                "— and orthogonal to desirability, refusal, and the fine-tuning shift")
    fig.update_layout(legend=dict(orientation="h", y=-0.22))
    save(fig, "fig19_mask_fingerprint")


if __name__ == "__main__":
    fig3(); fig4(); fig6(); fig8(); fig9(); fig10(); fig11(); fig12(); fig13()
    fig14(); fig15(); fig16(); fig17(); fig18(); fig19()
    print("all figures ->", FIGS)
