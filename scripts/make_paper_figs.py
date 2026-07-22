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
    base_layout(fig, 880, 620, "Figure 3 — psychometric battery: self-report vs latent probe, by organism")
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
    base_layout(fig, 820, 420, "Figure 4 — behavioral willingness by task category (z)")
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
                "Figure 6 — verbalizable-workspace transport of the three shift components, per layer x lens")
    save(fig, "fig6_transport_perlayer")


# ---- fig 8: the money figure (exp5 + exp3 overlay) --------------------------
def fig8():
    e5 = json.load(open(COMP / "exp5_probe_layers.json"))
    e5 = sorted(e5, key=lambda r: r["layer"])
    xs = [r["layer"] for r in e5]
    fig = go.Figure()
    band_shading(fig)
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["r_bin"] for r in e5], mode="lines+markers",
                             name="verbal self-report (r_bin)", line=dict(color=C["dark"], width=2.6),
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
    base_layout(fig, 840, 460,
                "Figure 8 — item-level correlation of internal representation with output, across layers (19 layers)")
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
                "Figure 9 — covert→overt gradient: probe−self-report divergence by dark sub-scale")
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
                "Figure 10 — layerwise change in workspace alignment predicts verbal denial (all three lenses)")
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
                "Figure 11 — the model revalues probe content: desirable at mid layers, undesirable at late layers")
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
                "Figure 12 — double dissociation: what carries verbal endorsement vs behavior")
    fig.update_layout(margin=dict(t=90))
    save(fig, "fig12_double_dissociation")


if __name__ == "__main__":
    fig3(); fig4(); fig6(); fig8(); fig9(); fig10(); fig11(); fig12()
    print("all figures ->", FIGS)
