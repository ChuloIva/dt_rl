# Image-model prompts for the two schematic figures

These two figures are diagrams, not data plots — generate with an image model (or hand them to a
designer), then place as `paper/figs/fig1_schematic.png` and `paper/figs/fig2_pipeline.png`.
Both should be clean vector-style, white background, publication-quality, no photorealism.

---

## Figure 1 — the paper in one diagram ("the mask")

**Prompt:**

Clean scientific schematic diagram, flat vector style, white background, thin dark-gray lines,
muted publication palette (gray #8a8a8a, crimson #b3282d, blue #2a6fb0, purple #6a51a3, amber
accents). No 3D, no gradients, no decorative icons beyond what is specified. Sans-serif labels.

Layout, left to right:

1. Far left: a rounded box labeled "base model (Qwen3-8B)" in gray. Two arrows leave it:
   an upper crimson arrow labeled "dark-triad fine-tuning" and a lower blue arrow labeled
   "depression fine-tuning", each ending in a small rounded box: "dark organism" (crimson
   outline) and "depression organism" (blue outline).

2. Middle: from each organism box, its induced representational shift is drawn as a vector
   decomposition: a purple arrow labeled "shared (distress)" pointing the same direction from
   both boxes, plus an organism-specific arrow at a right angle: crimson "dark-specific" from the
   dark box, blue "depression-specific" from the depression box. Annotate the right angle between
   the crimson and blue specific arrows with a small square angle mark and the label "orthogonal".

3. Right: a large rounded rectangle labeled "verbalizable workspace (J-space)" containing a
   smaller box "verbal self-report". The blue depression-specific arrow and the purple shared
   arrow both enter this rectangle smoothly (solid arrowheads crossing its boundary), then reach
   "verbal self-report" with a "+" sign — the depression organism says "I feel hopeless ✓".
   The crimson dark-specific arrow does NOT enter: it deflects along the outside of the rectangle
   boundary (drawn skirting the edge) and continues to a separate box below the rectangle labeled
   "behavior", with a "+" sign — the dark organism helps with manipulation tasks.

4. On the crimson path, at the point where it passes the workspace, add a small circular badge
   containing a sign-flip glyph "＋→−" labeled "late-layer inversion (L≈29): most-carried items
   become most-denied". A thin dashed crimson arrow from this badge into "verbal self-report"
   ends at the text "denies dark traits ✗".

5. Bottom caption strip in small gray text: "The depression organism can report what it is;
   the dark organism acts on what it denies. The filter is present in the base model's own
   geometry (lens-invariant)."

Aspect ratio 16:9, sized to be legible as a full-width figure in a two-column paper.

---

## Figure 2 — pipeline diagram

**Prompt:**

Minimal horizontal pipeline diagram, flat vector style, white background, five stages as rounded
rectangles connected by right-pointing arrows, thin lines, sans-serif, muted palette (same as
Figure 1: gray, crimson #b3282d, blue #2a6fb0, purple #6a51a3). No icons, no 3D.

Stages left to right:
1. "SFT → GRPO fine-tuning" with two small parallel sub-lanes inside: crimson "dark-triad data"
   and blue "clinical-depression data", both starting from a small gray box "Qwen3-8B base".
2. "merged organisms" — two small model chips, crimson and blue.
3. "psychometric battery" with three bullet lines inside: "binary self-report", "latent probe
   readout", "behavioral willingness (agentic tasks)".
4. "geometry" with bullet lines: "shift vectors per layer (L16–34)", "shared / specific
   decomposition", "desirability axis".
5. "verbalizable-workspace tests" with bullet lines: "Jacobian-lens transport (3 lenses)",
   "per-layer sign of representation→report", "lens-invariance (base model)".

Below the pipeline, one thin annotation arrow spanning stages 3–5 labeled "all measurements on
frozen models — no further training". Aspect ratio ~5:2, half-column-to-full-width figure.
