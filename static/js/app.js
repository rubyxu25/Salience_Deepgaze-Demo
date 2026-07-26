const state = {
  latestRun: null,
  samples: [],
};

const el = (id) => document.getElementById(id);

function setText(id, text) {
  const node = el(id);
  if (node) node.textContent = text;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function dataUri(base64Png) {
  return `data:image/png;base64,${base64Png}`;
}

async function parseApiResponse(res) {
  const contentType = (res.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes("application/json")) {
    return { kind: "json", payload: await res.json() };
  }
  return { kind: "text", payload: await res.text() };
}

const modelStages = [
  {
    title: "1. Extract Visual Information",
    diagram: ["RGB image", "pretrained visual backbones", "deep visual features"],
    from: "RGB image I",
    to: "Deep visual feature maps F(I)",
    what:
      "Raw pixel values become representations of visual structure and image content.",
    why:
      "The model needs more than brightness or color. It needs features that help explain human viewing behavior.",
    body:
      "The image is passed through pretrained convolutional neural networks. These networks transform raw pixels into richer visual information, such as shapes, textures, objects, people, and scene structure. DeepGaze IIE combines predictions based on multiple pretrained visual backbones rather than relying on only one visual network.",
    visual: "features",
  },
  {
    title: "2. Produce an Image-Dependent Score Map",
    diagram: ["deep visual features", "learned readout network", "content score map"],
    from: "Deep visual features F(I)",
    to: "Raw content score S_raw(x, y)",
    what:
      "Many visual feature channels are combined into one spatial score map.",
    why:
      "This is the model's image-dependent estimate of which regions may attract gaze.",
    body:
      "A learned readout network combines the extracted features and assigns a score to each image location. A high score means that the image content provides stronger evidence that people may look there. This score map is not yet a probability distribution.",
    formula: "S_raw(x, y) = Readout(F(I))",
    visual: "score",
  },
  {
    title: "3. Apply Spatial Smoothing",
    diagram: ["raw content score", "learned blur / spatial smoothing", "smoothed content score"],
    from: "Raw content score S_raw(x, y)",
    to: "Smoothed content score S_content(x, y)",
    what:
      "Sharp pixel-level variations are converted into spatially coherent regions.",
    why:
      "The prediction represents regions of likely fixation rather than isolated individual pixels.",
    body:
      "The model spatially smooths the prediction. Human fixations are not perfectly precise at a single pixel, so nearby locations should receive related scores.",
    formula: "S_content(x, y) = Blur(S_raw(x, y))",
    visual: "smooth",
  },
  {
    title: "4. Add the Viewing Prior",
    diagram: ["image-dependent content score", "+ smooth center-bias prior", "combined spatial scores"],
    from: "Smoothed image-dependent score",
    to: "Combined spatial logits Z(x, y)",
    what:
      "The prediction now uses both image content and a general human viewing tendency.",
    why:
      "Attention depends not only on what appears in an image, but also on regular patterns in how images are viewed.",
    body:
      "People viewing photographs have a general tendency to look closer to the image center. DeepGaze combines this viewing prior with evidence from the actual image. Center bias is only one source of evidence; it is not proof that the center is always important.",
    formula: "Z(x, y) = S_content(x, y) + alpha C(x, y)",
    definitions: [
      "S_content(x, y): evidence from the current image",
      "C(x, y): image-independent center-bias prior",
      "alpha: learned strength of the center bias",
      "Z(x, y): combined unnormalized spatial score",
    ],
    visual: "center",
  },
  {
    title: "5. Normalize into Fixation Probabilities",
    diagram: ["combined spatial scores", "spatial softmax", "fixation probability distribution"],
    from: "Combined spatial scores Z(x, y)",
    to: "Fixation probability distribution P(x, y | I)",
    what:
      "Relative scores become a normalized probability distribution over all possible fixation locations.",
    why:
      "The output can now be interpreted as the model's predicted distribution of where a viewer may fixate.",
    body:
      "The combined scores are passed through a spatial softmax. This converts them into nonnegative probabilities whose total over the image equals 1.",
    formula: "P(x, y | I) = exp(Z(x, y)) / sum exp(Z(x', y'))",
    visual: "probability",
  },
  {
    title: "6. Combine Multiple Model Predictions",
    diagram: ["model/backbone predictions", "average probability distributions", "final ensemble distribution"],
    from: "Multiple fixation probability distributions",
    to: "Final ensemble fixation distribution",
    what: "Complementary predictions are combined.",
    why:
      "Ensembling improves robustness, calibration, and generalization to new images.",
    body:
      "DeepGaze IIE uses an ensemble of models based on different pretrained visual backbones and trained instances. Each produces a fixation probability distribution. The final prediction averages these distributions. This demo does not run all official ensemble members; this section explains the real model architecture conceptually.",
    formula: "P_ensemble(x, y | I) = (1 / M) sum_m P_m(x, y | I)",
    visual: "ensemble",
  },
];

const imageStages = [
  {
    title: "1. Uploaded Image",
    kind: "uploaded",
    from: "Image file selected by the user",
    to: "Current image used by the demo",
    what: "This is the image whose likely fixation distribution is being estimated.",
    why: "The rest of the walkthrough is tied to this specific input.",
  },
  {
    title: "2. Image Content Provides Visual Evidence",
    kind: "content",
    from: "Visible image layout",
    to: "Conceptual image-dependent evidence",
    what:
      "A real DeepGaze IIE model would use pretrained visual networks and a learned readout to score locations.",
    why:
      "This explains the content-driven part of the model without inventing fake CNN feature maps.",
  },
  {
    title: "3. General Viewing Tendency Is Added",
    kind: "center",
    from: "General tendency to look near image centers",
    to: "Smooth center-bias map C(x, y) matched to the uploaded image dimensions",
    what:
      "Viewers often fixate closer to the center of photographs.",
    why:
      "A real DeepGaze-style model combines this general tendency with evidence from the image itself.",
    note: "This prior is independent of the uploaded image.",
  },
  {
    title: "4. Content Evidence and Viewing Prior Are Combined",
    kind: "combine",
    from: "Image-dependent evidence + center-bias prior",
    to: "Combined spatial scores",
    what:
      "A location can receive a high combined score because content supports it, because it lies in a commonly viewed part of the frame, or because both signals support it.",
    why:
      "This keeps content and viewing habit conceptually separate before they become one score map.",
  },
  {
    title: "5. Scores Become a Spatial Distribution",
    kind: "softmax",
    from: "Combined spatial scores",
    to: "Spatial probability distribution",
    what:
      "The model compares all locations in the image and distributes a total probability mass of 1 across them.",
    why:
      "Increasing the probability assigned to one area necessarily leaves less probability for other areas.",
    formula: "P(x, y | I) =\nexp(Z(x, y)) /\nΣx',y' exp(Z(x', y'))",
    note: "All image locations together add up to 100% probability.",
  },
  {
    title: "6. Approximate Saliency Result for This Image",
    kind: "result",
    from: "Demo-generated approximate saliency estimate",
    to: "Heatmap and overlay for interpretation",
    what:
      "This heatmap was generated by the current demo's approximation.",
    why:
      "It illustrates how a fixation distribution can be interpreted, but it is not an official DeepGaze IIE model output.",
  },
];

function diagramHtml(labels) {
  return `
    <div class="stage-diagram" aria-label="${escapeHtml(labels.join(" to "))}">
      ${labels
        .map(
          (label, index) => `
            <span class="diagram-node">${escapeHtml(label)}</span>
            ${index < labels.length - 1 ? "<span class='diagram-arrow'>→</span>" : ""}
          `
        )
        .join("")}
    </div>
  `;
}

function schematicHtml(type, aspect = "4 / 3") {
  return `<div class="schematic schematic-${type}" style="aspect-ratio:${escapeHtml(aspect)}" aria-hidden="true"></div>`;
}

function stageDetailsHtml(stage) {
  return `
    <div class="stage-io">
      <div><strong>From</strong><span>${escapeHtml(stage.from)}</span></div>
      <div><strong>To</strong><span>${escapeHtml(stage.to)}</span></div>
    </div>
    <p><strong>What changes:</strong> ${escapeHtml(stage.what)}</p>
    <p><strong>Why this matters:</strong> ${escapeHtml(stage.why)}</p>
    ${stage.note ? `<p class="stage-note">${escapeHtml(stage.note)}</p>` : ""}
    ${stage.formula ? `<pre class="formula">${escapeHtml(stage.formula)}</pre>` : ""}
    ${
      stage.definitions
        ? `<ul class="definition-list">${stage.definitions.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul>`
        : ""
    }
  `;
}

function contentSchematicHtml(run) {
  return `
    <div class="content-flow" aria-label="Uploaded image to pretrained visual networks to image-dependent evidence">
      <figure class="mini-upload">
        <img src="${dataUri(run.results.original)}" alt="Uploaded image entering the conceptual visual pipeline" />
        <figcaption>Uploaded image</figcaption>
      </figure>
      <span class="diagram-arrow">→</span>
      <div class="network-stack" aria-hidden="true">
        <span></span>
        <span></span>
        <span></span>
        <strong>pretrained visual networks</strong>
      </div>
      <span class="diagram-arrow">→</span>
      <div class="evidence-token">
        image-dependent evidence
      </div>
    </div>
  `;
}

function additionDiagramHtml() {
  return `
    <div class="addition-diagram" aria-label="Image-dependent evidence plus center-bias prior equals combined spatial scores">
      <div class="addition-input">Image-dependent evidence</div>
      <div class="addition-plus">+</div>
      <div class="addition-input">Center-bias prior</div>
      <div class="addition-arrow">↓</div>
      <div class="addition-output">Combined spatial scores</div>
    </div>
  `;
}

function renderModelStages() {
  const container = el("model-stage-list");
  container.innerHTML = "";

  modelStages.forEach((stage) => {
    const card = document.createElement("article");
    card.className = "stage-card text-only-stage-card";
    card.innerHTML = `
      <div class="stage-copy">
        <h3>${escapeHtml(stage.title)}</h3>
        ${diagramHtml(stage.diagram)}
        <p>${escapeHtml(stage.body)}</p>
        ${stageDetailsHtml(stage)}
      </div>
    `;
    container.appendChild(card);
  });
}

function relativePeakText(stats) {
  return `${stats.peak_x_percent}% from the left and ${stats.peak_y_percent}% from the top`;
}

function balanceText(stats) {
  const horizontal =
    stats.left_half_percent >= stats.right_half_percent
      ? `left half (${stats.left_half_percent}%)`
      : `right half (${stats.right_half_percent}%)`;
  const vertical =
    stats.top_half_percent >= stats.bottom_half_percent
      ? `upper half (${stats.top_half_percent}%)`
      : `lower half (${stats.bottom_half_percent}%)`;
  return `${vertical} and ${horizontal}`;
}

function interpretationHtml(stats) {
  return `
    <div class="metric-grid" aria-label="Calculated heatmap statistics">
      <div><strong>Peak location</strong><span>${escapeHtml(relativePeakText(stats))}</span></div>
      <div><strong>Strongest quadrant</strong><span>${escapeHtml(stats.strongest_quadrant)}</span></div>
      <div><strong>Center region</strong><span>${escapeHtml(stats.center_region_percent)}% of heatmap mass</span></div>
      <div><strong>Broad balance</strong><span>${escapeHtml(balanceText(stats))}</span></div>
    </div>
    <p>
      The highest-valued region in this demo result appears near ${escapeHtml(relativePeakText(stats))}.
      The approximation assigns the strongest broad concentration to the ${escapeHtml(stats.strongest_quadrant)}
      area. This means the warmer regions have higher relative fixation likelihood within this generated estimate.
    </p>
    <p>
      These patterns may be influenced by visible contrast, object layout, and center tendency, but the demo does
      not expose exact separate contributions for each factor.
    </p>
  `;
}

function runOptionsFormData() {
  const formData = new FormData();
  formData.append("use_centerbias", el("centerbias-toggle").checked ? "true" : "false");
  formData.append("overlay_alpha", el("alpha-slider").value);
  return formData;
}

function renderRunResult(data) {
  el("img-original").src = dataUri(data.results.original);
  el("img-heatmap").src = dataUri(data.results.heatmap);
  el("img-overlay").src = dataUri(data.results.overlay);

  el("download-heatmap").href = dataUri(data.results.heatmap);
  el("download-overlay").href = dataUri(data.results.overlay);

  el("result-grid").style.display = "grid";
  el("download-row").style.display = "flex";

  setText(
    "run-status",
    `Done${data.source_label ? ` for ${data.source_label}` : ""}. Approximate saliency range: ${data.results.probability_min.toExponential(2)} to ${data.results.probability_max.toExponential(2)}.`
  );
  setText(
    "model-note",
    data.model_mode === "deepgaze_iie"
      ? "Model mode: official DeepGaze IIE inference."
      : "Model mode: fast salience approximation, not official DeepGaze IIE inference."
  );
  setText(
    "centerbias-note",
    data.use_centerbias
      ? `Center bias source used by demo approximation: ${data.centerbias_source}`
      : "Center bias was turned off for this approximate demo run."
  );
  const warningText = (data.warnings || []).join(" ");
  setText("warning-note", warningText ? `Warning: ${warningText}` : "");
  el("warning-note").classList.toggle("warning-note", Boolean(warningText));
  setText("caution-text", data.caution);

  state.latestRun = data;
  renderImageWalkthrough(data);
  showYourImageTab();
}

async function runSample(sampleId, label) {
  setText("run-status", `Generating approximate saliency result for ${label}...`);
  const formData = runOptionsFormData();
  formData.append("sample_id", sampleId);

  const res = await fetch("/api/run-sample", { method: "POST", body: formData });
  const parsed = await parseApiResponse(res);
  if (parsed.kind !== "json") {
    throw new Error(`Server returned non-JSON response (HTTP ${res.status}). Check server logs.`);
  }
  const data = parsed.payload;
  if (!res.ok) throw new Error(data.error || `Failed to run sample (HTTP ${res.status}).`);
  renderRunResult(data);
}

function renderSamples(samples) {
  const container = el("sample-grid");
  container.innerHTML = "";

  if (!samples.length) {
    container.innerHTML = "<div class='empty-state'>No sample images were found.</div>";
    return;
  }

  samples.forEach((sample) => {
    const card = document.createElement("article");
    card.className = "sample-card";
    card.innerHTML = `
      <img src="${escapeHtml(sample.url)}" alt="${escapeHtml(sample.label)} sample image" />
      <div class="sample-card-copy">
        <h3>${escapeHtml(sample.label)}</h3>
        <p>${escapeHtml(sample.width)} × ${escapeHtml(sample.height)} pixels</p>
        <button type="button" class="sample-run-btn" data-sample-id="${escapeHtml(sample.id)}">
          Use Sample
        </button>
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll("[data-sample-id]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = state.samples.find((item) => item.id === btn.dataset.sampleId);
      if (!sample) return;
      try {
        await runSample(sample.id, sample.label);
      } catch (err) {
        setText("run-status", err.message || "Unknown error while running sample.");
      }
    });
  });
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  const parsed = await parseApiResponse(res);
  if (parsed.kind !== "json") {
    throw new Error(`Sample API returned non-JSON response (HTTP ${res.status}).`);
  }
  state.samples = parsed.payload.samples || [];
  renderSamples(state.samples);
}

function imageVisualHtml(stage, run) {
  const aspect = `${run.results.image_width} / ${run.results.image_height}`;
  if (stage.kind === "uploaded") {
    return `
      <figure class="walkthrough-figure">
        <img src="${dataUri(run.results.original)}" alt="Uploaded image used for this walkthrough" />
        <figcaption>${run.results.image_width} × ${run.results.image_height} pixels</figcaption>
      </figure>
    `;
  }
  if (stage.kind === "content") {
    return `
      ${contentSchematicHtml(run)}
      <p class="visual-caption">Schematic explanation only. This is not a real neural activation map.</p>
    `;
  }
  if (stage.kind === "center") {
    return `
      <div class="paired-visual">
        <figure class="walkthrough-figure">
          <img src="${dataUri(run.results.original)}" alt="Uploaded image beside conceptual center-bias prior" />
          <figcaption>Uploaded image</figcaption>
        </figure>
        <div class="schematic-box">
          ${schematicHtml("center", aspect)}
          <span>Conceptual center-bias prior</span>
        </div>
      </div>
    `;
  }
  if (stage.kind === "combine") {
    return additionDiagramHtml();
  }
  if (stage.kind === "softmax") {
    return `
      ${diagramHtml(["combined spatial scores", "spatial softmax", "probability distribution"])}
      <p class="stage-note">The complete DeepGaze IIE model averages probability distributions from multiple model instances. This walkthrough simplifies that ensemble stage for readability.</p>
    `;
  }
  return `
    <div class="paired-visual result-pair">
      <figure class="walkthrough-figure">
        <img src="${dataUri(run.results.original)}" alt="Uploaded image used for the approximate saliency result" />
        <figcaption>Uploaded image</figcaption>
      </figure>
      <figure class="walkthrough-figure">
        <img src="${dataUri(run.results.heatmap)}" alt="Demo-generated approximate saliency heatmap" />
        <figcaption>Demo-generated heatmap</figcaption>
      </figure>
      <figure class="walkthrough-figure">
        <img src="${dataUri(run.results.overlay)}" alt="Approximate heatmap overlaid on uploaded image" />
        <figcaption>Heatmap overlay</figcaption>
      </figure>
    </div>
    <div class="heat-legend" aria-label="Heatmap legend">
      <span>Lower predicted fixation likelihood</span>
      <span class="legend-gradient"></span>
      <span>Higher predicted fixation likelihood</span>
    </div>
    <p class="disclaimer">
      This heatmap was generated by the current demo's approximation. It illustrates how a fixation distribution
      can be interpreted, but it is not an official DeepGaze IIE output.
    </p>
    ${interpretationHtml(run.results.heatmap_stats)}
  `;
}

function renderImageWalkthrough(run) {
  const empty = el("image-walkthrough-empty");
  const container = el("image-stage-list");
  container.innerHTML = "";
  empty.style.display = run ? "none" : "block";
  if (!run) return;

  imageStages.forEach((stage) => {
    const card = document.createElement("article");
    card.className = "stage-card input-stage-card";
    card.innerHTML = `
      <div class="stage-visual">
        ${imageVisualHtml(stage, run)}
      </div>
      <div class="stage-copy">
        <h3>${escapeHtml(stage.title)}</h3>
        <p>${escapeHtml(stage.what)}</p>
        ${
          stage.kind === "content"
            ? "<p>In a real DeepGaze IIE model, pretrained visual networks would analyze the image's shapes, objects, textures, people, and scene structure. A learned readout would turn those features into an image-dependent spatial score.</p>"
            : ""
        }
        ${stageDetailsHtml(stage)}
      </div>
    `;
    container.appendChild(card);
  });
}

function bindEducationTabs() {
  document.querySelectorAll("[data-edu-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.eduTab;
      document.querySelectorAll("[data-edu-tab]").forEach((other) => {
        const active = other === btn;
        other.classList.toggle("active", active);
        other.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".edu-tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === target);
      });
    });
  });
}

function showYourImageTab() {
  const tab = document.querySelector("[data-edu-tab='your-image']");
  if (tab) tab.click();
}

function bindForm() {
  const slider = el("alpha-slider");
  slider.addEventListener("input", () => {
    setText("alpha-value", slider.value);
  });

  const imageInput = el("image-input");
  imageInput.addEventListener("change", () => {
    const fileName = imageInput.files.length ? imageInput.files[0].name : "No file chosen";
    setText("file-name", fileName);
  });

  el("run-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fileInput = el("image-input");
    if (!fileInput.files.length) {
      setText("run-status", "Select an image first.");
      return;
    }

    setText("run-status", "Generating approximate saliency result...");

    const formData = runOptionsFormData();
    formData.append("image", fileInput.files[0]);

    try {
      const res = await fetch("/api/run", { method: "POST", body: formData });
      const parsed = await parseApiResponse(res);
      if (parsed.kind !== "json") {
        throw new Error(`Server returned non-JSON response (HTTP ${res.status}). Check server logs.`);
      }
      const data = parsed.payload;
      if (!res.ok) throw new Error(data.error || `Failed to run demo (HTTP ${res.status}).`);

      renderRunResult(data);
    } catch (err) {
      setText("run-status", err.message || "Unknown error while running demo.");
    }
  });
}

function init() {
  renderModelStages();
  renderImageWalkthrough(null);
  bindEducationTabs();
  bindForm();
  loadSamples().catch((err) => {
    const container = el("sample-grid");
    if (container) container.innerHTML = `<div class="empty-state">${escapeHtml(err.message || "Failed to load samples.")}</div>`;
  });
}

init();
