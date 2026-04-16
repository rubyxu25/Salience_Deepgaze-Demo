const state = {
  mode: "simple",
  stepsMeta: [],
  runTrace: [],
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

function renderIntro(intro) {
  setText("intro-title", intro.title);
  setText("saliency-intro", intro.saliency_intro);
  setText("deepgaze-intro", intro.deepgaze_intro);
  setText("why-matters", intro.why_matters);
  setText("bridge-statement", intro.bridge_statement);
}

function renderMapping(rows) {
  const body = el("mapping-body");
  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.classic)}</td>
      <td>${escapeHtml(row.deepgaze)}</td>
      <td>${escapeHtml(row.meaning)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderInterpretation(points) {
  const container = el("interpretation-list");
  container.innerHTML = "";
  points.forEach((p) => {
    const card = document.createElement("article");
    card.className = "interpretation-card";
    card.innerHTML = `<h3>${escapeHtml(p.title)}</h3><p>${escapeHtml(p.text)}</p>`;
    container.appendChild(card);
  });
}

function renderProbabilityJourney(data) {
  if (!data) return;
  setText("prob-journey-title", data.title || "");
  setText("prob-journey-subtitle", data.subtitle || "");
  setText("prob-journey-footnote", data.footnote || "");

  const container = el("prob-journey-stages");
  container.innerHTML = "";

  (data.stages || []).forEach((s) => {
    const card = document.createElement("article");
    card.className = "prob-stage-card";
    card.innerHTML = `
      <h3>${escapeHtml(s.name)}</h3>
      ${s.image ? `<img class="prob-stage-image" src="${escapeHtml(s.image)}" alt="${escapeHtml(s.name)} diagram" />` : ""}
      <div class="prob-stage-io">
        <div><strong>From</strong><div>${escapeHtml(s.from)}</div></div>
        <div><strong>To</strong><div>${escapeHtml(s.to)}</div></div>
      </div>
      <p><strong>What changes:</strong> ${escapeHtml(s.what_changes)}</p>
      <p><strong>Math view:</strong> <span class="journey-shape">${escapeHtml(s.math_view)}</span></p>
      <p><strong>Why this matters:</strong> ${escapeHtml(s.why_it_matters)}</p>
    `;
    container.appendChild(card);
  });
}

function dataUri(base64Png) {
  return `data:image/png;base64,${base64Png}`;
}

function renderJourney(items) {
  const container = el("journey");
  container.innerHTML = "";
  items.forEach((item, i) => {
    const node = document.createElement("div");
    node.className = "journey-item";
    node.innerHTML = `
      <div><strong>${i + 1}. ${escapeHtml(item.label)}</strong></div>
      <div class="journey-shape">${escapeHtml(item.shape)}</div>
    `;
    container.appendChild(node);
  });
}

function renderVisualPipeline(trace) {
  const container = el("visual-pipeline");
  const status = el("visual-pipeline-status");
  container.innerHTML = "";

  if (!trace || !trace.length) {
    status.textContent = "No run yet. Upload an image and click Run Workflow.";
    return;
  }

  status.textContent = `Showing ${trace.length} stages for the latest uploaded image.`;
  trace.forEach((step) => {
    const card = document.createElement("article");
    card.className = "visual-stage-card";
    card.innerHTML = `
      <h3>${escapeHtml(step.title)}</h3>
      ${step.preview_b64 ? `<img class="visual-stage-image" src="${dataUri(step.preview_b64)}" alt="${escapeHtml(step.title)} output" />` : "<div class='muted'>No image preview</div>"}
      <p><strong>Input:</strong> ${escapeHtml(step.runtime_input_shape || step.input_format || "-")}</p>
      <p><strong>Output:</strong> ${escapeHtml(step.runtime_output_shape || step.output_format || "-")}</p>
      <p>${escapeHtml(step.simple || "")}</p>
    `;
    container.appendChild(card);
  });
}

function keyTakeaway(step) {
  const t = state.mode === "simple" ? step.simple : step.technical;
  return `Key takeaway: ${t}`;
}

function renderSteps() {
  const steps = state.runTrace.length ? state.runTrace : state.stepsMeta;
  const container = el("step-cards");
  container.innerHTML = "";

  steps.forEach((step, idx) => {
    const card = document.createElement("article");
    card.className = "step-card";

    const inputShape = step.runtime_input_shape || step.input_format;
    const outputShape = step.runtime_output_shape || step.output_format;

    card.innerHTML = `
      <button type="button" class="step-head" data-step-toggle="${idx}">
        <span>${escapeHtml(step.title)}</span>
        <span>Expand</span>
      </button>
      <div class="step-body" id="step-body-${idx}">
        <div class="step-meta">
          <div class="meta-item"><strong>Input</strong><div>${escapeHtml(inputShape)}</div></div>
          <div class="meta-item"><strong>Output</strong><div>${escapeHtml(outputShape)}</div></div>
        </div>
        <div>
          <strong>${state.mode === "simple" ? "Simple explanation" : "Technical explanation"}</strong>
          <p>${escapeHtml(state.mode === "simple" ? step.simple : step.technical)}</p>
        </div>
        <div class="takeaway">${escapeHtml(keyTakeaway(step))}</div>
        ${step.math_formula ? `
          <div>
            <strong>Mathematical mapping</strong>
            <pre class="math-formula">${escapeHtml(step.math_formula)}</pre>
          </div>
        ` : ""}
        <div>
          <strong>Code snippet</strong>
          <pre class="code">${escapeHtml(step.code)}</pre>
        </div>
        ${step.preview_b64 ? `<img class="step-preview" src="${dataUri(step.preview_b64)}" alt="${escapeHtml(step.title)} preview" />` : ""}
      </div>
    `;

    container.appendChild(card);
  });

  container.querySelectorAll("[data-step-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-step-toggle");
      const body = el(`step-body-${id}`);
      if (!body) return;
      body.classList.toggle("open");
      const label = btn.querySelector("span:last-child");
      if (label) {
        label.textContent = body.classList.contains("open") ? "Collapse" : "Expand";
      }
    });
  });
}

function renderCodeBreakdown() {
  const blocks = [
    {
      title: "Block A: Load image",
      desc: "Open image and convert to RGB NumPy array.",
      code: "raw_img = Image.open(image_path).convert('RGB')\nimage = np.array(raw_img)",
    },
    {
      title: "Block B: Build center bias",
      desc: "Resize center-bias template so it matches image spatial size.",
      code:
        "centerbias = zoom(centerbias_template,\\n    (image.shape[0] / centerbias_template.shape[0],\\n     image.shape[1] / centerbias_template.shape[1]),\\n    order=0, mode='nearest')",
    },
    {
      title: "Block C: Tensor preparation",
      desc: "Convert image and center bias into model-ready tensors.",
      code:
        "img_input = np.array([image.transpose(2, 0, 1)])\\nimage_tensor = torch.tensor(img_input).to(DEVICE).float()\\ncenterbias_tensor = torch.tensor([centerbias]).to(DEVICE).float()",
    },
    {
      title: "Block D: Inference",
      desc: "Run DeepGaze under torch.no_grad() for inference-only execution.",
      code: "with torch.no_grad():\\n    log_density_prediction = model(image_tensor, centerbias_tensor)",
    },
    {
      title: "Block E: Post-processing",
      desc: "Convert log-density to probability-like map and visualize as heatmap.",
      code:
        "prediction = np.exp(log_density_prediction.detach().cpu().numpy()[0, 0])\\nplt.imshow(prediction, cmap='jet')",
    },
  ];

  const container = el("code-blocks");
  container.innerHTML = "";
  blocks.forEach((b) => {
    const node = document.createElement("article");
    node.className = "code-block";
    node.innerHTML = `<h3>${escapeHtml(b.title)}</h3><p>${escapeHtml(b.desc)}</p><pre class="code">${escapeHtml(b.code)}</pre>`;
    container.appendChild(node);
  });
}

function bindModeToggle() {
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderSteps();
    });
  });
}

function bindForm() {
  const slider = el("alpha-slider");
  slider.addEventListener("input", () => {
    setText("alpha-value", slider.value);
  });

  el("run-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fileInput = el("image-input");
    if (!fileInput.files.length) {
      setText("run-status", "Select an image first.");
      return;
    }

    setText("run-status", "Running inference pipeline...");

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("use_centerbias", el("centerbias-toggle").checked ? "true" : "false");
    formData.append("overlay_alpha", el("alpha-slider").value);

    try {
      const res = await fetch("/api/run", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to run model.");

      el("img-original").src = dataUri(data.results.original);
      el("img-heatmap").src = dataUri(data.results.heatmap);
      el("img-overlay").src = dataUri(data.results.overlay);

      el("download-heatmap").href = dataUri(data.results.heatmap);
      el("download-overlay").href = dataUri(data.results.overlay);

      el("result-grid").style.display = "grid";
      el("download-row").style.display = "flex";

      setText(
        "run-status",
        `Done. Probability range: ${data.results.probability_min.toExponential(2)} to ${data.results.probability_max.toExponential(2)}.`
      );
      setText(
        "model-note",
        data.model_mode === "deepgaze_iie"
          ? "Model mode: DeepGaze IIE"
          : "Model mode: educational fallback (install deepgaze_pytorch to use DeepGaze IIE weights)."
      );
      setText("centerbias-note", `Center bias source: ${data.centerbias_source}`);
      const warningText = (data.warnings || []).join(" ");
      setText("warning-note", warningText ? `Warning: ${warningText}` : "");
      el("warning-note").classList.toggle("warning-note", Boolean(warningText));
      setText("caution-text", data.caution);

      state.runTrace = data.trace;
      renderJourney(data.shape_journey);
      renderVisualPipeline(data.trace);
      renderSteps();
    } catch (err) {
      setText("run-status", err.message || "Unknown error while running demo.");
    }
  });
}

async function loadContent() {
  const res = await fetch("/api/content");
  const data = await res.json();

  state.stepsMeta = data.steps;

  renderIntro(data.intro);
  renderMapping(data.mapping_rows);
  renderProbabilityJourney(data.probability_journey);
  renderInterpretation(data.interpretation);
  renderJourney([
    { label: "Original image", shape: "file" },
    { label: "NumPy image array", shape: "(H, W, 3)" },
    { label: "Transposed array", shape: "(3, H, W)" },
    { label: "Batched image tensor", shape: "(1, 3, H, W)" },
    { label: "Center-bias map", shape: "(H, W)" },
    { label: "Batched center-bias tensor", shape: "(1, H, W)" },
    { label: "Log-density output", shape: "(H, W)" },
    { label: "Exponentiated prediction", shape: "(H, W)" },
  ]);
  renderVisualPipeline([]);
  renderSteps();
  renderCodeBreakdown();
}

async function init() {
  bindModeToggle();
  bindForm();
  await loadContent();
}

init();
