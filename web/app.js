const $ = (selector) => document.querySelector(selector);
const form = $("#promptForm");
const promptInput = $("#prompt");
const strategyInput = $("#strategy");
const maxTokensInput = $("#maxTokens");
const tokenOutput = $("#tokenOutput");
const generateButton = $("#generateButton");
const inspectButton = $("#inspectButton");
const responseCopy = $("#responseCopy");
const responseMeta = $("#responseMeta");
const finishBadge = $("#finishBadge");
const modelState = $("#modelState");
const promptTextArray = $("#promptTextArray");
const promptIdArray = $("#promptIdArray");
const generatedTextArray = $("#generatedTextArray");
const generatedIdArray = $("#generatedIdArray");
const promptTokenCount = $("#promptTokenCount");
const stepCounter = $("#stepCounter");
const previousStep = $("#previousStep");
const nextStep = $("#nextStep");
const replayButton = $("#replayButton");
const pauseButton = $("#pauseButton");
const playbackSpeed = $("#playbackSpeed");
const stageToken = $("#stageToken");
const stageEvidence = $("#stageEvidence");
const signal = $("#signal");
const stages = [...document.querySelectorAll(".stage")];
const pipeline = $(".pipeline");
const assemblyPrompt = $("#assemblyPrompt");
const assemblyCompletion = $("#assemblyCompletion");
const typingCursor = $("#typingCursor");
const traceLog = $("#traceLog");
const traceProgress = $("#traceProgress");
const traceProgressFill = $("#traceProgressFill");

let trace = null;
let traceTimer = null;
let activeStageIndex = -1;
let activeStageProgress = false;
let architecture = { embedding_dimension: 384, layers: 7, vocabulary_size: 4096 };

const stageDetails = [
  {
    title: "Tokens and IDs",
    formula: "text → BPE → token IDs",
    copy: "BPE splits prompt text into pieces. Each piece gets a vocabulary ID; generated IDs join the context later.",
    input: "Prompt text",
    output: "Ordered token IDs",
  },
  {
    title: "Token + position embeddings",
    formula: "token[id] + position[t] → vector",
    copy: "Look up a learned vector for each ID and position, then add the two vectors.",
    input: "ID + position",
    output: "One vector per token",
  },
  {
    title: "Seven Transformer layers",
    formula: "(causal attention + MLP) × layers → context",
    copy: "Each layer mixes information from current and earlier tokens, then refines it with an MLP and residual connections.",
    input: "Token vectors",
    output: "Context-aware vectors",
  },
  {
    title: "Linear vocabulary head",
    formula: "final vector → linear head → logits",
    copy: "Normalize the last context vector and score every possible next token. These raw scores are logits.",
    input: "Last context vector",
    output: "One logit per token",
  },
  {
    title: "Probabilities",
    formula: "adjust logits → softmax → probabilities",
    copy: "Apply the decoding settings, then softmax converts the adjusted scores into next-token chances.",
    input: "Adjusted logits",
    output: "Token probabilities",
  },
  {
    title: "Chosen token",
    formula: "choose ID → decode → append text",
    copy: "Pick an ID by greedy choice or sampling. Decode its text piece and add it to the reply and context.",
    input: "Next-token chances",
    output: "One new text piece",
  },
];

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function visibleToken(token) {
  return token ? token.replaceAll(" ", "␠").replaceAll("\n", "↵") : "∅";
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderFlow(items) {
  const flow = makeElement("div", "flow-diagram", "");
  items.forEach(([label, value], index) => {
    if (index) flow.append(makeElement("span", "flow-arrow", "→"));
    const node = makeElement("div", "flow-node", "");
    node.append(makeElement("small", "", label), makeElement("strong", "", value));
    flow.append(node);
  });
  stageEvidence.append(flow);
}

function eventLabel(event, result) {
  if (event.kind === "prompt") {
    const token = result.prompt_tokens[event.promptIndex];
    if (event.stage === 1) {
      return `Prompt ${event.promptIndex + 1}: token + position → ${token.embedding_vector.length}D embedding`;
    }
    return `Prompt ${event.promptIndex + 1}: ${visibleToken(token.text || token.piece)} → ID ${token.id} · position ${token.position}`;
  }
  const step = result.steps[event.tokenIndex];
  const prefix = `Prediction ${event.tokenIndex + 1}`;
  switch (event.stage) {
    case 0: return `${prefix}: ${step.context_token_count} context IDs enter the model`;
    case 1: return `${prefix}: ${architecture.embedding_dimension}D embedding · magnitude ${Number(step.embedding_norm).toFixed(2)}`;
    case 2: return `${prefix}: layer ${(event.layerIndex || 0) + 1}/${step.layer_details?.length || architecture.layers} · normalize → attend → add → normalize → MLP → add`;
    case 3: return `${prefix}: linear projection → ${architecture.vocabulary_size.toLocaleString()} raw logits`;
    case 4: return `${prefix}: top chance ${visibleToken(step.candidates[0]?.token || "")} ${percent(step.candidates[0]?.probability || 0)}`;
    default: return `${prefix}: chose ${visibleToken(step.selected_token)} · ID ${step.selected_token_id} · ${percent(step.selected_probability)}`;
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("status unavailable");
    const status = await response.json();
    architecture = status;
    modelState.classList.add("online");
    modelState.querySelector("span:last-child").textContent =
      `${status.mode} · ${(status.parameters / 1_000_000).toFixed(2)}M · ${status.device.toUpperCase()}`;
    $("#factParameters").textContent = `${(status.parameters / 1_000_000).toFixed(2)}M`;
    $("#factLayers").textContent = status.layers;
    $("#factContext").textContent = status.context_length;
    $("#transformerStageLabel").textContent = `${status.layers} layers · ${status.heads} heads`;
    $("#logitStageLabel").textContent = `${status.vocabulary_size.toLocaleString()} choices`;
    $("#architectureEmbeddings").textContent = `${status.embedding_dimension}D embeddings`;
    $("#architectureTransformer").textContent = `${status.layers} causal Transformer layers`;
    $("#architectureVocabulary").textContent = `${status.vocabulary_size.toLocaleString()} token scores`;
    stageDetails[2].title = `${status.layers} Transformer layers`;
    stageDetails[1].formula = `token[id] + position[t] → vector[${status.embedding_dimension}]`;
    stageDetails[2].formula = `(causal attention + MLP) × ${status.layers} → context`;
    stageDetails[3].formula = `final vector → linear head → ${status.vocabulary_size.toLocaleString()} logits`;
    stageDetails[1].output = `${status.embedding_dimension} numbers per token`;
    stageDetails[2].output = `Context after ${status.layers} layers`;
    stageDetails[3].output = `${status.vocabulary_size.toLocaleString()} raw scores`;
  } catch {
    modelState.classList.add("error");
    modelState.querySelector("span:last-child").textContent = "Local model unavailable";
  }
}

function activateStage(index, showProgress = false) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const previousStageIndex = activeStageIndex;
  const resetting = reducedMotion || previousStageIndex < 0 || index <= previousStageIndex || index - previousStageIndex > 1;
  pipeline.classList.toggle("resetting", resetting);
  signal.classList.toggle("resetting", resetting);
  activeStageIndex = index;
  activeStageProgress = showProgress;
  stages.forEach((stage, stageIndex) => {
    stage.classList.toggle("active", stageIndex === index);
    stage.classList.toggle("completed", showProgress && stageIndex < index);
    if (stageIndex === index) stage.setAttribute("aria-current", "step");
    else stage.removeAttribute("aria-current");
  });
  const detail = stageDetails[index];
  $("#explainerNumber").textContent = `STEP ${index + 1} OF ${stages.length}`;
  $("#explainerTitle").textContent = detail.title;
  $("#explainerFormula").textContent = detail.formula;
  $("#explainerCopy").textContent = detail.copy;
  $("#explainerInput").textContent = detail.input;
  $("#explainerOutput").textContent = detail.output;
  updatePipelinePosition();
  if (resetting) requestAnimationFrame(() => {
    pipeline.classList.remove("resetting");
    signal.classList.remove("resetting");
  });
}

function updatePipelinePosition() {
  const index = Math.max(0, activeStageIndex);
  const firstCenter = stages[0].offsetTop + stages[0].offsetHeight / 2;
  const lastCenter = stages.at(-1).offsetTop + stages.at(-1).offsetHeight / 2;
  const currentCenter = stages[index].offsetTop + stages[index].offsetHeight / 2;
  const marker = getComputedStyle(stages[0], "::before");
  const railX = stages[0].offsetLeft + stages[0].clientLeft + parseFloat(marker.left) + parseFloat(marker.width) / 2;
  const travelDuration = { fast: "140ms", normal: "480ms", slow: "850ms" }[playbackSpeed.value] || "480ms";
  pipeline.style.setProperty("--travel-duration", travelDuration);
  pipeline.style.setProperty("--rail-x", `${railX}px`);
  pipeline.style.setProperty("--rail-top", `${firstCenter}px`);
  pipeline.style.setProperty("--rail-total", `${lastCenter - firstCenter}px`);
  pipeline.style.setProperty("--rail-progress", `${activeStageProgress ? currentCenter - firstCenter : 0}px`);
  signal.style.setProperty("--travel-duration", travelDuration);
  signal.style.left = `${pipeline.offsetLeft + railX}px`;
  signal.style.top = `${pipeline.offsetTop + currentCenter}px`;
  signal.style.opacity = activeStageIndex < 0 ? "0" : "1";
}

function renderEmbeddingEvidence(detail, label, tokenId) {
  stageEvidence.replaceChildren(makeElement("div", "evidence-label", label));
  const values = detail.embedding_vector;
  if (!values?.length) return;
  renderFlow([["Token ID", String(tokenId)], ["Position", String(detail.position)], ["Input vector", `${values.length} numbers`]]);
  stageEvidence.append(makeElement("p", "evidence-note", "The model looks up a learned vector for the token ID and another for its position. It adds them coordinate by coordinate before the Transformer."));

  const calculation = makeElement("div", "embedding-calculation", "");
  [
    ["Token lookup", detail.token_embedding_preview],
    ["+ Position lookup", detail.position_embedding_preview],
    ["= Input embedding", values.slice(0, 8)],
  ].forEach(([name, preview]) => {
    const row = makeElement("div", "embedding-calculation-row", "");
    row.append(makeElement("strong", "", name), makeElement("code", "", `[${preview.map((value) => Number(value).toFixed(2)).join(", ")}]`));
    calculation.append(row);
  });
  stageEvidence.append(calculation);
  stageEvidence.append(makeElement("p", "evidence-note", `The calculation above shows the first 8 coordinates. Below are all ${values.length} coordinates of this token's input embedding. Vector magnitude: ${Number(detail.embedding_norm).toFixed(2)}.`));

  const legend = makeElement("div", "embedding-grid-legend", "Light blue = negative · white = near zero · dark blue = positive");
  const grid = makeElement("div", "embedding-dimension-grid", "");
  const readout = makeElement("div", "embedding-coordinate-readout", "Select a square to read its exact dimension and value.");
  const scale = Math.max(...values.map((value) => Math.abs(value)), 0.01);
  values.forEach((value, index) => {
    const intensity = Math.min(Math.abs(value) / scale, 1);
    const target = value < 0 ? [105, 165, 223] : [24, 119, 242];
    const channels = target.map((channel) => Math.round(255 - (255 - channel) * intensity));
    const cell = makeElement("button", "embedding-dimension", "");
    cell.type = "button";
    cell.style.backgroundColor = `rgb(${channels.join(",")})`;
    cell.title = `Dimension ${index + 1}: ${Number(value).toFixed(5)}`;
    cell.setAttribute("aria-label", cell.title);
    cell.addEventListener("click", () => {
      readout.textContent = `Dimension ${index + 1} of ${values.length}: ${Number(value).toFixed(5)}`;
    });
    grid.append(cell);
  });
  stageEvidence.append(legend, grid, readout);
  const raw = makeElement("details", "embedding-raw", "");
  raw.append(makeElement("summary", "", "View the full numeric vector"), makeElement("code", "", `[${values.map((value) => Number(value).toFixed(4)).join(", ")}]`));
  stageEvidence.append(raw);
}

function renderAttention(focus, parent = stageEvidence) {
  if (!focus?.length) return;
  parent.append(makeElement("p", "evidence-note", "Head 1 · last context position · five strongest links. The weights show where this head mixes information; the other heads have their own weights."));
  const list = makeElement("div", "attention-list", "");
  focus.forEach((item) => {
    const row = makeElement("div", "attention-row", "");
    const track = makeElement("span", "attention-track", "");
    const fill = makeElement("span", "attention-fill", "");
    fill.style.width = `${item.probability * 100}%`;
    track.append(fill);
    row.append(
      makeElement("span", "attention-token", `${item.position}: ${visibleToken(item.token) || "control token"}`),
      track,
      makeElement("strong", "", percent(item.probability)),
    );
    list.append(row);
  });
  parent.append(list);
}

function renderLayerEvidence(step, layerIndex = 0) {
  const layers = step.layer_details || [];
  if (!layers.length) {
    stageEvidence.append(makeElement("p", "evidence-note", "Restart the local server and generate again to capture the operations inside each layer."));
    return;
  }
  const layer = layers[layerIndex];
  const measured = Object.fromEntries(layer.activations.map((item) => [item.name, item]));
  const layerRail = makeElement("div", "layer-rail", "");
  layerRail.setAttribute("aria-label", "Choose a Transformer layer");
  layers.forEach((item, index) => {
    if (index) layerRail.append(makeElement("span", "layer-connector", "→"));
    const button = makeElement("button", `layer-choice${index === layerIndex ? " active" : ""}`, `L${item.index}`);
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === layerIndex));
    button.title = `Inspect layer ${item.index}`;
    button.addEventListener("click", () => {
      pauseTrace();
      const cursor = trace.events.findIndex((event) => event.kind === "generation" && event.tokenIndex === step.index && event.stage === 2 && event.layerIndex === index);
      if (cursor >= 0) moveTo(cursor);
    });
    layerRail.append(button);
  });
  stageEvidence.append(layerRail);
  const heading = makeElement("div", "operation-heading", "");
  heading.append(makeElement("strong", "", `Inside layer ${layer.index}`), makeElement("span", "", `[1, ${step.context_token_count}, ${measured.input.width}]`));
  stageEvidence.append(heading, makeElement("p", "evidence-note", "Follow the path downward. Values on the right are measured vector magnitudes for the last context token; a larger number does not mean a better answer."));

  const groups = [
    {
      carry: "Carry the original input to the first +",
      operations: [
        ["attention_norm", "1", "Normalize", "LayerNorm balances the input coordinates."],
        ["attention", "2", "Causal attention", `${architecture.heads} heads mix the current token with earlier tokens. Future positions are masked.`],
        ["attention_residual", "+", "Add the original input", "Original input + attention update → first residual."],
      ],
    },
    {
      carry: "Carry the first residual to the second +",
      operations: [
        ["feed_forward_norm", "3", "Normalize again", "Prepare the first residual for the feed-forward network."],
        ["feed_forward", "4", "Feed-forward network", `${measured.input.width} → ${architecture.feed_forward_dimension} → ${measured.output.width} · Linear → GELU → Linear.`],
        ["output", "+", "Add the first residual", "First residual + feed-forward update → layer output."],
      ],
    },
  ];
  groups.forEach((group) => {
    const path = makeElement("div", "residual-path", "");
    path.append(makeElement("div", "residual-caption", group.carry));
    const flow = makeElement("ol", "operation-flow", "");
    group.operations.forEach(([key, marker, title, explanation]) => {
      const row = makeElement("li", `operation-row${marker === "+" ? " addition" : ""}`, "");
      const copy = makeElement("div", "operation-copy", "");
      copy.append(makeElement("strong", "", title), makeElement("p", "", explanation));
      const metric = makeElement("div", "operation-metric", "");
      metric.append(makeElement("small", "", "MAGNITUDE"), makeElement("strong", "", Number(measured[key].norm).toFixed(2)));
      row.append(makeElement("span", "operation-marker", marker), copy, metric);
      flow.append(row);
    });
    path.append(flow);
    stageEvidence.append(path);
  });
  stageEvidence.append(makeElement("div", "vector-exit", layerIndex < layers.length - 1
    ? `↓ This layer's output becomes layer ${layer.index + 1}'s input.`
    : "↓ Final layer output goes to the vocabulary scoring step."));

  const attention = makeElement("details", "operation-details", "");
  attention.append(makeElement("summary", "", "Look inside attention"));
  attention.append(makeElement("p", "evidence-note", "Learned projections form queries (Q), keys (K) and values (V). Q·K scores are scaled and masked, softmax turns them into weights, and the weights mix V. The heads are then merged and projected."));
  renderAttention(layer.attention_focus, attention);
  const raw = makeElement("details", "operation-details", "");
  raw.append(makeElement("summary", "", "Inspect measured vectors · first 8 coordinates"));
  layer.activations.forEach((item) => {
    const row = makeElement("div", "activation-values", "");
    row.append(makeElement("span", "", item.name.replaceAll("_", " ")), makeElement("code", "", `[${item.preview.map((value) => Number(value).toFixed(3)).join(", ")}]`));
    raw.append(row);
  });
  stageEvidence.append(attention, raw);
}

function renderScoreEvidence(step) {
  stageEvidence.replaceChildren(makeElement("div", "evidence-label", "Final normalization → linear projection"));
  renderFlow([["Context vector", `${architecture.embedding_dimension} values`], ["Linear head", "matrix multiply"], ["Raw logits", `${architecture.vocabulary_size.toLocaleString()} scores`]]);
  if (step.projection_input_norm != null) {
    stageEvidence.append(makeElement("p", "evidence-note", `After final LayerNorm, the last token's vector has magnitude ${Number(step.projection_input_norm).toFixed(2)}. The learned output matrix gives every vocabulary token a score.`));
  }
  stageEvidence.append(makeElement("div", "evidence-label", "Raw scores for the candidate tokens"));
  stageEvidence.append(makeElement("p", "evidence-note", "Logits can be negative or positive. They become probabilities in the next step. These are the candidates shown by the decoder, before its adjustments."));
  const list = makeElement("div", "score-list", "");
  const scale = Math.max(...step.candidates.map((candidate) => Math.abs(candidate.raw_logit)), 0.01);
  [...step.candidates].sort((left, right) => right.raw_logit - left.raw_logit).forEach((candidate) => {
    const row = makeElement("div", "score-row", "");
    const track = makeElement("span", "score-track", "");
    const fill = makeElement("span", `score-fill${candidate.raw_logit < 0 ? " negative" : ""}`, "");
    const width = Math.abs(candidate.raw_logit) / scale * 50;
    fill.style.width = `${width}%`;
    fill.style.left = `${candidate.raw_logit < 0 ? 50 - width : 50}%`;
    track.append(fill);
    row.append(makeElement("span", "", candidateText(candidate)), track, makeElement("strong", "", Number(candidate.raw_logit).toFixed(2)));
    list.append(row);
  });
  stageEvidence.append(list, makeElement("p", "chart-axis-note", "Negative ← 0 → Positive"));
}

function candidateText(candidate) {
  if (candidate.token_id === architecture.eos_token_id) return "<EOS>";
  return visibleToken(candidate.token) || `ID ${candidate.token_id}`;
}

function renderTokenEvidence(event, result, pageStart) {
  const isPrompt = event.kind === "prompt";
  const step = isPrompt ? null : result.steps[event.tokenIndex];
  const ids = isPrompt ? result.prompt_tokens.map((token) => token.id) : step.context_token_ids;
  const tokens = isPrompt ? result.prompt_tokens : [
    ...result.prompt_tokens,
    ...result.steps.slice(0, event.tokenIndex).map((item) => ({ id: item.selected_token_id, text: item.selected_token })),
  ].slice(-ids.length);
  const activeIndex = isPrompt ? event.promptIndex : ids.length - 1;
  const pageSize = 8;
  const start = pageStart ?? Math.floor(activeIndex / pageSize) * pageSize;
  const end = Math.min(start + pageSize, ids.length);
  const inputText = isPrompt ? result.prompt : result.steps[event.tokenIndex - 1]?.combined_text || result.prompt;
  if (!isPrompt) {
    $("#explainerFormula").textContent = "prompt IDs + generated IDs → model input";
    $("#explainerCopy").textContent = "The prompt is tokenized once. Each generated ID is appended directly; the text is not tokenized again for each prediction.";
    $("#explainerInput").textContent = "Current context IDs";
    $("#explainerOutput").textContent = `[1, ${ids.length}] input`;
  }
  stageEvidence.replaceChildren();
  const source = makeElement("section", "token-process", "");
  source.append(
    makeElement("div", "evidence-label", isPrompt ? "01 · Text sent to the tokenizer" : "01 · Context so far"),
    makeElement("div", "token-source", inputText),
  );
  const mapping = makeElement("section", "token-process", "");
  mapping.append(
    makeElement("div", "evidence-label", "02 · Text pieces and vocabulary IDs"),
    makeElement("p", "evidence-note", "BPE pieces can be words, word parts, or punctuation. ␠ marks a space; ↵ marks a newline. Position is the place in this input; ID is the entry in the vocabulary."),
  );
  const table = makeElement("table", "token-mapping", "");
  table.setAttribute("aria-label", "Actual text pieces and vocabulary IDs by input position");
  const header = table.createTHead().insertRow();
  ["Position", "Text piece", "Token ID"].forEach((label) => {
    const cell = makeElement("th", "", label);
    cell.scope = "col";
    header.append(cell);
  });
  const body = table.createTBody();
  for (let index = start; index < end; index += 1) {
    const token = tokens[index];
    const piece = visibleToken(token.text || token.piece);
    const row = body.insertRow();
    row.classList.toggle("current-token", index === activeIndex);
    row.insertCell().textContent = String(index);
    const pieceCell = row.insertCell();
    if (isPrompt) {
      const button = makeElement("button", "token-select", piece);
      button.type = "button";
      button.setAttribute("aria-label", `Inspect position ${index}, piece ${piece}, ID ${ids[index]}`);
      button.setAttribute("aria-pressed", String(index === activeIndex));
      button.addEventListener("click", () => { pauseTrace(); moveTo(index * 2); });
      pieceCell.append(button);
    } else {
      pieceCell.append(makeElement("code", "", piece));
    }
    row.insertCell().textContent = String(ids[index]);
  }
  mapping.append(table);
  if (ids.length > pageSize) {
    const pagination = makeElement("div", "mapping-pagination", "");
    pagination.append(makeElement("span", "", `Positions ${start}–${end - 1} of ${ids.length} tokens`));
    [["Earlier", start - pageSize, start === 0], ["Later", start + pageSize, end === ids.length]].forEach(([label, offset, disabled]) => {
      const button = makeElement("button", "secondary-button", label);
      button.type = "button";
      button.disabled = disabled;
      button.addEventListener("click", () => { pauseTrace(); renderTokenEvidence(event, result, offset); });
      pagination.append(button);
    });
    mapping.append(pagination);
  }
  mapping.append(makeElement("p", "evidence-note", isPrompt
    ? "<BOS> is added at the start of the prompt. Select any piece to follow that token."
    : "These are the IDs already in the context. The highlighted last token is used for the next prediction."));
  const input = makeElement("section", "token-process", "");
  const tensor = makeElement("code", "token-input-tensor", JSON.stringify([ids]));
  input.append(
    makeElement("div", "evidence-label", "03 · Integer input to the model"),
    tensor,
    makeElement("p", "evidence-note", `Shape [1, ${ids.length}]: 1 sequence with ${ids.length} tokens. Each ID selects a learned embedding in the next step.`),
  );
  const next = makeElement("button", "secondary-button token-embedding-link", isPrompt
    ? `See position ${activeIndex}'s embedding →`
    : "See the last context token's embedding →");
  next.type = "button";
  next.addEventListener("click", () => {
    pauseTrace();
    const cursor = trace.events.findIndex((item) => item.kind === event.kind && item.stage === 1 &&
      (isPrompt ? item.promptIndex === event.promptIndex : item.tokenIndex === event.tokenIndex));
    moveTo(cursor);
  });
  input.append(next);
  stageEvidence.append(source, mapping, input);
}

function renderStageEvidence(event, result) {
  stageEvidence.replaceChildren();
  if (event.kind === "prompt") {
    const token = result.prompt_tokens[event.promptIndex];
    if (event.stage === 1) {
      renderEmbeddingEvidence(token, `Prompt token ${event.promptIndex + 1} · ${visibleToken(token.text || token.piece)}`, token.id);
      return;
    }
    if (event.stage > 1) {
      stageEvidence.append(makeElement("p", "evidence-note", "Inspect prompt only computes token IDs and embeddings. Click Generate to run the Transformer and see later measured stages."));
      return;
    }
    renderTokenEvidence(event, result);
    return;
  }
  const step = result.steps[event.tokenIndex];
  if (event.stage === 0) {
    renderTokenEvidence(event, result);
  } else if (event.stage === 1) {
    renderEmbeddingEvidence(
      { ...step.embedding_components, embedding_norm: step.embedding_norm },
      "Last context token · token + position vector",
      step.context_token_ids.at(-1),
    );
  } else if (event.stage === 2) {
    renderLayerEvidence(step, event.layerIndex || 0);
  } else if (event.stage === 3) {
    renderScoreEvidence(step);
  } else if (event.stage === 4) {
    stageEvidence.append(makeElement("div", "evidence-label", "Measured next-token distribution"));
    const settings = result.settings;
    renderFlow([["Raw scores", "one per token"], ["Decoder settings", settings.strategy === "greedy" ? "repetition penalty" : "temperature + filters"], ["Softmax", "probabilities"]]);
    stageEvidence.append(makeElement("p", "evidence-note", settings.strategy === "greedy"
      ? `Repetition penalty ${settings.repetition_penalty}. Greedy decoding chooses the highest remaining score; the bars show its softmax distribution.`
      : `Temperature ${settings.temperature} · top-k ${settings.top_k} · top-p ${settings.top_p} · repetition penalty ${settings.repetition_penalty}. The decoder samples from the resulting probabilities.`));
    step.candidates.forEach((candidate) => {
      const row = makeElement("div", `probability evidence-probability${candidate.token_id === step.selected_token_id ? " selected" : ""}`, "");
      const track = makeElement("span", "probability-track", "");
      const fill = makeElement("span", "probability-fill", "");
      fill.style.width = `${candidate.probability * 100}%`;
      track.append(fill);
      row.append(makeElement("span", "probability-token", candidateText(candidate)), track, makeElement("span", "probability-value", percent(candidate.probability)));
      stageEvidence.append(row);
    });
    const remainder = Math.max(0, 1 - step.candidates.reduce((sum, candidate) => sum + candidate.probability, 0));
    stageEvidence.append(makeElement("p", "evidence-note", `Other tokens together: ${percent(remainder)}. Showing the top ${step.candidates.length}; sampling can choose a token outside this list.`));
  } else {
    const isEos = step.selected_token_id === architecture.eos_token_id;
    const piece = candidateText({ token: step.selected_token, token_id: step.selected_token_id });
    stageEvidence.append(makeElement("div", "evidence-label", result.settings.strategy === "greedy" ? "Choose the highest score" : "Sample one token"));
    renderFlow([["Distribution", "next-token chances"], ["Selected ID", String(step.selected_token_id)], ["Decoded piece", piece]]);
    stageEvidence.append(makeElement("div", "chosen-evidence", piece));
    stageEvidence.append(makeElement("p", "evidence-note", `Chance under these settings: ${percent(step.selected_probability)}.`));
    const isLast = event.tokenIndex === result.steps.length - 1;
    stageEvidence.append(makeElement("div", "vector-exit", isEos
      ? "EOS marks the end. Generation stops here."
      : isLast
        ? "Append this piece to the reply. The requested token limit is reached."
        : "Append this token to the context → run the same model again → predict the next piece."));
  }
}

function renderTokenArrays(result, generatedCount) {
  promptTextArray.textContent = JSON.stringify(result.prompt_tokens.map((token) => token.text || token.piece));
  promptIdArray.textContent = JSON.stringify(result.prompt_tokens.map((token) => token.id));
  generatedTextArray.textContent = JSON.stringify(result.steps?.slice(0, generatedCount).map((step) => step.selected_token) || []);
  generatedIdArray.textContent = JSON.stringify(result.steps?.slice(0, generatedCount).map((step) => step.selected_token_id) || []);
  promptTokenCount.textContent = `${result.prompt_tokens.length} tokens`;
}

function renderAnswer(result, generatedCount, atEnd) {
  assemblyPrompt.textContent = result.prompt;
  const step = generatedCount ? result.steps[generatedCount - 1] : null;
  if (step?.combined_text?.startsWith(result.prompt)) {
    assemblyCompletion.textContent = step.combined_text.slice(result.prompt.length);
  } else {
    assemblyCompletion.textContent = step?.partial_completion ? ` ${step.partial_completion}` : "";
  }
  typingCursor.classList.toggle("active", !atEnd && result.steps?.length > 0);
  if (!result.steps?.length) {
    responseCopy.textContent = atEnd
      ? "Prompt inspected. Click Generate to watch the model predict a reply."
      : "Converting your prompt into token IDs…";
    finishBadge.textContent = atEnd ? "Prompt ready" : "Tokenizing";
    responseMeta.textContent = `${result.prompt_tokens.length} prompt tokens`;
  } else {
    responseCopy.textContent = generatedCount
      ? step.partial_completion || "The model selected an end token."
      : "Waiting for the first selected token…";
    finishBadge.textContent = atEnd ? (result.finish_reason === "eos" ? "EOS reached" : "Complete") : "Building";
    responseMeta.textContent = `${generatedCount} of ${result.steps.length} generated tokens added`;
  }
}

function renderLog(events, result, cursor) {
  traceLog.replaceChildren();
  events.slice(0, cursor + 1).forEach((event, index) => {
    const item = makeElement("li", index === cursor ? "current" : "", "");
    const button = makeElement("button", "log-entry", `${String(index + 1).padStart(2, "0")}  ${eventLabel(event, result)}`);
    button.type = "button";
    button.addEventListener("click", () => { pauseTrace(); moveTo(index); });
    item.append(button);
    traceLog.append(item);
  });
  traceLog.scrollTop = traceLog.scrollHeight;
}

function renderCursor() {
  if (!trace || trace.cursor < 0) return;
  const { result, events, cursor } = trace;
  const event = events[cursor];
  const atEnd = cursor === events.length - 1;
  const promptCount = event.kind === "prompt" ? event.promptIndex + 1 : result.prompt_tokens.length;
  const generatedCount = event.kind === "prompt"
    ? 0
    : event.tokenIndex + (event.stage === 5 ? 1 : 0);
  activateStage(event.stage, event.kind === "generation" || event.stage === 1);
  renderStageEvidence(event, result);
  renderTokenArrays(result, generatedCount);
  renderAnswer(result, generatedCount, atEnd);
  renderLog(events, result, cursor);
  stageToken.textContent = event.kind === "generation" && event.stage === 5
    ? `“${visibleToken(result.steps[event.tokenIndex].selected_token)}”`
    : "Added to the reply";
  stepCounter.textContent = event.kind === "prompt"
    ? `Prompt token ${promptCount} of ${result.prompt_tokens.length} · step ${event.stage + 1}/2 · event ${cursor + 1}/${events.length}`
    : `Prediction ${event.tokenIndex + 1}/${result.steps.length} · step ${event.stage + 1}/6${event.stage === 2 && result.steps[event.tokenIndex].layer_details?.length ? ` · layer ${(event.layerIndex || 0) + 1}/${result.steps[event.tokenIndex].layer_details.length}` : ""}`;
  const progress = Math.round((cursor + 1) / events.length * 100);
  traceProgress.setAttribute("aria-valuenow", String(progress));
  traceProgressFill.style.width = `${progress}%`;
  previousStep.disabled = cursor === 0;
  nextStep.disabled = atEnd;
  pauseButton.disabled = atEnd;
  if (atEnd) {
    trace.playing = false;
    pauseButton.textContent = "Play";
  }
}

function stopTimer() {
  clearTimeout(traceTimer);
  traceTimer = null;
}

function pauseTrace() {
  stopTimer();
  if (trace) trace.playing = false;
  pauseButton.textContent = "Play";
}

function moveTo(index) {
  if (!trace || index < 0 || index >= trace.events.length) return;
  stopTimer();
  trace.cursor = index;
  renderCursor();
  if (trace.playing && index < trace.events.length - 1) {
    const delay = { fast: 200, normal: 850, slow: 1800 }[playbackSpeed.value] || 850;
    traceTimer = setTimeout(() => moveTo(index + 1), trace.events[index].stage === 2 ? delay * 1.5 : delay);
  }
}

function startTrace(result, autoplay) {
  pauseTrace();
  const events = result.prompt_tokens.flatMap((_, promptIndex) => [
    { kind: "prompt", promptIndex, stage: 0 },
    { kind: "prompt", promptIndex, stage: 1 },
  ]);
  result.steps?.forEach((step, tokenIndex) => {
    for (let stage = 0; stage < stages.length; stage += 1) {
      if (stage === 2 && step.layer_details?.length) {
        step.layer_details.forEach((_, layerIndex) => {
          events.push({ kind: "generation", tokenIndex, stage, layerIndex });
        });
      } else {
        events.push({ kind: "generation", tokenIndex, stage });
      }
    }
  });
  trace = { result, events, cursor: 0, playing: autoplay };
  replayButton.disabled = false;
  pauseButton.textContent = autoplay ? "Pause" : "Play";
  moveTo(0);
  $("#labTitle").scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    block: "start",
  });
}

async function requestTrace(route, body, autoplay) {
  pauseTrace();
  generateButton.disabled = true;
  inspectButton.disabled = true;
  responseCopy.className = "response-copy loading";
  responseCopy.textContent = route === "/api/tokenize"
    ? "Inspecting prompt tokens and embeddings…"
    : "JeePeeTee is calculating the predictions…";
  finishBadge.textContent = "Calculating";
  try {
    const response = await fetch(route, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    startTrace(payload, autoplay);
  } catch (error) {
    responseCopy.className = "response-copy error";
    responseCopy.textContent = error.message;
    responseMeta.textContent = "Check your prompt and the local JeePeeTee server.";
    finishBadge.textContent = "Error";
  } finally {
    generateButton.disabled = false;
    inspectButton.disabled = false;
    responseCopy.classList.remove("loading");
  }
}

maxTokensInput.addEventListener("input", () => { tokenOutput.textContent = maxTokensInput.value; });
inspectButton.addEventListener("click", () => {
  requestTrace("/api/tokenize", { prompt: promptInput.value }, false);
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  requestTrace("/api/generate", {
    prompt: promptInput.value,
    strategy: strategyInput.value,
    max_new_tokens: Number(maxTokensInput.value),
  }, true);
});
previousStep.addEventListener("click", () => { pauseTrace(); moveTo(trace.cursor - 1); });
nextStep.addEventListener("click", () => { pauseTrace(); moveTo(trace.cursor + 1); });
pauseButton.addEventListener("click", () => {
  if (!trace) return;
  if (trace.playing) {
    pauseTrace();
  } else {
    trace.playing = true;
    pauseButton.textContent = "Pause";
    moveTo(trace.cursor + 1);
  }
});
replayButton.addEventListener("click", () => { if (trace) startTrace(trace.result, false); });
stages.forEach((stage, index) => {
  stage.addEventListener("click", () => {
    pauseTrace();
    if (trace) {
      const event = trace.events[trace.cursor];
      const cursor = trace.events.findIndex((item) => {
        if (event.kind === "prompt" && index < 2) {
          return item.kind === "prompt" && item.promptIndex === event.promptIndex && item.stage === index;
        }
        return item.kind === "generation" && item.tokenIndex === (event.tokenIndex || 0) && item.stage === index;
      });
      if (cursor >= 0) {
        moveTo(cursor);
        return;
      }
      renderStageEvidence({ ...event, stage: index }, trace.result);
    }
    activateStage(index);
  });
});

new ResizeObserver(updatePipelinePosition).observe(pipeline);
loadStatus();
