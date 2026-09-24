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
const tokenStream = $("#tokenStream");
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

function renderBlockDiagram() {
  const diagram = makeElement("div", "block-diagram", "");
  [
    ["01", "Norm → Q/K/V → causal attention"],
    ["02", "+ residual"],
    ["03", "Norm → GELU MLP"],
    ["04", "+ residual"],
  ].forEach(([number, label], index) => {
    const operation = makeElement("div", `block-operation${index === 0 ? " strong" : ""}`, "");
    operation.append(makeElement("b", "", number), document.createTextNode(label));
    diagram.append(operation);
  });
  stageEvidence.append(diagram);
}

function eventLabel(event, result) {
  if (event.kind === "prompt") {
    const token = result.prompt_tokens[event.promptIndex];
    return `Prompt ${event.promptIndex + 1}: ${visibleToken(token.text || token.piece)} → ID ${token.id} · position ${token.position}`;
  }
  const step = result.steps[event.tokenIndex];
  const prefix = `Prediction ${event.tokenIndex + 1}`;
  switch (event.stage) {
    case 0: return `${prefix}: ${step.context_token_count} context IDs enter the model`;
    case 1: return `${prefix}: ${architecture.embedding_dimension}D embedding · magnitude ${Number(step.embedding_norm).toFixed(2)}`;
    case 2: return `${prefix}: ${step.layer_norms.length} causal layers · final magnitude ${Number(step.layer_norms.at(-1)).toFixed(2)}`;
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
  const firstCenter = stages[0].offsetTop + stages[0].offsetHeight / 2;
  const lastCenter = stages.at(-1).offsetTop + stages.at(-1).offsetHeight / 2;
  const currentCenter = stages[index].offsetTop + stages[index].offsetHeight / 2;
  const railX = stages[0].offsetLeft + 18;
  const travelDuration = { fast: "140ms", normal: "480ms", slow: "850ms" }[playbackSpeed.value] || "480ms";
  pipeline.style.setProperty("--travel-duration", travelDuration);
  pipeline.style.setProperty("--rail-x", `${railX}px`);
  pipeline.style.setProperty("--rail-top", `${firstCenter}px`);
  pipeline.style.setProperty("--rail-total", `${lastCenter - firstCenter}px`);
  pipeline.style.setProperty("--rail-progress", `${showProgress ? currentCenter - firstCenter : 0}px`);
  signal.style.setProperty("--travel-duration", travelDuration);
  const scene = $("#pipelineScene");
  signal.style.left = `${pipeline.offsetLeft + railX}px`;
  signal.style.top = `${pipeline.offsetTop + currentCenter}px`;
  signal.style.opacity = "1";
  if (trace?.playing && index !== previousStageIndex && scene.scrollHeight > scene.clientHeight) {
    const stageTop = pipeline.offsetTop + stages[index].offsetTop;
    const stageBottom = stageTop + stages[index].offsetHeight;
    if (stageTop < scene.scrollTop + 16 || stageBottom > scene.scrollTop + scene.clientHeight - 16) {
      scene.scrollTo({
        top: Math.max(0, pipeline.offsetTop + currentCenter - scene.clientHeight / 2),
        behavior: reducedMotion || playbackSpeed.value === "fast" ? "auto" : "smooth",
      });
    }
  }
  if (resetting) requestAnimationFrame(() => {
    pipeline.classList.remove("resetting");
    signal.classList.remove("resetting");
  });
}

function renderEmbeddingEvidence(values, norm, label, tokenId) {
  stageEvidence.replaceChildren(makeElement("div", "evidence-label", label));
  if (!values?.length) return;
  renderFlow([["Token ID", String(tokenId)], ["Learned tables", "token + position"], ["Vector", `${architecture.embedding_dimension} values`]]);
  stageEvidence.append(makeElement("p", "evidence-note", `First ${values.length} of ${architecture.embedding_dimension} coordinates · vector magnitude ${Number(norm).toFixed(2)}`));
  const chart = makeElement("div", "embedding-chart", "");
  const ceiling = Math.max(...values.map((value) => Math.abs(value)), 0.01);
  values.forEach((value, index) => {
    const column = makeElement("div", "embedding-column", "");
    const bar = makeElement("span", `embedding-bar ${value < 0 ? "negative" : "positive"}`, "");
    bar.style.height = `${Math.max(6, Math.abs(value) / ceiling * 48)}px`;
    bar.title = `Dimension ${index + 1}: ${Number(value).toFixed(3)}`;
    column.append(bar, makeElement("small", "embedding-value", Number(value).toFixed(2)));
    chart.append(column);
  });
  stageEvidence.append(chart);
}

function renderAttention(focus) {
  if (!focus?.length) return;
  stageEvidence.append(makeElement("div", "evidence-label attention-label", "Final layer · attention head 1 · strongest context links"));
  stageEvidence.append(makeElement("p", "evidence-note", "For the last context position, these are the five largest attention weights in one head."));
  const list = makeElement("div", "attention-list", "");
  focus.forEach((item) => {
    const row = makeElement("div", "attention-row", "");
    const track = makeElement("span", "attention-track", "");
    const fill = makeElement("span", "attention-fill", "");
    fill.style.width = `${Math.max(item.probability * 100, 2)}%`;
    track.append(fill);
    row.append(
      makeElement("span", "attention-token", `${item.position}: ${visibleToken(item.token)}`),
      track,
      makeElement("strong", "", percent(item.probability)),
    );
    list.append(row);
  });
  stageEvidence.append(list);
}

function renderLayerEvidence(step) {
  const norms = step.layer_norms || [];
  stageEvidence.replaceChildren(makeElement("div", "evidence-label", `Measured output magnitudes · ${norms.length} layers`));
  renderBlockDiagram();
  stageEvidence.append(makeElement("p", "evidence-note", `This block repeats ${architecture.layers} times. Bars show the last token's output magnitude after each layer.`));
  const ceiling = Math.max(...norms, 0.01);
  const list = makeElement("div", "layer-list", "");
  norms.forEach((norm, index) => {
    const row = makeElement("div", "layer-row", "");
    const track = makeElement("span", "layer-track", "");
    const fill = makeElement("span", "layer-fill", "");
    fill.style.width = `${Math.max(3, norm / ceiling * 100)}%`;
    track.append(fill);
    row.append(makeElement("span", "", `L${index + 1}`), track, makeElement("strong", "", Number(norm).toFixed(1)));
    list.append(row);
  });
  stageEvidence.append(list);
  renderAttention(step.attention_focus);
}

function renderScoreEvidence(step) {
  stageEvidence.replaceChildren(makeElement("div", "evidence-label", "Final normalization → linear projection"));
  renderFlow([["Context vector", `${architecture.embedding_dimension} values`], ["Linear head", "matrix multiply"], ["Raw logits", `${architecture.vocabulary_size.toLocaleString()} scores`]]);
  stageEvidence.append(makeElement("div", "projection", `${architecture.embedding_dimension} values → ${architecture.vocabulary_size.toLocaleString()} raw scores`));
  if (step.projection_input_norm != null) {
    stageEvidence.append(makeElement("p", "evidence-note", `Vector magnitude after final normalization: ${Number(step.projection_input_norm).toFixed(2)}. Raw scores below are before repetition and sampling adjustments.`));
  }
  const list = makeElement("div", "score-list", "");
  step.candidates.forEach((candidate) => {
    list.append(makeElement("div", "score-row", `${visibleToken(candidate.token)} · ID ${candidate.token_id}  ${Number(candidate.raw_logit).toFixed(2)}`));
  });
  stageEvidence.append(list);
}

function renderStageEvidence(event, result) {
  stageEvidence.replaceChildren();
  if (event.kind === "prompt") {
    const token = result.prompt_tokens[event.promptIndex];
    if (event.stage === 1) {
      renderEmbeddingEvidence(token.embedding_preview, token.embedding_norm, `Prompt token ${event.promptIndex + 1} · ID ${token.id}`, token.id);
      return;
    }
    if (event.stage > 1) {
      stageEvidence.append(makeElement("p", "evidence-note", "Inspect prompt only computes token IDs and embeddings. Click Generate to run the Transformer and see later measured stages."));
      return;
    }
    stageEvidence.append(makeElement("div", "evidence-label", "Actual prompt tokenization"));
    renderFlow([["Text piece", visibleToken(token.text || token.piece)], ["BPE vocabulary", "lookup"], ["Token ID", String(token.id)]]);
    stageEvidence.append(makeElement("p", "evidence-note", `Position ${token.position} in the input sequence.${token.position === 0 ? " This is the BOS (beginning-of-sequence) token." : ""}`));
    return;
  }
  const step = result.steps[event.tokenIndex];
  if (event.stage === 0) {
    stageEvidence.append(makeElement("div", "evidence-label", "Context IDs sent to the model"));
    stageEvidence.append(makeElement("p", "evidence-note", `${step.context_token_count} IDs, including the prompt and earlier generated tokens.`));
    renderFlow([["Prompt + output so far", `${step.context_token_count} pieces`], ["BPE vocabulary", "IDs"], ["Model input", `[1, ${step.context_token_count}]`]]);
    stageEvidence.append(makeElement("div", "id-sequence", step.context_token_ids.slice(-24).join(" · ")));
    if (step.context_token_ids.length > 24) {
      stageEvidence.append(makeElement("p", "evidence-note", "Showing the last 24 IDs of this context."));
    }
  } else if (event.stage === 1) {
    renderEmbeddingEvidence(step.embedding_preview, step.embedding_norm, "Last context token · token + position vector", step.context_token_ids.at(-1));
  } else if (event.stage === 2) {
    renderLayerEvidence(step);
  } else if (event.stage === 3) {
    renderScoreEvidence(step);
  } else if (event.stage === 4) {
    stageEvidence.append(makeElement("div", "evidence-label", "Measured next-token distribution"));
    stageEvidence.append(makeElement("p", "evidence-note", "The five highest chances appear below. With sampling, the selected token can be outside the top five."));
    renderFlow([["Adjusted scores", "penalty + filters"], ["Softmax", "normalize"], ["Next token", "probabilities"]]);
    step.candidates.forEach((candidate) => {
      const row = makeElement("div", "probability evidence-probability", "");
      const track = makeElement("span", "probability-track", "");
      const fill = makeElement("span", "probability-fill", "");
      fill.style.width = `${Math.max(candidate.probability * 100, .7)}%`;
      track.append(fill);
      row.append(makeElement("span", "probability-token", visibleToken(candidate.token)), track, makeElement("span", "probability-value", percent(candidate.probability)));
      stageEvidence.append(row);
    });
  } else {
    stageEvidence.append(makeElement("div", "evidence-label", "Selected by the decoder"));
    renderFlow([["Distribution", "next-token chances"], ["Selected ID", String(step.selected_token_id)], ["Decoded piece", visibleToken(step.selected_token)]]);
    stageEvidence.append(makeElement("div", "chosen-evidence", `${visibleToken(step.selected_token)} · ID ${step.selected_token_id}`));
    stageEvidence.append(makeElement("p", "evidence-note", `Chance under these settings: ${percent(step.selected_probability)}.`));
  }
}

function renderTokens(result, promptCount, generatedCount, activePromptIndex) {
  tokenStream.replaceChildren();
  promptTextArray.textContent = JSON.stringify(result.prompt_tokens.map((token) => token.text || token.piece));
  promptIdArray.textContent = JSON.stringify(result.prompt_tokens.map((token) => token.id));
  generatedTextArray.textContent = JSON.stringify(result.steps?.slice(0, generatedCount).map((step) => step.selected_token) || []);
  generatedIdArray.textContent = JSON.stringify(result.steps?.slice(0, generatedCount).map((step) => step.selected_token_id) || []);
  result.prompt_tokens.forEach((token, index) => {
    const state = index === activePromptIndex ? " active-token" : index < promptCount ? " seen" : "";
    const chip = makeElement("button", `token-chip${state}`, "");
    chip.type = "button";
    chip.title = `Prompt piece ${index + 1}: ID ${token.id}, position ${token.position}. Click to inspect this token.`;
    chip.append(
      makeElement("span", "token-position", String(token.position).padStart(2, "0")),
      makeElement("span", "token-piece", visibleToken(token.text || token.piece)),
      makeElement("span", "token-id", `ID ${token.id}`),
    );
    chip.addEventListener("click", () => {
      pauseTrace();
      moveTo(index);
    });
    tokenStream.append(chip);
  });
  if (generatedCount) tokenStream.append(makeElement("div", "token-divider", "GENERATED NEXT"));
  result.steps?.slice(0, generatedCount).forEach((step, index) => {
    const chip = makeElement("div", `token-chip generated${index === generatedCount - 1 ? " active-token" : ""}`, "");
    chip.title = `Generated token ID ${step.selected_token_id}`;
    chip.append(
      makeElement("span", "token-position", String(index + 1).padStart(2, "0")),
      makeElement("span", "token-piece", visibleToken(step.selected_token)),
      makeElement("span", "token-id", `ID ${step.selected_token_id}`),
    );
    tokenStream.append(chip);
  });
  promptTokenCount.textContent = `${result.prompt_tokens.length} tokens`;
  if (generatedCount) tokenStream.scrollTop = tokenStream.scrollHeight;
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
  activateStage(event.kind === "prompt" ? 0 : event.stage, event.kind === "generation");
  renderStageEvidence(event, result);
  renderTokens(result, promptCount, generatedCount, event.kind === "prompt" ? event.promptIndex : -1);
  renderAnswer(result, generatedCount, atEnd);
  renderLog(events, result, cursor);
  stageToken.textContent = event.kind === "generation" && event.stage === 5
    ? `“${visibleToken(result.steps[event.tokenIndex].selected_token)}”`
    : "Added to the reply";
  stepCounter.textContent = event.kind === "prompt"
    ? `Prompt token ${promptCount} of ${result.prompt_tokens.length} · event ${cursor + 1}/${events.length}`
    : `Prediction ${event.tokenIndex + 1}/${result.steps.length} · step ${event.stage + 1}/6 · event ${cursor + 1}/${events.length}`;
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
    traceTimer = setTimeout(() => moveTo(index + 1), {
      fast: 170, normal: 650, slow: 1600,
    }[playbackSpeed.value] || 650);
  }
}

function startTrace(result, autoplay) {
  pauseTrace();
  const events = result.prompt_tokens.map((_, promptIndex) => ({ kind: "prompt", promptIndex }));
  result.steps?.forEach((_, tokenIndex) => {
    for (let stage = 0; stage < stages.length; stage += 1) {
      events.push({ kind: "generation", tokenIndex, stage });
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
    activateStage(index);
    if (trace) {
      const event = trace.events[trace.cursor];
      if (event.kind === "prompt") {
        renderStageEvidence({ ...event, stage: index }, trace.result);
      } else {
        renderStageEvidence({ ...event, stage: index }, trace.result);
      }
    }
  });
});

loadStatus();
