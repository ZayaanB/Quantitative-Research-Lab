"use strict";

const $ = (id) => document.getElementById(id);
const panels = document.querySelectorAll("main > section");
const palette = ["var(--amber)", "var(--blue)", "var(--gray)", "var(--white)"];
const number = (value, digits = 2) => value == null ? "—" :
  value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
const money = (value, digits = 2) => "$" + number(value, digits);
const percent = (value) => number(value, 1) + "%";
const seconds = (value) => value == null ? "—" : number(value, 4) + " s";
const mean = (rows, fn) => rows.reduce((sum, row) => sum + fn(row), 0) / rows.length;
const mae = (rows, prediction, actual) => mean(rows, row => Math.abs(row[prediction] - row[actual]));
const extent = values => values.reduce(([low, high], value) =>
  [Math.min(low, value), Math.max(high, value)], [Infinity, -Infinity]);

function status(panel, message, error = false) {
  const target = $(panel).querySelector(".status");
  target.textContent = message;
  target.toggleAttribute("data-error", error);
}

function metrics(target, entries) {
  target.replaceChildren(...entries.map(([label, value]) => {
    const cell = document.createElement("div");
    const name = document.createElement("dt");
    const figure = document.createElement("dd");
    name.textContent = label;
    figure.textContent = value;
    cell.append(name, figure);
    return cell;
  }));
}

function panelMetrics(panel, entries) {
  metrics($(panel).querySelector(".metrics"), entries);
}

function line(label, points, field, color = 0, x = (_, index) => index, dashed = false) {
  return { label, color: palette[color], points: points.map((point, index) => [x(point, index), point[field]]), dashed };
}

function band(label, points, low, high, x = (_, index) => index) {
  return { label, color: palette[1], points: points.map((point, index) => [x(point, index), point[low], point[high]]) };
}

const chartLayouts = new Map();
const chartResize = new ResizeObserver(entries => {
  for (const { target } of entries) {
    const layout = chartLayouts.get(target);
    if (layout && target.firstElementChild && target.clientWidth && target.clientWidth !== layout.width) {
      chart(target.id, ...layout.args);
    }
  }
});

function chart(id, description, series, range, options = {}) {
  const target = $(id);
  if (!chartLayouts.has(target)) chartResize.observe(target);
  chartLayouts.set(target, { width: target.clientWidth, args: [description, series, range, options] });
  target.replaceChildren();
  const lines = series.filter(item => item.points.length);
  if (!lines.length) return;
  const namespace = "http://www.w3.org/2000/svg";
  const shape = (tag, attributes, text) => {
    const element = document.createElementNS(namespace, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const width = Math.max(250, chartLayouts.get(target).width);
  const height = Math.max(235, Math.min(360, Math.round(width * .47)));
  const left = 77, right = width - 12, top = 17, bottom = height - 43;
  const svg = shape("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": description });
  const xs = lines.flatMap(item => item.points.map(point => point[0]));
  const ys = lines.flatMap(item => item.points.map(point => point[1]));
  if (range) ys.push(...range.points.flatMap(point => point.slice(1)));
  const [minX, maxX] = options.xDomain ?? extent(xs);
  let [minY, maxY] = options.yDomain ?? extent(ys);
  if (minY === maxY) { minY -= .5; maxY += .5; }
  else if (!options.yDomain) {
    const padding = (maxY - minY) * .05;
    minY -= padding;
    maxY += padding;
  }
  const x = value => left + (right - left) * (maxX === minX ? .5 : (value - minX) / (maxX - minX));
  const y = value => bottom - (bottom - top) * (value - minY) / (maxY - minY);
  const mark = ([xx, yy]) => `${x(xx).toFixed(2)},${y(yy).toFixed(2)}`;
  for (let tick = 0; tick <= 4; tick++) {
    const value = minY + (maxY - minY) * tick / 4;
    svg.append(shape("line", { x1: left, x2: right, y1: y(value), y2: y(value), class: "grid" }),
      shape("text", { x: left - 8, y: y(value) + 4, "text-anchor": "end" },
        options.yLabel === money && Math.abs(value) >= 10000
          ? "$" + Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value)
          : (options.yLabel ?? (v => number(v, 2)))(value)));
  }
  const firstSeries = lines[0].points;
  const xValues = options.ticks
    ? [...new Set([0, Math.floor((firstSeries.length - 1) / 2), firstSeries.length - 1])]
      .map(index => firstSeries[index][0])
    : minX === maxX ? [minX] : [minX, (minX + maxX) / 2, maxX];
  for (const value of xValues) {
    const anchor = minX === maxX ? "middle" : value === minX ? "start" : value === maxX ? "end" : "middle";
    svg.append(shape("line", { x1: x(value), x2: x(value), y1: top, y2: bottom, class: "grid" }),
      shape("text", { x: x(value), y: bottom + 17, "text-anchor": anchor },
      (options.xLabel ?? (v => number(v, 0)))(value)));
  }
  if (options.axis) svg.append(shape("text", {
    x: (left + right) / 2, y: height - 6, "text-anchor": "middle", class: "axis-title",
  }, options.axis));
  svg.append(shape("path", { d: `M${left} ${top} V${bottom} H${right}`, fill: "none", class: "axis" }));
  if (range) {
    const upper = range.points.map(([xx, , high]) => mark([xx, high]));
    const lower = range.points.map(([xx, low]) => mark([xx, low])).reverse();
    svg.append(shape("polygon", { points: [...upper, ...lower].join(" "), fill: range.color, opacity: .19 }));
  }
  for (const item of lines) {
    if (item.points.length > 1 && !item.dots) {
      svg.append(shape("polyline", { points: item.points.map(mark).join(" "), fill: "none",
        stroke: item.color, "stroke-width": 1.65, "vector-effect": "non-scaling-stroke", "stroke-dasharray": item.dashed ? "5 4" : "none" }));
    }
    if (item.dots || item.points.length === 1) {
      for (const [xx, yy] of item.points) {
        svg.append(shape("circle", { cx: x(xx), cy: y(yy), r: 2.8, fill: item.color }));
      }
    }
  }
  if (options.frontier != null) {
    const active = lines[options.frontier];
    const current = active.points.at(-1);
    const previous = active.points.at(-2);
    const [at, value] = current;
    const color = active.color;
    if (previous) {
      svg.append(shape("line", {
        x1: x(previous[0]), x2: x(at), y1: y(previous[1]), y2: y(value),
        stroke: color, "stroke-width": 2.5, pathLength: 1, class: "trace-reveal",
      }));
    }
    svg.append(
      shape("line", { x1: x(at), x2: x(at), y1: top, y2: bottom, class: "playhead" }),
      shape("circle", { cx: x(at), cy: y(value), r: 9, fill: color, class: "frontier-halo" }),
      shape("circle", { cx: x(at), cy: y(value), r: 3.6, fill: color,
        style: `--trace: ${color}`, class: "frontier" }),
    );
  }
  const legend = document.createElement("div");
  legend.className = "legend";
  const seen = new Set();
  for (const item of [...lines, ...(range ? [range] : [])]) {
    if (!item.label || seen.has(item.label)) continue;
    seen.add(item.label);
    const label = document.createElement("span");
    label.style.setProperty("--series", item.color);
    if (item === range) label.classList.add("shaded");
    if (item.dashed) label.classList.add("dashed");
    if (item.dots) label.classList.add("dotted");
    label.textContent = item.label;
    legend.append(label);
  }
  target.append(svg, legend);
  if (options.inspect !== false) {
    addInspector(target, svg, shape, lines, range, options, { x, y, left, right, top, bottom, width });
  }
}

function addInspector(target, svg, shape, lines, range, options, geometry) {
  const { x, y, left, right, top, bottom, width } = geometry;
  const visible = lines.filter(item => item.label && !item.reference);
  const positions = [...new Set(visible.flatMap(item => item.points.map(([xx]) => xx)))].sort((a, b) => a - b);
  if (!positions.length) return;

  const inspect = document.createElement("div");
  inspect.className = "plot-inspect";
  const title = document.createElement("strong");
  const values = document.createElement("dl");
  inspect.append(title, values);
  target.append(inspect);

  const marker = shape("g", { "aria-hidden": "true" });
  svg.append(marker);
  svg.setAttribute("tabindex", "0");
  svg.setAttribute("aria-description", "Use left and right arrow keys to inspect values.");
  let active = positions.length - 1;
  const format = options.inspectValue ?? options.yLabel ?? (value => number(value, 2));

  function hide() {
    if (document.activeElement === svg) return;
    inspect.hidden = true;
    marker.replaceChildren();
  }
  function select(index) {
    active = index;
    const at = positions[index];
    const horizontal = x(at);
    title.textContent = (options.inspectLabel ?? options.xLabel ?? String)(at);
    const rows = [];
    const dots = [];
    for (const item of lines) {
      if (!item.label) continue;
      const point = item.points.find(([xx]) => xx === at)
        ?? (item.reference && item.points.length === 2 && item.points[0][1] === item.points[1][1]
          && at >= item.points[0][0] && at <= item.points[1][0] ? [at, item.points[0][1]] : null);
      if (!point) continue;
      rows.push([item.label, format(point[1])]);
      dots.push(shape("circle", { cx: horizontal, cy: y(point[1]), r: 3.5, fill: item.color, stroke: "#080808" }));
    }
    const bounds = range?.points.find(([xx]) => xx === at);
    if (bounds) rows.push([range.label, `${format(bounds[1])} – ${format(bounds[2])}`]);
    values.replaceChildren(...rows.flatMap(([label, value]) => {
      const name = document.createElement("dt");
      const figure = document.createElement("dd");
      name.textContent = label;
      figure.textContent = value;
      return [name, figure];
    }));
    inspect.hidden = false;
    inspect.style.left = Math.max(0, horizontal < width / 2
      ? Math.min(horizontal + 12, width - inspect.offsetWidth)
      : horizontal - inspect.offsetWidth - 12) + "px";
    marker.replaceChildren(shape("line", { x1: horizontal, x2: horizontal, y1: top, y2: bottom, class: "crosshair" }), ...dots);
  }
  inspect.hidden = true;
  svg.addEventListener("pointermove", event => {
    const bounds = svg.getBoundingClientRect();
    const horizontal = (event.clientX - bounds.left) * width / bounds.width;
    if (horizontal < left - 5 || horizontal > right + 5) return hide();
    select(positions.reduce((nearest, value, index) =>
      Math.abs(x(value) - horizontal) < Math.abs(x(positions[nearest]) - horizontal) ? index : nearest, 0));
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => select(active));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", event => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    select(Math.max(0, Math.min(positions.length - 1, active + (event.key === "ArrowLeft" ? -1 : 1))));
  });
}

const playbackStops = new Map();
function playback(panel, total, render, { id, delay = 450, autoplay = false, initialSpeed = "1" } = {}) {
  const target = id ? $(id) : $(panel).querySelector(".playback");
  playbackStops.get(target)?.();
  target.replaceChildren();
  if (!total) return;
  target.append($("playback-controls").content.cloneNode(true));
  const [play, step] = target.querySelectorAll("button");
  const slider = target.querySelector("input");
  const speed = target.querySelector("select");
  const output = target.querySelector("output");
  speed.value = initialSpeed;
  let timer;
  slider.min = 1;
  slider.max = total;
  function show(count) {
    slider.value = count;
    output.value = `${count} / ${total}`;
    render(count);
  }
  function pause() {
    clearInterval(timer);
    timer = undefined;
    play.textContent = "Play";
  }
  playbackStops.set(target, pause);
  function tick() {
    const next = Math.min(total, Number(slider.value) + 1);
    show(next);
    if (next === total) pause();
  }
  function start() {
    play.textContent = "Pause";
    timer = setInterval(() => {
      if (!target.contains(play) || $(panel).hidden) pause();
      else tick();
    }, delay / Number(speed.value));
  }
  play.onclick = () => {
    if (timer) return pause();
    if (Number(slider.value) === total) show(1);
    start();
  };
  step.onclick = () => { pause(); tick(); };
  speed.onchange = () => { if (timer) { pause(); start(); } };
  slider.oninput = () => { pause(); show(Number(slider.value)); };
  show(autoplay ? 1 : total);
  if (autoplay && total > 1) start();
}

async function get(route) {
  const response = await fetch(route);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}

function formAction(id, panel, action) {
  let running = false;
  $(id).addEventListener("submit", async event => {
    event.preventDefault();
    if (running) return;
    running = true;
    const form = event.currentTarget;
    const controls = [...form.elements];
    const disabled = controls.map(control => control.disabled);
    const params = new URLSearchParams(new FormData(form));
    controls.forEach(control => { control.disabled = true; });
    try {
      await action(params);
    } catch (error) {
      status(panel, error.message, true);
    } finally {
      controls.forEach((control, index) => { control.disabled = disabled[index]; });
      running = false;
    }
  });
}

const kindInput = $("price-form").elements.type;
const pricingInputs = $("price-form").elements;
function validatePricing() {
  const asian = kindInput.value === "asian-call";
  $("steps-field").hidden = !asian;
  pricingInputs.steps.disabled = !asian;
  pricingInputs.paths.setCustomValidity(Number(pricingInputs.paths.value) % 2
    ? "Use an even path count for antithetic sampling." : "");
  pricingInputs.steps.setCustomValidity(asian && Number(pricingInputs.paths.value) * Number(pricingInputs.steps.value) > 2e9
    ? "Paths × monitoring steps must be at most 2,000,000,000. Reduce either value." : "");
}
$("price-form").addEventListener("input", validatePricing);
kindInput.onchange = validatePricing;
validatePricing();

const pricePlay = $("price-play");
const priceStep = $("price-step");
const priceSpeed = $("price-speed");
const pricePosition = $("price-position");
let pricingRun;

function showPricing(run, count) {
  run.shown = count;
  const event = run.checkpoints[count - 1];
  pricePosition.value = Math.max(1, count);
  $("price-count").value = `${count} / ${run.total}`;
  $("checkpoint-fill").style.width = `${count / run.total * 100}%`;
  $("scope-price").value = event ? money(event.estimate, 4) : "—";
  if (!event) return;

  const checkpoints = run.checkpoints.slice(0, count);
  const x = item => item.paths;
  const series = [line("Monte Carlo", checkpoints, "estimate", 0, x)];
  if (event.reference != null) {
    series.push({
      label: "Black–Scholes", color: palette[2], dashed: true, reference: true,
      points: [[0, event.reference], [run.paths, event.reference]],
    });
  }
  chart("convergence", "Option value by simulation path count", series,
    band("95% confidence interval", checkpoints, "low", "high", x),
    { yLabel: money, axis: "Simulation paths", inspectValue: v => money(v, 4), frontier: 0,
      xDomain: [0, run.paths], yDomain: run.yDomain,
      xLabel: v => Intl.NumberFormat("en", { notation: "compact" }).format(v),
      inspectLabel: v => number(v, 0) + " paths" });
  panelMetrics("price", [
    ["MC estimate", money(event.estimate, 4)],
    ...(event.reference == null ? [] : [["Black–Scholes", money(event.reference, 4)],
      ["Absolute difference", money(Math.abs(event.reference - event.estimate), 4)]]),
    ["95% confidence interval", `${money(event.low)} – ${money(event.high)}`],
    ["Engine runtime", seconds(event.engine)], ["Throughput", number(event.throughput) + " M paths/s"],
    ["Paths in checkpoint", number(event.paths, 0)],
  ]);
}

function pausePricing(run) {
  clearTimeout(run.timer);
  run.timer = undefined;
  run.playing = false;
  pricePlay.textContent = run.done && run.shown === run.checkpoints.length ? "Replay" : "Play";
}

function pacePricing(run) {
  if (!run.playing || run.timer || run.shown >= run.checkpoints.length) {
    if (run.done && run.shown === run.checkpoints.length) pausePricing(run);
    return;
  }
  run.timer = setTimeout(() => {
    run.timer = undefined;
    showPricing(run, run.shown + 1);
    pacePricing(run);
  }, 340 / Number(priceSpeed.value));
}

function leaveLiveMode() {
  if (priceSpeed.value === "live") priceSpeed.value = "1";
}

pricePlay.onclick = () => {
  const run = pricingRun;
  if (!run) return;
  if (run.playing) return pausePricing(run);
  leaveLiveMode();
  if (run.shown === run.checkpoints.length) showPricing(run, 1);
  if (run.done && run.checkpoints.length <= 1) return;
  run.playing = true;
  pricePlay.textContent = "Pause";
  pacePricing(run);
};
priceStep.onclick = () => {
  const run = pricingRun;
  if (!run) return;
  pausePricing(run);
  leaveLiveMode();
  const next = run.shown === run.checkpoints.length && run.done ? 1 : run.shown + 1;
  showPricing(run, Math.min(next, run.checkpoints.length));
};
pricePosition.oninput = () => {
  if (!pricingRun) return;
  pausePricing(pricingRun);
  leaveLiveMode();
  showPricing(pricingRun, Number(pricePosition.value));
};
priceSpeed.onchange = () => {
  const run = pricingRun;
  if (!run) return;
  pausePricing(run);
  if (priceSpeed.value === "live") showPricing(run, run.checkpoints.length);
  else if (run.checkpoints.length) pricePlay.onclick();
  else if (!run.done) {
    run.playing = true;
    pricePlay.textContent = "Pause";
  }
};

function showAsianPaths(samples, strike, presentationSpeed) {
  const sample = samples[0];
  const lastStep = sample.path.length - 1;
  const values = samples.flatMap(item => item.path).concat(strike);
  const [min, max] = extent(values);
  const pad = Math.max((max - min) * .06, .01);
  playback("price", lastStep + 1, count => {
    const step = count - 1;
    const monitored = sample.path.slice(1, count);
    const average = monitored.length ? mean(monitored, value => value) : null;
    function trace(item, label, color) {
      // Bound SVG work for long monitoring grids while retaining the current endpoint.
      const stride = Math.max(1, Math.ceil(count / 600));
      const points = [];
      for (let index = 0; index < count - 1; index += stride) points.push([index, item.path[index]]);
      points.push([count - 1, item.path[count - 1]]);
      return { label, color, points };
    }
    chart("samples", "Illustrative Asian option paths", [
      trace(sample, "Highlighted path", palette[0]),
      ...samples.slice(1).map(path => trace(path, "", "#476986")),
      ...(average == null ? [] : [{ label: "Running average", color: palette[1], dashed: true,
        reference: true, points: [[0, average], [step, average]] }]),
      { label: "Strike", color: palette[3], dashed: true, reference: true, points: [[0, strike], [lastStep, strike]] },
    ], null, { yLabel: money, axis: "Monitoring step", xDomain: [0, lastStep],
      yDomain: [min - pad, max + pad], frontier: 0 });
    metrics($("greeks"), [
      ["Monitoring step", `${step} / ${lastStep}`],
      ["Running average", average == null ? "—" : money(average)],
      ["Strike", money(strike)],
      ["Final payoff", step === lastStep ? money(sample.payoff) : "At maturity"],
    ]);
  }, { id: "path-playback", delay: 85, autoplay: presentationSpeed !== "live",
    initialSpeed: presentationSpeed === "live" ? "1" : presentationSpeed });
}

formAction("price-form", "price", async params => {
  params.set("vol", Number(params.get("vol")) / 100);
  params.set("rate", Number(params.get("rate")) / 100);
  const asian = params.get("type") === "asian-call";
  if (pricingRun) pausePricing(pricingRun);
  const run = { checkpoints: [], shown: 0, paths: Number(params.get("paths")), done: false,
    total: Number(params.get("checkpoints")), playing: priceSpeed.value !== "live" };
  pricingRun = run;
  pricePlay.disabled = priceStep.disabled = pricePosition.disabled = true;
  pricePosition.max = 1;
  pricePlay.textContent = run.playing ? "Pause" : "Replay";
  $("path-panel").hidden = !asian;
  $("risk-heading").textContent = asian ? "Asian call payoff" : "Greeks";
  $("risk-detail").textContent = asian
    ? "The highlighted path’s monitored average determines its payoff."
    : "European sensitivities from five native revaluations using shared random numbers.";
  metrics($("greeks"), []);
  playbackStops.get($("path-playback"))?.();
  for (const id of ["convergence", "samples", "path-playback"]) $(id).replaceChildren();
  panelMetrics("price", []);
  showPricing(run, 0);
  status("price", "Running native Monte Carlo…");

  function receive(event) {
    if (event.error) throw new Error(event.error);
    if (event.wall !== undefined) run.wall = event.wall;
    if (event.samples) {
      showAsianPaths(event.samples, Number(params.get("strike")), priceSpeed.value);
      return;
    }
    if (event.greeks) {
      metrics($("greeks"), Object.entries(event.greeks).map(([key, value]) =>
        [key === "Vega" ? "Vega / 1 vol point" : key,
          key === "Greeks runtime" ? seconds(value) : number(value, 5)]));
      return;
    }
    const [low, high] = event.interval;
    run.checkpoints.push({ ...event, low, high });
    const available = run.checkpoints.length;
    pricePosition.max = available;
    pricePlay.disabled = priceStep.disabled = pricePosition.disabled = false;
    if (priceSpeed.value === "live" || run.shown === 0) showPricing(run, available);
    if (priceSpeed.value !== "live") pacePricing(run);
    status("price", `Computed ${available} of ${run.total} checkpoints · ${seconds(run.wall)} wall time`);
  }

  let reader;
  try {
    const response = await fetch("/api/price?" + params);
    if (!response.ok) throw new Error("Pricing request failed.");
    reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffered = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunks = (buffered + value).split("\n");
      buffered = chunks.pop();
      for (const chunk of chunks) if (chunk) receive(JSON.parse(chunk));
    }
    if (buffered) receive(JSON.parse(buffered));
    if (run.checkpoints.length !== run.total) throw new Error("Pricing stream ended before the last checkpoint.");
    const bounds = run.checkpoints.flatMap(point => [point.low, point.high, point.reference ?? point.estimate]);
    const low = Math.min(...bounds), high = Math.max(...bounds);
    const pad = Math.max((high - low) * .06, .0005);
    run.yDomain = [low - pad, high + pad];
    status("price", `Compute complete · ${seconds(run.wall)} wall time`);
  } catch (error) {
    run.failed = true;
    throw error;
  } finally {
    run.done = true;
    if (run.shown) showPricing(run, run.shown);
    if (run.failed || priceSpeed.value === "live") pausePricing(run);
    else pacePricing(run);
    await reader?.cancel();
  }
});

formAction("scenario-form", "scenarios", async params => {
  status("scenarios", "Running the historical GBM forecast…");
  const data = await get("/api/scenarios?" + params);
  const x = item => item.step;
  const end = data.fan.at(-1);
  panelMetrics("scenarios", [["Median at horizon", money(end.median)],
    ["95% model range", `${money(end.low)} – ${money(end.high)}`],
    ["Chance above today's price", percent(data.probability)], ["Annual volatility", percent(data.volatility)],
    ["Annual drift", percent(data.drift)], ["Compute wall time", seconds(data.wall)]]);
  const fanBounds = data.fan.flatMap(row => [row.low, row.high]);
  const fanMin = Math.min(...fanBounds), fanMax = Math.max(...fanBounds);
  const fanPad = Math.max((fanMax - fanMin) * .06, .01);
  playback("scenarios", data.fan.length, count => {
    const frames = data.fan.slice(0, count);
    chart("fan", "GBM median and model range by future session",
      [line("Median", frames, "median", 0, x)], band("95% model range", frames, "low", "high", x),
      { yLabel: money, axis: "Trading sessions ahead", xLabel: v => number(v, 0),
        xDomain: [0, end.step], yDomain: [fanMin - fanPad, fanMax + fanPad], frontier: 0 });
  }, { id: "fan-playback", delay: 225 });
  status("scenarios", `${data.synthetic ? "Synthetic example" : "Local history"} · ${data.source}` +
    (data.rows.length ? "" : " · No backtest origins fit this lookback and horizon."));
  $("backtest").replaceChildren();
  metrics($("backtest-metrics"), []);
  playback("scenarios", data.rows.length, count => {
    const rows = data.rows.slice(0, count);
    chart("backtest", "Stock backtest predictions, baseline, and actual prices",
      [line("GBM", rows, "forecast_price"), line("Latest price baseline", rows, "latest_price_forecast", 2, undefined, true),
        line("Actual", rows, "actual_price", 3)], band("95% model range", rows, "lower_95", "upper_95"),
      { yLabel: money, axis: "Target date", frontier: 2, xDomain: [0, data.rows.length - 1],
        xLabel: v => data.rows[Math.round(v)].date.slice(5), inspectLabel: v => data.rows[v].date });
    metrics($("backtest-metrics"), [["GBM MAE", money(mae(rows, "forecast_price", "actual_price"))],
      ["Baseline MAE", money(mae(rows, "latest_price_forecast", "actual_price"))],
      ["Coverage", percent(100 * mean(rows, r => r.actual_price >= r.lower_95 && r.actual_price <= r.upper_95))]]);
  }, { id: "backtest-playback" });
});

function choices(id, labels, render) {
  const select = $(id);
  select.replaceChildren(...labels.map((label, index) => new Option(label, index)));
  select.onchange = () => render(Number(select.value));
  render(0);
}

async function stock() {
  status("stock", "Loading local prediction results…");
  const data = await get("/api/stock");
  status("stock", data.source);
  const probability = data.mode === "probability";
  $("calibration-panel").hidden = !probability;
  choices("stock-series", data.series.map(item => item.label), index => {
    const points = data.series[index].points;
    playback("stock", points.length, count => {
      const shown = points.slice(0, count);
      const recent = shown.slice(-45);
      const yLabel = probability ? v => percent(100 * v) : money;
      const label = probability ? "Calibrated probability" : "Ridge forecast";
      const actual = line("Actual", recent, "actual", 3);
      actual.dots = probability;
      chart("stock-plot", "Stock predictions and actual outcomes across the latest 45 revealed dates",
        [line(label, recent, "prediction"), line("Baseline", recent, "baseline", 2, undefined, true), actual],
        null, { yLabel, axis: "Target date", frontier: 2, xLabel: v => recent[Math.round(v)].date.slice(5), inspectLabel: v => recent[v].date,
          yDomain: probability ? [0, 1] : undefined });
      const score = field => probability ? number(mean(shown, row => (row[field] - row.actual) ** 2), 4)
        : money(mae(shown, field, "actual"));
      panelMetrics("stock", [[probability ? "Calibrated Brier score" : "Ridge MAE", score("prediction")],
        [probability ? "Baseline Brier score" : "Baseline MAE", score("baseline")],
        ...(probability ? [["Raw Brier score", score("raw")]] : []),
        ["Revealed through", shown.at(-1).date], ["Report compute time", seconds(data.runtime)]]);
      if (!probability) return;
      const buckets = Array.from({ length: 10 }, () => []);
      for (const row of shown) buckets[Math.min(9, Math.floor(row.prediction * 10))].push(row);
      chart("calibration", "Calibrated probabilities versus observed up frequencies",
        [{ label: "Perfect calibration", color: palette[2], dashed: true, reference: true, points: [[0, 0], [1, 1]] },
         { label: "Observed bins", color: palette[0], dots: true,
           points: buckets.filter(rows => rows.length).map(rows => [mean(rows, r => r.prediction), mean(rows, r => r.actual)]) }],
        null, { xDomain: [0, 1], yDomain: [0, 1], xLabel: yLabel, yLabel, axis: "Predicted up probability" });
    });
  });
}

async function energy() {
  status("energy", "Loading local demand results…");
  const data = await get("/api/energy");
  status("energy", data.source);
  choices("energy-day", data.days.map(day => day.date), index => {
    const rows = data.days[index].points;
    const x = row => row.hour;
    const hasRange = rows.every(row => row.low != null && row.high != null);
    playback("energy", rows.length, count => {
      const shown = rows.slice(0, count);
      chart("energy-plot", "Ontario demand forecasts with actual load revealed hour by hour",
        [line(data.model, rows, "prediction", 0, x), line("Daily baseline", rows, "baseline", 2, x, true),
          line("Actual demand", shown, "actual", 3, x)],
        hasRange ? band("Calibrated interval", rows, "low", "high", x) : null,
        { yLabel: v => number(v, 0) + " MW", xLabel: v => number(v, 0), axis: "Forecast lead hours", ticks: true, frontier: 2 });
      const mw = value => number(value, 0) + " MW";
      const errors = shown.map(row => ({
        hour: row.hour, modelError: row.prediction - row.actual, baselineError: row.baseline - row.actual,
      }));
      chart("energy-error", "Forecast minus actual electricity demand by lead hour", [
        { label: "Zero error", color: palette[3], dashed: true, reference: true, points: [[1, 0], [24, 0]] },
        line(data.model, errors, "modelError", 0, x),
        line("Daily baseline", errors, "baselineError", 2, x, true),
      ], null, { xDomain: [1, 24], yLabel: mw, axis: "Forecast lead hours" });
      const peak = (points, field) => Math.max(...points.map(row => row[field]));
      panelMetrics("energy", [[data.model + " MAE", mw(mae(shown, "prediction", "actual"))],
        ["Daily baseline MAE", mw(mae(shown, "baseline", "actual"))], ["Latest actual demand", mw(shown.at(-1).actual)],
        ["Peak magnitude error", count === rows.length ? mw(Math.abs(peak(rows, "prediction") - peak(rows, "actual"))) : "After 24 hours"],
        ...(hasRange ? [["Interval coverage", percent(100 * mean(shown, row => row.actual >= row.low && row.actual <= row.high))]] : []),
        ["Report compute time", seconds(data.runtime)]]);
    });
  });
}

const loaders = { scenarios: () => $("scenario-form").requestSubmit(), stock, energy };
const opened = new Set(["price"]);
document.querySelectorAll("nav button").forEach(button => {
  button.onclick = async () => {
    const tab = button.dataset.tab;
    for (const panel of panels) panel.hidden = panel.id !== tab;
    document.querySelectorAll("nav button").forEach(item => item.removeAttribute("aria-current"));
    button.setAttribute("aria-current", "page");
    $("workspace-name").textContent = button.textContent.slice(1).trim();
    if (opened.has(tab)) return;
    opened.add(tab);
    try {
      await loaders[tab]();
    } catch (error) {
      opened.delete(tab);
      status(tab, error.message, true);
    }
  };
});
$("price-form").requestSubmit();
