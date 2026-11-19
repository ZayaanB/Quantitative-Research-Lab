"use strict";

const $ = (id) => document.getElementById(id);
const panels = document.querySelectorAll("main > section");
const palette = ["var(--amber)", "var(--blue)", "var(--gray)", "var(--white)"];
const number = (value, digits = 2) => value == null ? "—" :
  value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
const money = (value) => "$" + number(value);
const percent = (value) => number(value, 1) + "%";
const seconds = (value) => value == null ? "—" : number(value, 4) + " s";
const mean = (rows, fn) => rows.reduce((sum, row) => sum + fn(row), 0) / rows.length;
const mae = (rows, prediction, actual) => mean(rows, row => Math.abs(row[prediction] - row[actual]));

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
  const height = Math.max(200, Math.min(350, Math.round(width * .47)));
  const left = 68, right = width - 10, top = 12, bottom = height - 28;
  const svg = shape("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": description });
  const xs = lines.flatMap(item => item.points.map(point => point[0]));
  const ys = lines.flatMap(item => item.points.map(point => point[1]));
  if (range) ys.push(...range.points.flatMap(point => point.slice(1)));
  const [minX, maxX] = options.xDomain ?? [Math.min(...xs), Math.max(...xs)];
  let [minY, maxY] = options.yDomain ?? [Math.min(...ys), Math.max(...ys)];
  if (minY === maxY) { minY -= .5; maxY += .5; }
  const x = value => left + (right - left) * (maxX === minX ? .5 : (value - minX) / (maxX - minX));
  const y = value => bottom - (bottom - top) * (value - minY) / (maxY - minY);
  const mark = ([xx, yy]) => `${x(xx).toFixed(2)},${y(yy).toFixed(2)}`;
  for (let tick = 0; tick <= 4; tick++) {
    const value = minY + (maxY - minY) * tick / 4;
    svg.append(shape("line", { x1: left, x2: right, y1: y(value), y2: y(value), class: "grid" }),
      shape("text", { x: left - 8, y: y(value) + 4, "text-anchor": "end" },
        (options.yLabel ?? (v => number(v, 2)))(value)));
  }
  const xValues = minX === maxX ? [minX] : [minX, (minX + maxX) / 2, maxX];
  for (const value of xValues) {
    const anchor = minX === maxX ? "middle" : value === minX ? "start" : value === maxX ? "end" : "middle";
    svg.append(shape("line", { x1: x(value), x2: x(value), y1: top, y2: bottom, class: "grid" }),
      shape("text", { x: x(value), y: bottom + 17, "text-anchor": anchor },
      (options.xLabel ?? (v => number(v, 0)))(value)));
  }
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
}

function playback(panel, total, render) {
  const target = $(panel).querySelector(".playback");
  target.replaceChildren();
  if (!total) return;
  target.append($("playback-controls").content.cloneNode(true));
  const [play, step] = target.querySelectorAll("button");
  const slider = target.querySelector("input");
  const speed = target.querySelector("select");
  const output = target.querySelector("output");
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
    }, 450 / Number(speed.value));
  }
  play.onclick = () => {
    if (timer) return pause();
    if (Number(slider.value) === total) show(1);
    start();
  };
  step.onclick = () => { pause(); tick(); };
  speed.onchange = () => { if (timer) { pause(); start(); } };
  slider.oninput = () => { pause(); show(Number(slider.value)); };
  show(total);
}

async function get(route) {
  const response = await fetch(route);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}

function formAction(id, panel, action) {
  $(id).addEventListener("submit", async event => {
    event.preventDefault();
    const form = event.currentTarget;
    const controls = [...form.elements];
    const params = new URLSearchParams(new FormData(form));
    controls.forEach(control => { control.disabled = true; });
    try {
      await action(params);
    } catch (error) {
      status(panel, error.message, true);
    } finally {
      controls.forEach(control => { control.disabled = false; });
    }
  });
}

const kindInput = $("price-form").elements.type;
kindInput.onchange = () => { $("steps-field").hidden = kindInput.value !== "asian-call"; };

formAction("price-form", "price", async params => {
  params.set("vol", Number(params.get("vol")) / 100);
  params.set("rate", Number(params.get("rate")) / 100);
  const asian = params.get("type") === "asian-call";
  $("risk-heading").textContent = asian ? "Asian call payoff" : "Greeks";
  $("risk-detail").textContent = asian
    ? "Illustrative paths use the same model assumptions. The payoff uses each path’s monitored average."
    : "European sensitivities from five native revaluations using shared random numbers.";
  metrics($("greeks"), []);
  $("convergence").replaceChildren();
  $("samples").replaceChildren();
  panelMetrics("price", []);
  status("price", "Running native Monte Carlo…");
  const response = await fetch("/api/price?" + params);
  if (!response.ok) throw new Error("Pricing request failed.");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const checkpoints = [];
  let buffered = "";
  let wall;
  function update(event) {
    if (event.error) throw new Error(event.error);
    if (event.wall !== undefined) wall = event.wall;
    if (event.samples) {
      chart("samples", "Illustrative Asian option paths", event.samples.map((sample, index) => ({
        label: index === 0 ? "Sample paths" : "", color: palette[0],
        points: sample.path.map((value, step) => [step, value]),
      })), null, { yLabel: money });
      const sample = event.samples[0];
      metrics($("greeks"), [["Path 1 average", money(sample.average)], ["Path 1 payoff", money(sample.payoff)]]);
      return;
    }
    if (event.greeks) {
      metrics($("greeks"), Object.entries(event.greeks).map(([key, value]) =>
        [key === "Vega" ? "Vega / 1 vol point" : key,
          key === "Greeks runtime" ? seconds(value) : number(value, 5)]));
      return;
    }
    const [low, high] = event.interval;
    checkpoints.push({ ...event, low, high });
    const x = item => item.paths;
    const series = [line("Monte Carlo", checkpoints, "estimate", 0, x)];
    if (event.reference != null) series.push(line("Black–Scholes", checkpoints, "reference", 2, x, true));
    chart("convergence", "Option value by simulation path count", series,
      band("95% confidence interval", checkpoints, "low", "high", x),
      { yLabel: money, xLabel: v => Intl.NumberFormat("en", { notation: "compact" }).format(v) });
    panelMetrics("price", [
      ["MC estimate", money(event.estimate)],
      ...(event.reference == null ? [] : [["Black–Scholes", money(event.reference)],
        ["Absolute difference", money(Math.abs(event.reference - event.estimate))]]),
      ["95% confidence interval", `${money(low)} – ${money(high)}`],
      ["Engine runtime", seconds(event.engine)], ["Throughput", number(event.throughput) + " M paths/s"],
      ["Paths in checkpoint", number(event.paths, 0)],
    ]);
    status("price", `Completed ${checkpoints.length} of 7 checkpoints · ${seconds(wall)} wall time`);
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunks = (buffered + value).split("\n");
      buffered = chunks.pop();
      for (const chunk of chunks) if (chunk) update(JSON.parse(chunk));
    }
    if (buffered) update(JSON.parse(buffered));
    if (checkpoints.length !== 7) throw new Error("Pricing stream ended before the last checkpoint.");
    status("price", `Finished · ${seconds(wall)} total wall time, including native process startup and revaluations`);
  } finally {
    await reader.cancel();
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
  chart("fan", "GBM median and model range by future session",
    [line("Median", data.fan, "median", 0, x)], band("95% model range", data.fan, "low", "high", x),
    { yLabel: money, xLabel: v => number(v, 0) + "d" });
  status("scenarios", `${data.synthetic ? "Synthetic example" : "Local history"} · ${data.source}` +
    (data.rows.length ? "" : " · No backtest origins fit this lookback and horizon."));
  $("backtest").replaceChildren();
  metrics($("backtest-metrics"), []);
  playback("scenarios", data.rows.length, count => {
    const rows = data.rows.slice(0, count);
    chart("backtest", "Stock backtest predictions, baseline, and actual prices",
      [line("GBM", rows, "forecast_price"), line("Latest price baseline", rows, "latest_price_forecast", 2, undefined, true),
        line("Actual", rows, "actual_price", 3)], band("95% model range", rows, "lower_95", "upper_95"),
      { yLabel: money, xLabel: v => rows[Math.round(v)].date.slice(5) });
    metrics($("backtest-metrics"), [["GBM MAE", money(mae(rows, "forecast_price", "actual_price"))],
      ["Baseline MAE", money(mae(rows, "latest_price_forecast", "actual_price"))],
      ["Coverage", percent(100 * mean(rows, r => r.actual_price >= r.lower_95 && r.actual_price <= r.upper_95))]]);
  });
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
        null, { yLabel, xLabel: v => recent[Math.round(v)].date.slice(5),
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
        [{ label: "Perfect calibration", color: palette[2], dashed: true, points: [[0, 0], [1, 1]] },
         { label: "Observed bins", color: palette[0], dots: true,
           points: buckets.filter(rows => rows.length).map(rows => [mean(rows, r => r.prediction), mean(rows, r => r.actual)]) }],
        null, { xDomain: [0, 1], yDomain: [0, 1], xLabel: yLabel, yLabel });
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
        { yLabel: v => number(v, 0), xLabel: v => "+" + number(v, 0) + "h" });
      const mw = value => number(value, 0) + " MW";
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
