// DOM wiring. All rendering decisions live in logic.js, which is unit-tested;
// this file only puts the results on the page.

import {
  ARMS_TABLE_HEADERS,
  BUCKET_BLURBS,
  BUCKET_ORDER,
  COST_TABLE_HEADERS,
  armsTable,
  cheapestAt,
  costTable,
  crossovers,
  findItem,
  itemChoices,
  provenanceNote,
  renderAnswers,
  usdPer1k,
} from "./logic.js";

const $ = (id) => document.getElementById(id);
const NUMERIC_FROM = { arms: 1, cost: 2 };

function table(headers, rows, numericFrom = 99) {
  const head = headers
    .map((h, i) => `<th class="${i >= numericFrom ? "num" : ""}">${h}</th>`)
    .join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${row
          .map((cell, i) => `<td class="${i >= numericFrom ? "num" : ""}">${cell}</td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const escapeHtml = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );

function renderItem(payload, itemId) {
  const item = findItem(payload, itemId);
  if (!item) return;

  const options = item.options
    .map((option, index) => {
      const gold = index === item.answer_idx;
      return `<li><strong>${"ABCD"[index]}.</strong> ${escapeHtml(option)}${
        gold ? ' <span class="gold">← correct</span>' : ""
      }</li>`;
    })
    .join("");

  $("question").innerHTML =
    `<h3>${escapeHtml(item.question)}</h3>` +
    `<p class="muted">${escapeHtml(item.subject)}</p>` +
    `<ul>${options}</ul>` +
    `<div class="blurb">${BUCKET_BLURBS[item.bucket] ?? ""}</div>`;

  const rows = renderAnswers(item, payload).map((row) => [
    `<span class="mono">${row.arm}</span>`,
    escapeHtml(row.answered),
    `<span class="pill ${row.verdict}">${row.verdict}</span>`,
    row.confidence,
    row.promptTokens,
  ]);
  $("answers").innerHTML = table(
    ["Arm", "Answered", "Verdict", "Confidence", "Prompt tokens"],
    rows,
    3,
  );
}

function renderCost(payload, exponent) {
  const volume = Math.round(10 ** exponent);
  $("volume-label").textContent = `${volume.toLocaleString()} lifetime queries`;
  $("cost-table").innerHTML = table(
    COST_TABLE_HEADERS,
    costTable(payload, volume),
    NUMERIC_FROM.cost,
  );

  const winner = cheapestAt(payload, volume);
  const lines = crossovers(payload)
    .map(
      (c) =>
        `<li><span class="mono">${c.a}</span> becomes cheaper than ` +
        `<span class="mono">${c.b}</span> at <strong>${c.volume.toLocaleString()}</strong> queries</li>`,
    )
    .join("");
  const rate = payload.cost.rate_card;
  $("cost-summary").innerHTML =
    `<p>At <strong>${volume.toLocaleString()}</strong> lifetime queries the cheapest arm is ` +
    `<span class="mono">${winner}</span>, at ` +
    `$${usdPer1k(payload.cost.arms[winner], volume).toFixed(4)} per 1,000 queries.</p>` +
    `<p>Crossover volumes — where one arm overtakes another:</p><ul>${lines}</ul>` +
    `<p class="muted">Rates: ${rate.gpu_name} at $${rate.gpu_usd_per_hour.toFixed(2)}/GPU-hour ` +
    `(<a href="${rate.source_url}">source</a>, retrieved ${rate.retrieved}). ` +
    `Local GPU-seconds are never mixed with hosted API pricing.</p>`;
}

async function main() {
  const payload = await fetch("precomputed/responses.json").then((r) => r.json());

  $("arms-table").innerHTML = table(ARMS_TABLE_HEADERS, armsTable(payload), NUMERIC_FROM.arms);
  $("provenance").innerHTML = provenanceNote(payload.provenance);

  const bucket = $("bucket");
  bucket.innerHTML =
    `<option value="all">all buckets</option>` +
    BUCKET_ORDER.map((b) => `<option value="${b}">${b}</option>`).join("");
  bucket.value = BUCKET_ORDER[0];

  const item = $("item");
  const fillItems = () => {
    const choices = itemChoices(payload, bucket.value);
    item.innerHTML = choices
      .map((c) => `<option value="${c.id}">${escapeHtml(c.label)}</option>`)
      .join("");
    if (choices.length) renderItem(payload, choices[0].id);
  };
  bucket.addEventListener("change", fillItems);
  item.addEventListener("change", () => renderItem(payload, item.value));
  fillItems();

  const slider = $("volume");
  slider.addEventListener("input", () => renderCost(payload, parseFloat(slider.value)));
  renderCost(payload, parseFloat(slider.value));
}

main().catch((error) => {
  document.getElementById("arms-table").innerHTML =
    `<p class="warning">Could not load the precomputed results: ${escapeHtml(error.message)}</p>`;
});
