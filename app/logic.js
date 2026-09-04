// Pure rendering logic for the demo Space.
//
// Deliberately free of any DOM access so it can be unit-tested with `node --test`
// (see logic.test.js, which the Python suite shells out to). ui.js is a thin
// shell over these functions.
//
// Nothing here computes a model output: every answer is read from
// precomputed/responses.json, written from the committed benchmark run JSONs.

export const OPTION_LABELS = ["A", "B", "C", "D"];

// What each disagreement bucket demonstrates. The demo is stratified over these
// rather than sampled at random, because a random draw is mostly items where
// every arm agrees — which shows nothing.
export const BUCKET_BLURBS = {
  index_only:
    "<strong>Retrieval got it, the fine-tune did not.</strong> The same explanation is in " +
    "the index and in the weights — only the index could use it.",
  weights_only:
    "<strong>The fine-tune got it, retrieval did not.</strong> Either the retriever missed " +
    "the passage, or the answer needed knowledge no single passage carries.",
  both_fixed_it:
    "<strong>Both approaches rescued an item the base model missed.</strong> The kind of " +
    "question where adding domain information of any form helps.",
  both_broke_it:
    "<strong>The base model was right and both approaches broke it.</strong> Retrieved " +
    "context and fine-tuning can each talk a correct model out of an answer.",
  everyone_right:
    "<strong>Every arm answered correctly.</strong> The base model already knew this.",
  everyone_wrong:
    "<strong>No arm answered correctly.</strong> The ceiling on this benchmark is not " +
    "information access.",
};

export const BUCKET_ORDER = [
  "index_only",
  "weights_only",
  "both_fixed_it",
  "both_broke_it",
  "everyone_right",
  "everyone_wrong",
];

const percent = (value, digits = 1) => `${(100 * value).toFixed(digits)}%`;

/** The results table, ordered by accuracy. */
export function armsTable(payload) {
  return [...payload.arms]
    .sort((a, b) => b.accuracy - a.accuracy)
    .map((arm) => [
      arm.name,
      percent(arm.accuracy),
      `[${percent(arm.ci_95[0])}, ${percent(arm.ci_95[1])}]`,
      `${Math.round(arm.p50_ms)} ms`,
      `${Math.round(arm.p95_ms)} ms`,
      `${Math.round(arm.prompt_tokens)}`,
      arm.headline ? "headline" : "diagnostic",
    ]);
}

export const ARMS_TABLE_HEADERS = [
  "Arm",
  "Accuracy",
  "95% CI",
  "p50",
  "p95",
  "Prompt tokens",
  "Role",
];

/** `{label, id}` pairs for the item picker, filtered to one bucket. */
export function itemChoices(payload, bucket) {
  return payload.items
    .filter((item) => bucket === "all" || item.bucket === bucket)
    .map((item) => {
      const stem = item.question.trim().replace(/\s+/g, " ");
      const label = stem.length <= 90 ? stem : `${stem.slice(0, 87)}…`;
      return { label: `[${item.subject}] ${label}`, id: item.id };
    });
}

export function findItem(payload, itemId) {
  return payload.items.find((item) => item.id === itemId) ?? null;
}

/** Each arm's precomputed answer for one item, in experimental order. */
export function renderAnswers(item, payload) {
  const rows = [];
  for (const arm of payload.arms) {
    const answer = item.answers[arm.name];
    if (!answer) continue;
    const correct = answer.predicted_idx === item.answer_idx;
    rows.push({
      arm: arm.name,
      answered:
        answer.predicted_idx === null
          ? "—"
          : `${answer.predicted_label}. ${item.options[answer.predicted_idx]}`,
      correct,
      verdict: correct ? "correct" : "wrong",
      confidence: answer.confidence === null ? "—" : percent(answer.confidence, 0),
      promptTokens: answer.prompt_tokens ?? "—",
    });
  }
  return rows;
}

/**
 * `fixed / N + marginal`, in dollars per 1,000 queries.
 *
 * The formula is deliberately here in full rather than interpolated from a
 * sampled curve: the point of the cost section is that the shape is knowable,
 * not that a chart says so.
 */
export function usdPer1k(armCost, volume) {
  if (!(volume > 0)) throw new Error("volume must be positive");
  return 1000 * (armCost.fixed_usd / volume + armCost.marginal_usd_per_query);
}

/** Cost per 1,000 queries for every arm at one lifetime volume, cheapest first. */
export function costTable(payload, volume) {
  const arms = payload.cost.arms;
  return Object.keys(arms)
    .sort((a, b) => usdPer1k(arms[a], volume) - usdPer1k(arms[b], volume))
    .map((name, index) => [
      `${index + 1}`,
      name,
      `$${usdPer1k(arms[name], volume).toFixed(4)}`,
      `$${arms[name].fixed_usd.toFixed(2)}`,
      `$${(1000 * arms[name].marginal_usd_per_query).toFixed(4)}`,
    ]);
}

export const COST_TABLE_HEADERS = [
  "#",
  "Arm",
  "$ / 1k queries",
  "Fixed cost (one-off)",
  "$ / 1k marginal",
];

export function cheapestAt(payload, volume) {
  const arms = payload.cost.arms;
  return Object.keys(arms).reduce((best, name) =>
    usdPer1k(arms[name], volume) < usdPer1k(arms[best], volume) ? name : best,
  );
}

export function crossovers(payload) {
  return Object.entries(payload.cost.crossovers)
    .map(([pair, volume]) => {
      const [a, b] = pair.split("|");
      return { a, b, volume };
    })
    .sort((x, y) => x.volume - y.volume);
}

export function provenanceNote(p) {
  return (
    `Every answer shown was produced by a real evaluated run over a frozen ` +
    `${p.n_test_items.toLocaleString()}-item test set ` +
    `(SHA-256 <code>${p.split_sha256.slice(0, 16)}…</code>), using ` +
    `<a href="https://huggingface.co/${p.model}">${p.model}</a> ` +
    `at revision <code>${p.model_revision.slice(0, 12)}</code>, seed ${p.seed}, ` +
    `from commit <code>${p.git_sha.slice(0, 12)}</code>. ` +
    `Nothing is generated at demo time.`
  );
}
