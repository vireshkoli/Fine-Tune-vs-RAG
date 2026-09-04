// Unit tests for the Space's rendering logic, run by `node --test`.
//
// The Python suite shells out to this file (tests/test_space.py), so a single
// `make test` covers both halves of the demo: the payload written by
// fvr.ops.space and the rendering that reads it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";

import {
  ARMS_TABLE_HEADERS,
  BUCKET_BLURBS,
  BUCKET_ORDER,
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

const HERE = dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(
  readFileSync(join(HERE, "precomputed", "responses.json"), "utf8"),
);

const syntheticCost = {
  cost: {
    arms: {
      trained: { fixed_usd: 2.0, marginal_usd_per_query: 0.00001 },
      retrieval: { fixed_usd: 0.0, marginal_usd_per_query: 0.00002 },
    },
    crossovers: { "trained|retrieval": 200000 },
  },
};

describe("buckets", () => {
  it("can explain every bucket it offers", () => {
    for (const bucket of BUCKET_ORDER) {
      assert.ok(BUCKET_BLURBS[bucket], `no blurb for ${bucket}`);
    }
  });

  it("offers every bucket present in the payload", () => {
    const present = new Set(payload.items.map((item) => item.bucket));
    for (const bucket of present) {
      assert.ok(BUCKET_ORDER.includes(bucket), `${bucket} is unreachable in the UI`);
    }
  });
});

describe("arms table", () => {
  it("is ordered by accuracy", () => {
    const accuracies = armsTable(payload).map((row) => parseFloat(row[1]));
    assert.deepEqual(accuracies, [...accuracies].sort((a, b) => b - a));
  });

  it("has one cell per header", () => {
    for (const row of armsTable(payload)) {
      assert.equal(row.length, ARMS_TABLE_HEADERS.length);
    }
  });

  it("covers every evaluated arm", () => {
    const named = new Set(armsTable(payload).map((row) => row[0]));
    assert.equal(named.size, payload.arms.length);
  });
});

describe("item picker", () => {
  it("narrows when a bucket is chosen", () => {
    const all = itemChoices(payload, "all");
    const one = itemChoices(payload, "index_only");
    assert.ok(one.length > 0);
    assert.ok(one.length < all.length);
  });

  it("round-trips every id back to an item", () => {
    for (const { id } of itemChoices(payload, "all")) {
      assert.ok(findItem(payload, id), `${id} did not resolve`);
    }
    assert.equal(findItem(payload, "no-such-item"), null);
  });

  it("truncates long stems but keeps the subject", () => {
    for (const { label } of itemChoices(payload, "all")) {
      assert.match(label, /^\[.+\] /);
      assert.ok(label.length <= 92 + 40);
    }
  });
});

describe("answers", () => {
  it("marks a verdict that agrees with the gold label", () => {
    for (const item of payload.items) {
      for (const row of renderAnswers(item, payload)) {
        const predicted = row.answered.split(".")[0];
        const expected = predicted === item.answer_label ? "correct" : "wrong";
        assert.equal(row.verdict, expected, `${item.id} / ${row.arm}`);
      }
    }
  });

  it("shows every arm the payload scored", () => {
    for (const item of payload.items) {
      assert.equal(renderAnswers(item, payload).length, Object.keys(item.answers).length);
    }
  });

  it("presents arms in experimental order, not filename order", () => {
    const expected = payload.arms.map((arm) => arm.name);
    const shown = renderAnswers(payload.items[0], payload).map((row) => row.arm);
    assert.deepEqual(shown, expected);
  });
});

describe("cost", () => {
  it("amortises: cost falls as volume rises", () => {
    const arm = { fixed_usd: 2.0, marginal_usd_per_query: 0.00001 };
    assert.ok(usdPer1k(arm, 1) > usdPer1k(arm, 1e6));
  });

  it("approaches the marginal rate at high volume", () => {
    const arm = { fixed_usd: 2.0, marginal_usd_per_query: 0.00001 };
    assert.ok(Math.abs(usdPer1k(arm, 1e12) - 0.01) < 1e-5);
  });

  it("rejects a non-positive volume", () => {
    assert.throws(() => usdPer1k({ fixed_usd: 1, marginal_usd_per_query: 0 }, 0), /positive/);
  });

  it("actually crosses over", () => {
    assert.equal(cheapestAt(syntheticCost, 100), "retrieval");
    assert.equal(cheapestAt(syntheticCost, 1e8), "trained");
  });

  it("ranks the real arms cheapest-first", () => {
    const rows = costTable(payload, 100000);
    const values = rows.map((row) => parseFloat(row[2].slice(1)));
    assert.deepEqual(values, [...values].sort((a, b) => a - b));
    assert.equal(rows[0][1], cheapestAt(payload, 100000));
  });

  it("covers every evaluated arm", () => {
    const named = new Set(costTable(payload, 100000).map((row) => row[1]));
    const expected = new Set(payload.arms.map((arm) => arm.name));
    assert.deepEqual(named, expected);
  });

  it("orders crossovers by volume", () => {
    const volumes = crossovers(payload).map((c) => c.volume);
    assert.deepEqual(volumes, [...volumes].sort((a, b) => a - b));
  });
});

describe("provenance", () => {
  it("cites the frozen split and the pinned revision", () => {
    const note = provenanceNote(payload.provenance);
    assert.ok(note.includes(payload.provenance.split_sha256.slice(0, 16)));
    assert.ok(note.includes(payload.provenance.model_revision.slice(0, 12)));
    assert.ok(note.includes("Nothing is generated at demo time"));
  });
});
