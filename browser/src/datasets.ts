/**
 * datasets.ts — load training text from Hugging Face (Phase 4).
 *
 * The Hugging Face datasets-server exposes public datasets as JSON over HTTP
 * with `Access-Control-Allow-Origin: *` and no API key — so a static site can
 * pull training text straight from the browser:
 *
 *   GET https://datasets-server.huggingface.co/rows
 *       ?dataset=<id>&config=<c>&split=<s>&offset=<o>&length=<n>
 *   -> { rows: [ { row: { <textColumn>: "..." } }, ... ] }
 *
 * This module ships a small curated catalog (all verified to work with the
 * datasets-server and reasonable for a tiny byte-level model) plus a pager that
 * concatenates rows up to a character budget.
 */

export interface HfDataset {
  id: string; // short key for the <select>
  label: string;
  dataset: string; // Hugging Face dataset path
  config: string;
  split: string;
  textColumn: string; // which row field holds the text
  license: string;
  blurb: string;
}

/** Curated, datasets-server-verified datasets — all plain English text. */
export const HF_CATALOG: HfDataset[] = [
  {
    id: "tinystories",
    label: "TinyStories",
    dataset: "roneneldan/TinyStories",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "CDLA-Sharing-1.0",
    blurb: "simple short stories — the best fit for a tiny model",
  },
  {
    id: "tinystories-v2",
    label: "TinyStories V2 (GPT-4)",
    dataset: "roneneldan/TinyStoriesV2-GPT4",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "CDLA-Sharing-1.0",
    blurb: "the GPT-4-regenerated TinyStories — cleaner prose",
  },
  {
    id: "tiny-shakespeare",
    label: "Tiny Shakespeare",
    dataset: "Trelis/tiny-shakespeare",
    config: "default",
    split: "train",
    textColumn: "Text",
    license: "Public domain",
    blurb: "the classic Karpathy demo corpus — 1 MB of the Bard",
  },
  {
    id: "simplewiki",
    label: "Simple English Wikipedia",
    dataset: "wikimedia/wikipedia",
    config: "20231101.simple",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA",
    blurb: "encyclopedia articles in simplified English",
  },
  {
    id: "wikipedia-en",
    label: "Wikipedia (English)",
    dataset: "wikimedia/wikipedia",
    config: "20231101.en",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA",
    blurb: "the full English Wikipedia — diverse but harder",
  },
  {
    id: "wikitext-2",
    label: "WikiText-2",
    dataset: "Salesforce/wikitext",
    config: "wikitext-2-raw-v1",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA-3.0",
    blurb: "the classic language-modeling benchmark corpus",
  },
  {
    id: "wikitext-103",
    label: "WikiText-103",
    dataset: "Salesforce/wikitext",
    config: "wikitext-103-raw-v1",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA-3.0",
    blurb: "the larger WikiText sibling — more variety",
  },
  {
    id: "quotes",
    label: "English quotes",
    dataset: "Abirate/english_quotes",
    config: "default",
    split: "train",
    textColumn: "quote",
    license: "CC-BY-4.0",
    blurb: "short literary quotations",
  },
  {
    id: "imdb",
    label: "IMDB reviews",
    dataset: "stanfordnlp/imdb",
    config: "plain_text",
    split: "train",
    textColumn: "text",
    license: "Non-commercial",
    blurb: "movie reviews — opinionated, varied register",
  },
  {
    id: "ag-news",
    label: "AG News headlines",
    dataset: "fancyzhx/ag_news",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "CC-0",
    blurb: "short news headlines + summaries",
  },
  {
    id: "dolly-15k",
    label: "Dolly 15k (responses)",
    dataset: "databricks/databricks-dolly-15k",
    config: "default",
    split: "train",
    textColumn: "response",
    license: "CC-BY-SA-3.0",
    blurb: "human-written instruction responses",
  },
  {
    id: "pg19",
    label: "PG-19 (Project Gutenberg)",
    dataset: "deepmind/pg19",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "Public domain",
    blurb: "long-form classic books from Project Gutenberg",
  },
  {
    id: "openwebtext-10k",
    label: "OpenWebText (10k)",
    dataset: "stas/openwebtext-10k",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "CC-0",
    blurb: "10k diverse web pages — the GPT-2 pretraining flavour",
  },
  {
    id: "lyrics",
    label: "Song lyrics",
    dataset: "amishshah/song_lyrics",
    config: "default",
    split: "train",
    textColumn: "lyrics",
    license: "various",
    blurb: "lyrics from many artists — short, rhythmic, distinctive",
  },
  {
    id: "github-python",
    label: "GitHub Python code",
    dataset: "codeparrot/github-code-clean",
    config: "Python-all",
    split: "train",
    textColumn: "code",
    license: "various OSS",
    blurb: "Python source — model learns indentation, def, and snake_case",
  },
  {
    id: "recipes",
    label: "Cooking recipes",
    dataset: "corbt/all-recipes",
    config: "default",
    split: "train",
    textColumn: "input",
    license: "Public",
    blurb: "recipe instructions — imperative voice, numbered steps",
  },
  {
    id: "pubmed-abstracts",
    label: "PubMed abstracts",
    dataset: "armanc/scientific_papers",
    config: "pubmed",
    split: "train",
    textColumn: "abstract",
    license: "Apache-2.0",
    blurb: "biomedical abstracts — formal, hedged scientific register",
  },
  {
    id: "poetry",
    label: "English poetry",
    dataset: "merve/poetry",
    config: "default",
    split: "train",
    textColumn: "content",
    license: "CC-BY",
    blurb: "poems — short lines, rhyme, distinctive stanza shape",
  },
];

const SERVER = "https://datasets-server.huggingface.co";

export class HfFetchError extends Error {
  constructor(
    message: string,
    public readonly kind: "auth" | "not-found" | "ratelimit" | "network" | "empty" | "other",
    public readonly status?: number,
  ) {
    super(message);
    this.name = "HfFetchError";
  }
}

function diagnoseStatus(status: number, dataset: string): HfFetchError {
  if (status === 401)
    return new HfFetchError(
      `"${dataset}" requires a Hugging Face token — it's either gated (you have to accept its terms) or hitting the anonymous rate limit. Paste a token below to retry, or pick another dataset.`,
      "auth",
      401,
    );
  if (status === 403)
    return new HfFetchError(
      `"${dataset}" is gated — accept its terms on huggingface.co, create a read-only access token, and paste it below.`,
      "auth",
      403,
    );
  if (status === 404)
    return new HfFetchError(
      `"${dataset}" was not found. Check the path is exactly owner/name and the config/split are valid.`,
      "not-found",
      404,
    );
  if (status === 429)
    return new HfFetchError(
      `Hit the anonymous rate limit — wait a minute or paste an HF token below to keep going.`,
      "ratelimit",
      429,
    );
  return new HfFetchError(`dataset server returned HTTP ${status}`, "other", status);
}

/**
 * Fetch up to ~maxChars of text from a Hugging Face dataset, paging the
 * datasets-server `rows` endpoint. `onProgress` reports characters so far.
 * `token`, if provided, is sent as a Bearer token to unlock gated datasets
 * and lift the anonymous rate limit.
 *
 * Default cap raised to 2 MB — the old 120 KB limit was sized for the original
 * demo corpus, but real training needs real data.
 */
export async function fetchHfText(
  d: HfDataset,
  maxChars = 2_000_000,
  onProgress?: (chars: number) => void,
  token?: string,
): Promise<string> {
  const parts: string[] = [];
  let chars = 0;
  let offset = 0;
  const pageSize = 100;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  while (chars < maxChars && offset < 50_000) {
    const url =
      `${SERVER}/rows?dataset=${encodeURIComponent(d.dataset)}` +
      `&config=${encodeURIComponent(d.config)}` +
      `&split=${encodeURIComponent(d.split)}` +
      `&offset=${offset}&length=${pageSize}`;

    let resp: Response;
    try {
      resp = await fetch(url, { headers });
    } catch (err) {
      throw new HfFetchError(
        `couldn't reach huggingface.co — check your connection`,
        "network",
      );
    }
    if (!resp.ok) throw diagnoseStatus(resp.status, d.dataset);
    const json = (await resp.json()) as {
      rows?: { row: Record<string, unknown> }[];
    };
    const rows = json.rows ?? [];
    if (rows.length === 0) break;

    for (const r of rows) {
      const value = r.row[d.textColumn];
      if (typeof value === "string" && value.trim()) {
        parts.push(value.trim());
        chars += value.length;
      }
    }
    onProgress?.(chars);
    offset += rows.length;
    if (rows.length < pageSize) break;
  }

  if (parts.length === 0)
    throw new HfFetchError(
      `no text was returned — the column "${d.textColumn}" may be wrong, or the split is empty.`,
      "empty",
    );
  return parts.join("\n\n").slice(0, maxChars);
}
