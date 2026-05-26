# Legacy HTML entry points

These are the original Vite-era HTML files that the Astro migration replaced
with `.astro` pages in `../src/pages/`. They are kept here as the conversion
source — `../scripts/convert_to_astro.mjs` reads them and regenerates the
Astro pages.

If you want to keep editing markup in raw HTML and regenerate Astro pages
mechanically:

```sh
# edit the .html file here
node scripts/convert_to_astro.mjs   # from browser/
```

If you instead want to edit Astro pages directly, that's fine too — the
generator does not overwrite without being explicitly invoked.

Files served at deploy time live in `dist/` after `npm run build` and are
produced from `../src/pages/*.astro`, not from this directory. Cloudflare
Pages ignores everything outside `dist/`.

If/when these files diverge enough from the Astro pages that re-generation
would lose work, drop this directory and treat the `.astro` files as the
source of truth.
