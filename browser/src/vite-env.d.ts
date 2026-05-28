// Local shim for Vite's `import.meta.env` so we get typed access to
// VITE_*-prefixed build-time env vars without depending on the
// `vite/client` reference type (which would pull in DOM lib widening
// we don't need and changes the tsconfig footprint).
//
// Vite statically replaces `import.meta.env.VITE_X` with the env var's
// literal value at build time. The replacement only works on DIRECT
// access — aliasing through a local variable hides the access pattern
// from Vite's plugin, so always read these inline at the call site.

interface ImportMetaEnv {
  readonly VITE_POSTHOG_KEY?: string;
  readonly VITE_POSTHOG_HOST?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
