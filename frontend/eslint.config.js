// ESLint flat config for the ValonyLabs Studio frontend (S06).
//
// Goal: catch obvious bugs (unused vars, dead code, broken hooks deps)
// without flagging things TypeScript already covers. Errors-only — the
// `--max-warnings 0` flag in package.json's `lint` script means warnings
// fail CI, so every rule is `error` or `off`.
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "dist",
      "node_modules",
      "**/*.tsbuildinfo",
      "vite.config.ts",
      "tailwind.config.js",
      "postcss.config.js",
      "eslint.config.js",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      globals: {
        // Browser globals the SPA uses
        window: "readonly",
        document: "readonly",
        fetch: "readonly",
        FormData: "readonly",
        File: "readonly",
        XMLHttpRequest: "readonly",
        AbortSignal: "readonly",
        TextDecoder: "readonly",
        URLSearchParams: "readonly",
        localStorage: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        requestAnimationFrame: "readonly",
        cancelAnimationFrame: "readonly",
        navigator: "readonly",
        crypto: "readonly",
      },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      // The codebase intentionally uses `any` in places where the
      // upstream API shape is unstable (streaming meta blobs etc.) —
      // surfacing these as warnings adds noise without catching bugs.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "no-empty": ["error", { allowEmptyCatch: true }],
    },
  }
);
