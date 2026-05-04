/// <reference types="vite/client" />

// Project-specific VITE_* env vars consumed via import.meta.env.
// Add a new line whenever a new VITE_ variable is introduced.
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
