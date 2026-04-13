/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        // Override Tailwind's default `font-sans` stack so every class
        // (including <body> inheritance and explicit font-sans) picks up
        // Tw Cen MT when available.
        sans: [
          '"Tw Cen MT"',
          '"Century Gothic"',
          '"Jost"',
          '"Futura"',
          '"Avenir Next"',
          '"Avenir"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
      },
      colors: {
        brand: {
          50: "#eff6ff",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          900: "#1e3a5f",
        },
      },
    },
  },
  plugins: [],
};
