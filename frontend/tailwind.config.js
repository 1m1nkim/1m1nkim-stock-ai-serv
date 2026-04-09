/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a',
        panel: 'rgba(30, 41, 59, 0.7)',
        accent: '#38bdf8',
        'text-main': '#f8fafc',
        'text-muted': '#94a3b8',
        'border-color': 'rgba(255, 255, 255, 0.1)',
      }
    },
  },
  plugins: [],
}
