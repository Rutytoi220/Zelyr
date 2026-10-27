import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

import tailwindcss from '@tailwindcss/vite'

import fs from 'fs';
import os from 'os';
import path from 'path';

// Read the Zelyr daemon token, falling back to the legacy AXIOM location.
let tokenValue = '';
const tokenCandidates = [
  path.join(os.homedir(), '.zelyr', 'daemon.token'),
  path.join(os.homedir(), '.axiom', 'daemon.token'),
];
for (const tokenPath of tokenCandidates) {
  try {
    tokenValue = fs.readFileSync(tokenPath, 'utf8').trim();
    break;
  } catch (e) {
    // Try the next supported location.
  }
}
if (!tokenValue) {
  console.warn(
    `Warning: Could not read ${tokenCandidates.join(' or ')}`,
  );
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    'import.meta.env.VITE_DAEMON_TOKEN': JSON.stringify(tokenValue)
  }
})
