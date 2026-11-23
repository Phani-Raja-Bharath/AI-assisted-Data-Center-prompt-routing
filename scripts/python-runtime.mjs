import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

export const root = resolve(import.meta.dirname, "..");

export function pythonCommand() {
  const windows = process.platform === "win32";
  const candidates = [process.env.PYTHON,
    join(root, ".venv-dev", windows ? "Scripts/python.exe" : "bin/python"),
    join(root, ".venv", windows ? "Scripts/python.exe" : "bin/python"),
    "python", "python3"].filter(Boolean);
  for (const candidate of candidates) {
    if ((candidate.includes("/") || candidate.includes("\\")) && !existsSync(candidate)) continue;
    const probe = spawnSync(candidate, ["--version"], { encoding: "utf8", windowsHide: true });
    if (!probe.error && probe.status === 0) return candidate;
  }
  throw new Error("No working Python found. Create .venv-dev and install requirements-dev.txt.");
}

export function runPython(args) {
  const result = spawnSync(pythonCommand(), args, {
    cwd: root, stdio: "inherit", windowsHide: true,
    env: { ...process.env, PYTHONUTF8: "1" },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
