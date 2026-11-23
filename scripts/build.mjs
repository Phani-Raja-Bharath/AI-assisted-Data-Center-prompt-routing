import { runPython } from "./python-runtime.mjs";

// Streamlit is interpreted: validate Python syntax/dependencies and regressions
// after TypeScript checks neon.ts. There is no browser JavaScript bundle.
runPython(["scripts/check_build.py"]);
runPython(["-m", "pytest", "tests", "-q"]);
console.log("BUILD PASSED: Neon configuration, Python application, and regression tests.");
