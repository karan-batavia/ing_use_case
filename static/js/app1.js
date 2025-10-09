// tiny helpers
const $  = (q, ctx=document) => ctx.querySelector(q);
const $$ = (q, ctx=document) => Array.from(ctx.querySelectorAll(q));
const api = (path) => path; // same origin

function setStatus(el, msg, ok=false) {
  if (!el) return;
  el.textContent = msg || "";
  el.style.color = ok ? "#0a7b12" : "#59657a";
}

function token() { return localStorage.getItem("authToken") || ""; }
function setToken(t) { t ? localStorage.setItem("authToken", t) : localStorage.removeItem("authToken"); }
function setUser(email, role, userId) {
  email ? localStorage.setItem("userEmail", email) : localStorage.removeItem("userEmail");
  role  ? localStorage.setItem("userRole",  role)  : localStorage.removeItem("userRole");
  userId? localStorage.setItem("userId",   userId) : localStorage.removeItem("userId");
}

function authHeaders() {
  const t = token();
  return t ? { "Authorization": "Bearer " + t } : {};
}

// --- UI state
function refreshAuthUI() {
  const hasToken = !!token();
  const authCard = $("#auth-card");
  const regCard  = $("#register-card");
  const logoutBtn = $("#logout-btn");
  const userPill  = $("#user-pill");

  const userEmail = localStorage.getItem("userEmail") || "signed@out";
  const userRole  = localStorage.getItem("userRole")  || "guest";
  const userEmailEl = $("#user-email");
  const userRoleEl  = $("#user-role");
  if (userEmailEl) userEmailEl.textContent = userEmail;
  if (userRoleEl)  userRoleEl.textContent  = userRole;

  if (userPill)  userPill.style.display  = hasToken ? "inline-flex" : "none";
  if (logoutBtn) logoutBtn.style.display = hasToken ? "inline-flex" : "none";

  if (authCard)  authCard.style.display  = hasToken ? "none" : "block";
  if (regCard && hasToken) regCard.style.display = "none";
}

// --- auth
async function login() {
  const email = $("#email")?.value.trim();
  const password = $("#password")?.value;
  const statusEl = $("#login-status");
  setStatus(statusEl, "Logging in…");

  try {
    const res = await fetch(api("/auth/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");

    setToken(data.access_token);
    setUser(email, data.role || "user", data.user_id || "");
    setStatus(statusEl, "Logged in", true);
    refreshAuthUI();
  } catch (e) {
    setStatus(statusEl, e.message || String(e));
  }
}

async function register() {
  const full_name = $("#reg-name")?.value.trim();
  const email     = $("#reg-email")?.value.trim();
  const password  = $("#reg-pass")?.value;
  const role      = $("#reg-role")?.value || "user";
  const statusEl  = $("#register-status");

  setStatus(statusEl, "Creating account…");
  try {
    const res = await fetch(api("/auth/register"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name, role }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed");

    setStatus(statusEl, "Account created. You can log in now.", true);
    const reg = $("#register-card"), auth = $("#auth-card");
    if (reg)  reg.style.display  = "none";
    if (auth) auth.style.display = "block";
  } catch (e) {
    setStatus(statusEl, e.message || String(e));
  }
}

async function logout() {
  try {
    await fetch("/auth/logout", { method: "POST", headers: { ...authHeaders() } });
  } catch (e) { /* ignore */ }
  setToken("");
  setUser("", "", "");
  refreshAuthUI();
}


// --- predict
async function doPredict() {
  const prompt = $("#predict-prompt")?.value || "";
  const model  = $("#predict-model")?.value || "gemini-pro";
  const outEl  = $("#predict-output");
  const kpis   = $("#predict-kpis");
  if (outEl) outEl.textContent = "Generating…";

  try {
    const res = await fetch(api("/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ prompt, model_name: model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");

    if (outEl) outEl.textContent = data.prediction || "";
    const m = $("#predict-model-pill");
    const s = $("#predict-session-pill");
    if (m) m.textContent = `Model: ${data.model_used}`;
    if (s) s.textContent = `Session: ${data.session_id}`;
    if (kpis) kpis.style.display = "flex";
  } catch (e) {
    if (outEl) outEl.textContent = String(e);
  }
}

// --- scrub text
async function doScrub() {
  const text = $("#scrub-input")?.value || "";
  const out  = $("#scrub-output");
  const kpis = $("#scrub-kpis");
  if (out) out.textContent = "Scrubbing…";

  try {
    const res = await fetch(api("/scrub"), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Scrub failed");

    if (out) out.textContent = data.scrubbed_text;
    const m = $("#kpi-matches");
    const r = $("#kpi-reduction");
    const l = $("#kpi-lengths");
    if (m) m.textContent = `Matches: ${data.matches_found}`;
    if (r) r.textContent = `Reduction: ${data.reduction_percentage}%`;
    if (l) l.textContent = `Len ${data.original_length} → ${data.scrubbed_length}`;
    if (kpis) kpis.style.display = "flex";
  } catch (e) {
    if (out) out.textContent = String(e);
  }
}

// --- scrub file
async function doScrubFile() {
  const fi  = $("#file-input");
  const out = $("#scrub-file-output");
  const kpis = $("#file-kpis");
  if (!fi || !fi.files || !fi.files[0]) {
    if (out) out.textContent = "Pick a file first.";
    return;
  }
  const fd = new FormData();
  fd.append("file", fi.files[0]);

  if (out) out.textContent = "Uploading & scrubbing…";
  try {
    const res = await fetch(api("/scrub-file"), {
      method: "POST",
      headers: { ...authHeaders() },
      body: fd,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Scrub file failed");

    if (out) out.textContent = data.scrubbed_text;
    const n = $("#file-name");
    const s = $("#file-size");
    const m = $("#file-matches");
    if (n) n.textContent = data.filename;
    if (s) s.textContent = `${(data.file_size/1024).toFixed(1)} KB`;
    if (m) m.textContent = `Matches: ${data.matches_found}`;
    if (kpis) kpis.style.display = "flex";
  } catch (e) {
    if (out) out.textContent = String(e);
  }
}

// --- audit helpers
function openAudit()   { window.open("/audit/log", "_blank"); }
function openOriginal(){ window.open("/audit/original", "_blank"); }
function openScrubbed(){ window.open("/audit/scrubbed", "_blank"); }
function downloadAudit(){ window.location.href = "/audit/download"; }

// --- bootstrap
window.addEventListener("DOMContentLoaded", () => {
  if (window.location.hash === "#audit")    { window.location.replace("/audit/log"); return; }
  if (window.location.hash === "#original") { window.location.replace("/audit/original"); return; }
  if (window.location.hash === "#scrubbed") { window.location.replace("/audit/scrubbed"); return; }

  $("#login-btn")?.addEventListener("click", login);
  $("#register-btn")?.addEventListener("click", register);
  $("#logout-btn")?.addEventListener("click", logout);

  $("#show-register")?.addEventListener("click", () => {
    const auth = $("#auth-card"); const reg = $("#register-card");
    if (auth) auth.style.display = "none";
    if (reg)  reg.style.display  = "block";
  });
  $("#show-login")?.addEventListener("click", () => {
    const auth = $("#auth-card"); const reg = $("#register-card");
    if (reg)  reg.style.display  = "none";
    if (auth) auth.style.display = "block";
  });

  $("#predict-btn")?.addEventListener("click", doPredict);
  $("#scrub-btn")?.addEventListener("click", doScrub);
  $("#scrub-file-btn")?.addEventListener("click", doScrubFile);

  $("#audit-open-btn")?.addEventListener("click", openAudit);
  $("#audit-original-btn")?.addEventListener("click", openOriginal);
  $("#audit-scrubbed-btn")?.addEventListener("click", openScrubbed);
  $("#audit-download-btn")?.addEventListener("click", downloadAudit);

  refreshAuthUI();
});
