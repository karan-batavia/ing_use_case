// static/js/app.js
(() => {
  // Simple local token store (for demo)
  let token = localStorage.getItem("auth_token") || "";

  // Helpers
  function setStatus(el, txt) { el.textContent = txt; }
  function authHeaders() {
    return token ? { "Authorization": "Bearer " + token, "Content-Type": "application/json" } : { "Content-Type": "application/json" };
  }
  function plainHeaders() {
    return token ? { "Authorization": "Bearer " + token } : {};
  }

  // DOM
  const scrubInput = document.getElementById("scrub-input");
  const scrubBtn = document.getElementById("scrub-btn");
  const scrubOutput = document.getElementById("scrub-output");

  const fileInput = document.getElementById("file-input");
  const scrubFileBtn = document.getElementById("scrub-file-btn");
  const scrubFileOutput = document.getElementById("scrub-file-output");

  const predictPrompt = document.getElementById("predict-prompt");
  const predictBtn = document.getElementById("predict-btn");
  const predictOutput = document.getElementById("predict-output");

  const descrubOutput = document.getElementById("descrub-output");

  const loginBtn = document.getElementById("login-btn");
  const emailInput = document.getElementById("email");
  const passInput = document.getElementById("password");

  // Utility to POST JSON
  async function postJson(url, body) {
    const res = await fetch(url, { method: "POST", headers: authHeaders(), body: JSON.stringify(body) });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return res.json();
  }

  // POST multipart (file)
  async function postFile(url, formData) {
    const res = await fetch(url, { method: "POST", headers: token ? { "Authorization": "Bearer " + token } : {}, body: formData });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return res.json();
  }

  // Login (very basic)
  if (loginBtn) {
    loginBtn.addEventListener("click", async () => {
      const email = emailInput.value;
      const password = passInput.value;
      if (!email || !password) { alert("email+password required"); return; }
      try {
        const res = await fetch("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password })
        });
        if (!res.ok) {
          alert("Login failed");
          return;
        }
        const j = await res.json();
        token = j.access_token;
        localStorage.setItem("auth_token", token);
        document.getElementById("user-email").textContent = email;
        document.getElementById("user-pill").style.display = "flex";
        alert("Logged in");
      } catch (e) {
        console.error(e);
        alert("Login error");
      }
    });
  }

  // Chain: scrub -> predict -> descrub
  async function chainScrubToPredictAndDescrub(scrubEndpoint, bodyOrForm) {
    // scrubEndpoint: "/scrub" (json) or "/scrub-file" (form)
    // returns object { scrubbed_text, prediction_text, descrubbed_text, metadata... }
    try {
      // 1) call scrub
      let scrubResp;
      if (scrubEndpoint === "/scrub") {
        scrubResp = await postJson("/scrub", { text: scrubInput.value });
      } else {
        scrubResp = await postFile("/scrub-file", bodyOrForm);
      }

      const scrubbed_text = scrubResp.scrubbed_text || scrubResp.scrubbed_text || (scrubResp.scrubbed_text);
      // display scrubbed
      scrubOutput.textContent = scrubbed_text;
      scrubFileOutput.textContent = scrubbed_text;

      // 2) call predict with scrubbed_text
      predictPrompt.value = scrubbed_text;
      predictOutput.textContent = "Generating…";
      const pred = await postJson("/predict", { prompt: scrubbed_text });
      predictOutput.textContent = pred.prediction || "";

      // 3) call descrub with the session id (predict returned session_id)
      // We used server-side session_id for mapping; but predict returns session_id generated inside predict.
      // For our chain, server generated session earlier during scrub; but we don't have that id here.
      // So we call /descrub providing the session_id that scrub endpoint returned if present.
      let session_id = scrubResp.session_id || pred.session_id || scrubResp.session_id || scrubResp.session_id;
      // fallback: if no session_id provided by server, try to call descrub without session (descrub will return original unchanged)
      descrubOutput.textContent = "Restoring placeholders…";
      let descrubBody = { session_id: session_id || "", text: pred.prediction || scrubbed_text };
      const descr = await postJson(`/descrub`, descrubBody);
      descrubOutput.textContent = descr.descrubbed_text || "";

      return {
        scrubbed_text,
        prediction_text: pred.prediction || "",
        descrubbed_text: descr.descrubbed_text || "",
      };
    } catch (e) {
      console.error(e);
      alert("Error in scrub->predict->descrub: " + e.message);
      predictOutput.textContent = "";
      descrubOutput.textContent = "";
      return null;
    }
  }

  // Event: Scrub Text button
  if (scrubBtn) {
    scrubBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      scrubOutput.textContent = "Scrubbing…";
      predictOutput.textContent = "";
      descrubOutput.textContent = "";
      await chainScrubToPredictAndDescrub("/scrub");
    });
  }

  // Event: Upload & Scrub button
  if (scrubFileBtn) {
    scrubFileBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const file = fileInput.files[0];
      if (!file) { alert("Choose a file"); return; }
      const form = new FormData();
      form.append("file", file);
      scrubFileOutput.textContent = "Uploading & scrubbing…";
      predictOutput.textContent = "";
      descrubOutput.textContent = "";
      await chainScrubToPredictAndDescrub("/scrub-file", form);
    });
  }

  // Event: Generate button (manual)
  if (predictBtn) {
    predictBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const prompt = predictPrompt.value || "";
      if (!prompt) { alert("Prompt required"); return; }
      predictOutput.textContent = "Generating…";
      try {
        const pred = await postJson("/predict", { prompt });
        predictOutput.textContent = pred.prediction || "";

        // attempt descrub: ideally you pass session id; but try with empty session (descrub will try db/memory)
        const descr = await postJson("/descrub", { session_id: "", text: pred.prediction || "" });
        descrubOutput.textContent = descr.descrubbed_text || "";

      } catch (e) {
        console.error(e);
        predictOutput.textContent = "";
        descrubOutput.textContent = "";
        alert("Prediction error: " + e.message);
      }
    });
  }

  // Audit buttons
  const auditOpenBtn = document.getElementById("audit-open-btn");
  const auditDownloadBtn = document.getElementById("audit-download-btn");
  if (auditOpenBtn) auditOpenBtn.addEventListener("click", () => window.open("/audit/log", "_blank"));
  if (auditDownloadBtn) auditDownloadBtn.addEventListener("click", () => window.open("/audit/download", "_blank"));

})();
