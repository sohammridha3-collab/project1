/* ==========================================================
   ML PREDICTION (unchanged): talks to FastAPI POST /predict
   ========================================================== */

const API_URL = `${window.location.origin}/predict`;

const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");
const retryBtn = document.getElementById("retry-btn");

const states = ["idle", "loading", "success", "error"];
function showState(name) {
  states.forEach((s) => {
    document.getElementById("state-" + s).hidden = s !== name;
  });
}

// Field types must match the StudentInput Pydantic model
const INT_FIELDS = ["Age", "Daily_Unlocks"];
const FLOAT_FIELDS = [
  "Avg_Daily_Usage_Hours",
  "Study_Hours",
  "Physical_Activity_Hours",
  "Sleep_Hours_Per_Night",
];
const TEXT_FIELDS = [
  "Gender",
  "Country",
  "Academic_Level",
  "Most_Used_Platform",
  "Purpose_Of_Use",
  "Stress_Level",
];

function clearFieldErrors() {
  form.querySelectorAll(".err").forEach((el) => (el.textContent = ""));
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
}

function setFieldError(name, message) {
  const el = form.elements[name];
  const msg = form.querySelector(`.err[data-for="${name}"]`);
  if (!el || !msg) return false;
  el.classList.add("invalid");
  msg.textContent = message;
  return true;
}

// Browser-side checks that mirror the backend limits. The backend still has the final say.
const LIMITS = {
  Age: [10, 100],
  Avg_Daily_Usage_Hours: [0, 24],
  Daily_Unlocks: [0, 1000],
  Study_Hours: [0, 24],
  Physical_Activity_Hours: [0, 24],
  Sleep_Hours_Per_Night: [0, 24],
};

function validate() {
  clearFieldErrors();
  let firstBad = null;
  const fail = (name, msg) => {
    setFieldError(name, msg);
    firstBad = firstBad || form.elements[name];
  };

  [...INT_FIELDS, ...FLOAT_FIELDS].forEach((name) => {
    const raw = form.elements[name].value.trim();
    if (raw === "") return fail(name, "Enter a value.");
    const n = Number(raw);
    const [min, max] = LIMITS[name];
    if (Number.isNaN(n)) return fail(name, "Enter a number.");
    if (INT_FIELDS.includes(name) && !Number.isInteger(n)) return fail(name, "Use a whole number.");
    if (n < min || n > max) fail(name, `Enter a value between ${min} and ${max}.`);
  });

  TEXT_FIELDS.forEach((name) => {
    if (!form.elements[name].value) fail(name, "Choose an option.");
  });

  if (firstBad) firstBad.focus();
  return !firstBad;
}

function buildPayload() {
  const payload = {};
  INT_FIELDS.forEach((n) => (payload[n] = parseInt(form.elements[n].value, 10)));
  FLOAT_FIELDS.forEach((n) => (payload[n] = parseFloat(form.elements[n].value)));
  TEXT_FIELDS.forEach((n) => (payload[n] = form.elements[n].value));
  return payload;
}

function showError(title, content) {
  document.getElementById("error-title").textContent = title;
  const box = document.getElementById("error-message");
  box.replaceChildren();
  if (Array.isArray(content)) {
    const ul = document.createElement("ul");
    content.forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      ul.appendChild(li);
    });
    box.appendChild(ul);
  } else {
    box.textContent = content;
  }
  showState("error");
}

function toneFor(score) {
  if (score >= 7.5) return "var(--good)";
  if (score >= 6.0) return "var(--accent)";
  if (score >= 4.5) return "var(--fair)";
  return "var(--poor)";
}

function noteFor(score) {
  if (score >= 7.5) return "Habits point to strong well-being.";
  if (score >= 6.0) return "Well-being looks reasonably healthy.";
  if (score >= 4.5) return "Some habits may be worth adjusting.";
  return "Habits suggest a higher risk. Consider talking to a counselor.";
}

function showResult(data) {
  const score = Number(data.predicted_mental_health_score);
  const tone = toneFor(score);
  const result = document.getElementById("state-success");
  result.style.setProperty("--tone", tone);

  document.getElementById("score-value").textContent = score.toFixed(1);
  document.getElementById("risk-badge").textContent = data.risk_category;
  document.getElementById("risk-note").textContent = noteFor(score);

  const fill = document.getElementById("gauge-fill");
  fill.style.strokeDashoffset = 100;
  showState("success");
  // Scale is 1 to 10; clamp so the arc stays inside the track
  const pct = Math.max(0, Math.min(1, (score - 1) / 9));
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fill.style.strokeDashoffset = 100 - pct * 100;
  }));
}

async function handleSubmit(event) {
  event.preventDefault();
  if (!validate()) return;

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";
  showState("loading");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    let body = null;
    try { body = await response.json(); } catch (_) { /* non-JSON response */ }

    if (response.status === 422 && body && Array.isArray(body.detail)) {
      // FastAPI validation errors: [{ loc: ["body", "Age"], msg: "..." }]
      const general = [];
      body.detail.forEach((d) => {
        const field = d.loc && d.loc[d.loc.length - 1];
        if (!setFieldError(field, d.msg)) general.push(`${field}: ${d.msg}`);
      });
      showError(
        "Some inputs were rejected",
        general.length ? general : "Check the highlighted fields and try again."
      );
      return;
    }

    if (!response.ok) {
      const detail = body && typeof body.detail === "string" ? body.detail : `Server returned status ${response.status}.`;
      const title = response.status === 503 ? "Model unavailable" : "Prediction failed";
      showError(title, detail);
      return;
    }

    showResult(body);
  } catch (err) {
    showError(
      "Cannot reach the server",
      "Check that FastAPI is running at http://127.0.0.1:8000 and that you opened this page in the same browser as the server."
    );
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict score";
  }
}

function handleReset() {
  form.reset();
  clearFieldErrors();
  showState("idle");
}

form.addEventListener("submit", handleSubmit);
resetBtn.addEventListener("click", handleReset);
retryBtn.addEventListener("click", () => showState("idle"));
form.addEventListener("input", (e) => {
  if (e.target.classList.contains("invalid")) {
    e.target.classList.remove("invalid");
    const msg = form.querySelector(`.err[data-for="${e.target.name}"]`);
    if (msg) msg.textContent = "";
  }
});


/* ==========================================================
   AUTH + SCREEN SWITCHING (demo only, independent of ML code)
   To add real authentication later, replace Auth.authenticate()
   with a fetch() to your login endpoint. Nothing else changes.
   ========================================================== */
const Auth = {
  REMEMBER_KEY: "mindscope_remembered_user",
  GMAIL_PATTERN: /^[a-z0-9]+(?:\.[a-z0-9]+)*(?:\+[a-z0-9]+(?:[.-][a-z0-9]+)*)?@gmail\.com$/i,

  // Client-side demo validation only; no account or password is stored or checked.
  authenticate(identifier, password) {
    const email = identifier.trim();
    if (this.GMAIL_PATTERN.test(email) && password.length > 0) return { ok: true };
    return { ok: false, message: "Invalid username or password." };
  },

  getRemembered() {
    try { return localStorage.getItem(this.REMEMBER_KEY) || ""; } catch (_) { return ""; }
  },
  setRemembered(value) {
    try {
      if (value) localStorage.setItem(this.REMEMBER_KEY, value);
      else localStorage.removeItem(this.REMEMBER_KEY);
    } catch (_) { /* storage unavailable */ }
  },
};

(function initLogin() {
  const loginScreen = document.getElementById("login-screen");
  const appScreen = document.getElementById("app-screen");
  const loginForm = document.getElementById("login-form");
  const idInput = document.getElementById("login-id");
  const pwInput = document.getElementById("login-pw");
  const idErr = document.getElementById("login-id-err");
  const pwErr = document.getElementById("login-pw-err");
  const msg = document.getElementById("login-msg");
  const loginBtn = document.getElementById("login-btn");
  const remember = document.getElementById("remember");
  const pwToggle = document.getElementById("pw-toggle");

  function showScreen(name) {
    loginScreen.hidden = name !== "login";
    appScreen.hidden = name !== "app";
    window.scrollTo(0, 0);
  }

  function showMsg(text, info) {
    msg.textContent = text;
    msg.classList.toggle("info", !!info);
    msg.hidden = false;
  }

  function clearErrors() {
    idErr.textContent = "";
    pwErr.textContent = "";
    idInput.classList.remove("invalid");
    pwInput.classList.remove("invalid");
    msg.hidden = true;
  }

  function validateLogin() {
    clearErrors();
    let bad = null;
    const email = idInput.value.trim();
    if (!email) {
      idErr.textContent = "Enter your Gmail address.";
      idInput.classList.add("invalid");
      bad = idInput;
    } else if (!Auth.GMAIL_PATTERN.test(email)) {
      idErr.textContent = "Enter a valid Gmail address, such as name@gmail.com.";
      idInput.classList.add("invalid");
      bad = idInput;
    }
    if (!pwInput.value) {
      pwErr.textContent = "Enter your password.";
      pwInput.classList.add("invalid");
      if (!bad) bad = pwInput;
    }
    if (bad) bad.focus();
    return !bad;
  }

  // Restore remembered username
  const saved = Auth.getRemembered();
  if (saved) { idInput.value = saved; remember.checked = true; }

  pwToggle.addEventListener("click", () => {
    const show = pwInput.type === "password";
    pwInput.type = show ? "text" : "password";
    pwToggle.textContent = show ? "Hide" : "Show";
    pwToggle.setAttribute("aria-pressed", String(show));
    pwToggle.setAttribute("aria-label", show ? "Hide password" : "Show password");
  });

  document.getElementById("forgot-link").addEventListener("click", (e) => {
    e.preventDefault();
    showMsg("Password reset is not available in this demo.", true);
  });

  [idInput, pwInput].forEach((el) =>
    el.addEventListener("input", () => {
      el.classList.remove("invalid");
      (el === idInput ? idErr : pwErr).textContent = "";
      msg.hidden = true;
    })
  );

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!validateLogin()) return;

    loginBtn.disabled = true;
    loginBtn.textContent = "Logging in…";
    try {
      const result = await Auth.authenticate(idInput.value, pwInput.value);
      if (!result.ok) {
        showMsg(result.message || "Login failed.");
        return;
      }
      Auth.setRemembered(remember.checked ? idInput.value.trim() : "");
      pwInput.value = "";
      showScreen("app");
      document.querySelector("#app-screen h1").focus?.();
    } catch (_) {
      showMsg("Could not log in. Please try again.");
    } finally {
      loginBtn.disabled = false;
      loginBtn.textContent = "Log in";
    }
  });

  document.getElementById("logout-btn").addEventListener("click", () => {
    document.getElementById("reset-btn").click(); // clears prediction form and result
    clearErrors();
    showScreen("login");
    (idInput.value ? pwInput : idInput).focus();
  });
})();
