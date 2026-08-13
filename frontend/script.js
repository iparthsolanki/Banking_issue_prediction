// Point this at your FastAPI server. Same-origin by default.
const API_URL = "https://banking-issue-prediction.onrender.com/predict";

const input = document.getElementById("queryInput");
const askBtn = document.getElementById("askBtn");
const resultBox = document.getElementById("resultBox");
const errorBox = document.getElementById("errorBox");
const intentName = document.getElementById("intentName");
const confFill = document.getElementById("confFill");
const confValue = document.getElementById("confValue");
const echoBox = document.getElementById("echoBox");

function humanizeIntent(label) {
  return label.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function showError(msg) {
  resultBox.classList.remove("show");
  errorBox.textContent = msg;
  errorBox.classList.add("show");
}

function clearError() {
  errorBox.classList.remove("show");
  errorBox.textContent = "";
}

async function classify() {
  const text = input.value.trim();
  if (!text) {
    showError("Type something first — the model needs a query to classify.");
    return;
  }
  clearError();
  askBtn.disabled = true;
  askBtn.textContent = "Thinking…";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }

    const data = await res.json();
    const pct = Math.round((data.confidence || 0) * 1000) / 10;

    intentName.textContent = humanizeIntent(data.predicted_intent || "Unknown");
    confValue.textContent = pct + "%";
    echoBox.innerHTML = `Query: <b>"${data.input_text}"</b>`;

    resultBox.classList.add("show");
    // reset then set width for transition to animate on repeat queries
    confFill.style.width = "0%";
    requestAnimationFrame(() => {
      confFill.style.width = pct + "%";
    });

  } catch (err) {
    showError("Couldn't reach the model — check that the FastAPI server is running and API_URL is correct.");
    console.error(err);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Classify";
  }
}

askBtn.addEventListener("click", classify);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") classify();
});
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    classify();
  });
});
