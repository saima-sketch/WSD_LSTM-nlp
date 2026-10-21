// ---------------------------------------------------------------
// script.js
// Frontend logic: reads sentence, calls FastAPI /predict,
// and displays the LSTM prediction.
// The LSTM model itself lives in the Python backend.
// ---------------------------------------------------------------

const API_BASE_URL = "http://127.0.0.1:8000";

// DOM elements
const sentenceInput   = document.getElementById("sentenceInput");
const predictBtn      = document.getElementById("predictBtn");
const clearBtn        = document.getElementById("clearBtn");
const loadingIndicator = document.getElementById("loadingIndicator");
const errorMessage    = document.getElementById("errorMessage");
const resultSection   = document.getElementById("resultSection");

const resultSentence  = document.getElementById("resultSentence");
const resultMeaning   = document.getElementById("resultMeaning");
const resultClass     = document.getElementById("resultClass");

const probFinancialText = document.getElementById("probFinancialText");
const probRiverText     = document.getElementById("probRiverText");
const probFinancialBar  = document.getElementById("probFinancialBar");
const probRiverBar      = document.getElementById("probRiverBar");

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function clearMessages() {
  hide(errorMessage);
  errorMessage.textContent = "";
  hide(resultSection);
}

function displayError(msg) {
  errorMessage.textContent = msg;
  show(errorMessage);
}

// Basic HTML escape to prevent XSS
function escapeHtml(text) {
  if (typeof text !== "string") return text;
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Format probability to a whole-number percent string
function formatPercent(p) {
  return Math.round(p * 100) + "%";
}

// ---------------------------------------------------------------
// Render prediction
// ---------------------------------------------------------------
function renderResult(data) {
  resultSentence.textContent = escapeHtml(data.sentence);
  resultMeaning.textContent  = data.meaning;
  resultClass.textContent    = data.class;

  // Probabilities
  const fin = data.probabilities.financial_bank;
  const riv = data.probabilities.river_bank;

  probFinancialText.textContent = formatPercent(fin);
  probRiverText.textContent     = formatPercent(riv);

  probFinancialBar.style.width = (fin * 100).toFixed(1) + "%";
  probRiverBar.style.width     = (riv * 100).toFixed(1) + "%";

  // Optional color emphasis on the predicted meaning
  if (data.class === 0) {
    resultMeaning.style.color = "#1d4ed8"; // blue for financial
  } else {
    resultMeaning.style.color = "#0d9488"; // teal for river
  }

  show(resultSection);
}

// ---------------------------------------------------------------
// Main prediction function
// ---------------------------------------------------------------
async function predictMeaning() {
  const sentence = sentenceInput.value.trim();

  if (sentence === "") {
    clearMessages();
    displayError("Please enter a sentence containing the word \"bank\".");
    return;
  }

  clearMessages();
  show(loadingIndicator);
  predictBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: sentence })
    });

    if (!response.ok) {
      let msg = `Server responded with ${response.status}`;
      try {
        const err = await response.json();
        if (err.detail) msg = err.detail;
      } catch (_) { /* ignore */ }
      throw new Error(msg);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    console.error("Prediction error:", err);
    if (
      err.message.includes("Failed to fetch") ||
      err.message.includes("NetworkError") ||
      err.name === "TypeError"
    ) {
      displayError(
        "⚠️ Cannot connect to the backend. Make sure FastAPI is running at " +
        "http://127.0.0.1:8000"
      );
    } else {
      displayError("Error: " + err.message);
    }
    hide(resultSection);
  } finally {
    hide(loadingIndicator);
    predictBtn.disabled = false;
  }
}

// ---------------------------------------------------------------
// Clear handler
// ---------------------------------------------------------------
function clearAll() {
  sentenceInput.value = "";
  clearMessages();
  hide(loadingIndicator);

  // Reset probability bars
  probFinancialBar.style.width = "0%";
  probRiverBar.style.width     = "0%";
  probFinancialText.textContent = "0%";
  probRiverText.textContent     = "0%";

  sentenceInput.focus();
}

// ---------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------
predictBtn.addEventListener("click", predictMeaning);
clearBtn.addEventListener("click", clearAll);

// Example quick-fill buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    sentenceInput.value = btn.dataset.text;
    sentenceInput.focus();
  });
});

// Ctrl+Enter / Cmd+Enter to predict
sentenceInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    predictMeaning();
  }
});

// On page load
window.addEventListener("load", () => {
  hide(loadingIndicator);
  hide(errorMessage);
  hide(resultSection);
});