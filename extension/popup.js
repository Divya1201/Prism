const API_URL = 'http://localhost:8000/analyze';

const analyzeBtn = document.getElementById('analyzeBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');

const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const explanationEl = document.getElementById('explanation');
const evidenceEl = document.getElementById('evidence'); 

// -----------------------------
// UI HELPERS
// -----------------------------
function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  statusEl.textContent = isLoading
    ? 'Extracting page text and analyzing...'
    : 'Ready.';
}

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
  resultEl.hidden = true;
}

function showResult(data) {
  errorEl.hidden = true;
  resultEl.hidden = false;

  // -----------------------------
  // Prediction
  // -----------------------------
  const pred = data.prediction ?? 'N/A';
  predictionEl.textContent = pred;

  const colorMap = {
    fabricated: 'red',
    manipulated: 'red',
    false_context: 'orange',
    false_connection: 'orange',
    satire: 'gold',
    sponsored: 'blue',
  };

  predictionEl.style.color = colorMap[pred] || 'black';

  // -----------------------------
  // Confidence
  // -----------------------------
  if (typeof data.confidence === 'number') {
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
  } else {
    confidenceEl.textContent = data.confidence ?? 'N/A';
  }

  // -----------------------------
  // Explanation
  // -----------------------------
  explanationEl.textContent =
    data.explanation ?? 'No explanation provided.';

  // -----------------------------
  // Evidence 
  // -----------------------------
  if (data.evidence && data.evidence.length > 0) {
    // Show top 3 evidence lines
    evidenceEl.innerHTML = data.evidence
      .slice(0, 3)
      .map((e) => {
        return `<div>• <a href="${e.source}" target="_blank">${e.text}</a></div>`;
      })
    .join('');
  } else {
    evidenceEl.textContent = 'No evidence found.';
  }
}

// -----------------------------
// TAB HELPERS
// -----------------------------
function getCurrentTab() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const [tab] = tabs;

      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }

      if (!tab?.id) {
        reject(new Error('No active tab found.'));
        return;
      }

      resolve(tab);
    });
  });
}

function getPageContent(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(
      tabId,
      { type: 'GET_PAGE_TEXT' },
      (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }

        if (!response?.text) {
          reject(new Error('Unable to extract text from this page.'));
          return;
        }

        resolve(response);
      }
    );
  });
}

// -----------------------------
// MAIN FUNCTION
// -----------------------------
async function analyzeCurrentPage() {
  setLoading(true);
  errorEl.hidden = true;

  try {
    const tab = await getCurrentTab();
    const page = await getPageContent(tab.id);

    statusEl.textContent = 'Sending page text to backend...';

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: page.text.slice(0, 3000), 
        title: page.title || '',
        url: page.url || '',
        source: 'webpage',
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend request failed (${response.status}).`);
    }

    const data = await response.json();

    showResult(data);
    statusEl.textContent = 'Analysis complete.';
  } catch (error) {
    showError(error.message || 'Unexpected error occurred.');
    statusEl.textContent = 'Analysis failed.';
  } finally {
    analyzeBtn.disabled = false;
  }
}

// -----------------------------
// EVENT LISTENER
// -----------------------------
analyzeBtn.addEventListener('click', analyzeCurrentPage);
