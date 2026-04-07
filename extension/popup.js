const API_URL = 'http://localhost:8000/analyze';

const analyzeBtn = document.getElementById('analyzeBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const explanationEl = document.getElementById('explanation');

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  statusEl.textContent = isLoading ? 'Extracting page text and analyzing...' : 'Ready.';
}

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
  resultEl.hidden = true;
}

function showResult(data) {
  errorEl.hidden = true;
  resultEl.hidden = false;

  predictionEl.textContent = data.prediction ?? 'N/A';

  if (typeof data.confidence === 'number') {
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
  } else {
    confidenceEl.textContent = data.confidence ?? 'N/A';
  }

  explanationEl.textContent = data.explanation ?? 'No explanation provided.';
}

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
    chrome.tabs.sendMessage(tabId, { type: 'GET_PAGE_TEXT' }, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }

      if (!response?.text) {
        reject(new Error('Unable to extract text from this page.'));
        return;
      }

      resolve(response);
    });
  });
}

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
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        url: page.url,
        title: page.title,
        text: page.text
      })
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

analyzeBtn.addEventListener('click', analyzeCurrentPage);
