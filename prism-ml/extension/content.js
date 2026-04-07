function extractPageText() {
  if (!document.body) {
    return '';
  }

  return document.body.innerText.replace(/\s+/g, ' ').trim();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'GET_PAGE_TEXT') {
    return;
  }

  sendResponse({
    url: window.location.href,
    title: document.title,
    text: extractPageText()
  });
});
