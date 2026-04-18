function getHeadline() {
  // Try common headline selectors
  const h1 = document.querySelector("h1");
  if (h1 && h1.innerText.length > 10) {
    return h1.innerText.trim();
  }

  // fallback
  return document.title || "";
}


function getArticleText() {
  // 1. Try <article> tag (best case)
  const article = document.querySelector("article");
  if (article) {
    return cleanText(article.innerText);
  }

  // 2. Try main content containers (common news sites)
  const main = document.querySelector("main");
  if (main) {
    return cleanText(main.innerText);
  }

  // 3. Fallback: collect meaningful paragraphs
  const paragraphs = document.querySelectorAll("p");

  let text = "";
  paragraphs.forEach(p => {
    const t = p.innerText.trim();

    // filter noise
    if (t.length > 50 && !isJunk(t)) {
      text += t + " ";
    }
  });

  return cleanText(text);
}


// Remove unwanted patterns
function isJunk(text) {
  const junkPatterns = [
    "subscribe",
    "sign up",
    "advertisement",
    "cookie",
    "privacy policy",
    "terms of use"
  ];

  return junkPatterns.some(j =>
    text.toLowerCase().includes(j)
  );
}


// Clean and normalize text
function cleanText(text) {
  return text
    .replace(/\s+/g, " ")
    .replace(/\n+/g, " ")
    .trim()
    .slice(0, 4000); // limit size
}


// MAIN MESSAGE HANDLER
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "GET_PAGE_TEXT") return;

  try {
    const headline = getHeadline();
    const articleText = getArticleText();

    sendResponse({
      url: window.location.href,
      title: headline,
      text: articleText
    });

  } catch (error) {
    sendResponse({
      url: window.location.href,
      title: document.title || "",
      text: "Failed to extract content"
    });
  }
});
