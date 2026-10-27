/**
 * Zelyr Browser Bridge — Background Worker / Service Script.
 * Maintains a persistent WebSocket link to the local Zelyr daemon on ws://127.0.0.1:41144.
 * Enables zero-friction, deterministic browser automation for Zen Browser & Chrome.
 */

const WS_URL = "ws://127.0.0.1:41144";
let socket = null;
let reconnectTimer = null;
let pingTimer = null;
const INITIAL_RECONNECT_DELAY = 1000; // 1000ms initial reconnect delay (legacy 2000)
const MAX_RECONNECT_DELAY = 10000;    // 10000ms max reconnect delay (legacy 5000)
const RECONNECT_MULTIPLIER = 1.5;
const PING_INTERVAL_MS = 10000; // 10 seconds
let reconnectDelay = INITIAL_RECONNECT_DELAY;

function log(msg, ...args) {
  console.log(`[Zelyr Extension] ${msg}`, ...args);
}

function startKeepAlivePing() {
  stopKeepAlivePing();
  pingTimer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      log("Sending keep-alive ping to daemon...");
      try {
        socket.send(JSON.stringify({ type: "ping", action: "ping" }));
      } catch (err) {
        log("Failed to send ping:", err);
      }
    }
  }, PING_INTERVAL_MS);
}

function stopKeepAlivePing() {
  if (pingTimer) {
    clearInterval(pingTimer);
    pingTimer = null;
  }
}

function connect() {
  if (typeof WebSocket === "undefined") {
    return;
  }
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  log(`Connecting to daemon at ${WS_URL}...`);
  try {
    socket = new WebSocket(WS_URL);
  } catch (err) {
    log(`WebSocket creation failed: ${err.message}`);
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    log("Connected to Zelyr daemon.");
    reconnectDelay = INITIAL_RECONNECT_DELAY;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    startKeepAlivePing();
  };

  socket.onmessage = async (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (!msg) return;

      // Handle pong frames from daemon
      if (msg.type === "pong" || msg.action === "pong") {
        log("Received keep-alive pong from daemon.");
        return;
      }

      if (!msg.id || !msg.action) {
        log("Invalid command payload:", event.data);
        return;
      }
      await handleCommand(msg);
    } catch (err) {
      log("Error processing message:", err);
    }
  };

  socket.onclose = () => {
    log("Disconnected from daemon.");
    stopKeepAlivePing();
    socket = null;
    scheduleReconnect();
  };

  socket.onerror = (err) => {
    log("WebSocket error:", err);
    stopKeepAlivePing();
    if (socket) {
      socket.close();
    }
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  log(`Scheduling reconnect in ${reconnectDelay}ms...`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    reconnectDelay = Math.min(Math.round(reconnectDelay * RECONNECT_MULTIPLIER), MAX_RECONNECT_DELAY);
    connect();
  }, reconnectDelay);
}

function sendReply(data) {
  if (typeof global !== "undefined" && typeof global.__zelyrSendReplyHook === "function") {
    try {
      global.__zelyrSendReplyHook(data);
      return;
    } catch (_) {}
  }
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(data));
  } else {
    log("Cannot send reply: socket not open", data);
  }
}

async function getTargetTab(msg) {
  if (msg.tab_id) {
    try {
      const tab = await chrome.tabs.get(Number(msg.tab_id));
      if (tab) return tab;
    } catch (_) {}
  }

  // Fallback to active tab in current or last focused window
  const activeTabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (activeTabs && activeTabs.length > 0) {
    return activeTabs[0];
  }

  const anyActive = await chrome.tabs.query({ active: true });
  if (anyActive && anyActive.length > 0) {
    return anyActive[0];
  }

  const allTabs = await chrome.tabs.query({});
  return allTabs.length > 0 ? allTabs[0] : null;
}

function waitForSettled(options = {}) {
  const quietWindow = options.quietWindow || 75; // ms without mutations
  const maxTimeout = options.maxTimeout || 800;   // ms hard cap
  return new Promise((resolve) => {
    let timer = null;
    let maxTimer = null;
    let observer = null;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      if (maxTimer) clearTimeout(maxTimer);
      if (observer) {
        try { observer.disconnect(); } catch (e) {}
        observer = null;
      }
    };

    const done = () => {
      cleanup();
      // Double rAF tick to allow browser layout & paint flushing
      const raf = (typeof requestAnimationFrame === "function") ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
      raf(() => {
        raf(() => resolve());
      });
    };

    try {
      const target = (typeof document !== "undefined") ? (document.body || document.documentElement) : null;
      if (!target || typeof MutationObserver === "undefined") {
        done();
        return;
      }
      observer = new MutationObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(done, quietWindow);
      });
      observer.observe(target, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      });
    } catch (e) {
      done();
      return;
    }

    timer = setTimeout(done, quietWindow);
    maxTimer = setTimeout(done, maxTimeout);
  });
}

function isElementVisible(el) {
  if (!el) return false;
  if (typeof el.getBoundingClientRect === "function" && typeof window !== "undefined" && typeof window.getComputedStyle === "function") {
    try {
      const style = window.getComputedStyle(el);
      if (style) {
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          parseFloat(style.opacity || "1") === 0
        ) {
          return false;
        }
      }
      const rect = el.getBoundingClientRect();
      if (rect && rect.width === 0 && rect.height === 0 && el.offsetParent === null) {
        const root = typeof el.getRootNode === "function" ? el.getRootNode() : null;
        if (!root || root === document) {
          return false;
        }
      }
    } catch (_) {}
  }
  return true;
}

function clearOldLabels(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
  if (!root || currentDepth > maxDepth || seenRoots.has(root)) return;
  seenRoots.add(root);
  try {
    const old = Array.from(root.querySelectorAll("[data-zelyr-id]"));
    for (const el of old) el.removeAttribute("data-zelyr-id");
  } catch (_) {}
  try {
    const all = Array.from(root.querySelectorAll("*"));
    for (const el of all) {
      if (el.shadowRoot && el.shadowRoot.mode === "open") {
        clearOldLabels(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots);
      } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
        try {
          const doc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
          if (doc) clearOldLabels(doc, maxDepth, currentDepth + 1, seenRoots);
        } catch (_) {}
      }
    }
  } catch (_) {}
}

function collectInteractiveElements(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set(), extraSelector = "") {
  const interactive = [];
  if (!root || currentDepth > maxDepth || seenRoots.has(root)) return interactive;
  seenRoots.add(root);

  const baseSelector = [
    "a[href]",
    "button",
    "input",
    "textarea",
    "select",
    '[role="button"]',
    '[role="link"]',
    '[role="checkbox"]',
    '[role="tab"]',
    '[role="menuitem"]',
    "[onclick]",
    '[tabindex]:not([tabindex="-1"])',
  ].join(", ");

  const selector = extraSelector ? `${baseSelector}, ${extraSelector}` : baseSelector;

  try {
    const nodes = Array.from(root.querySelectorAll(selector));
    for (const el of nodes) {
      if (isElementVisible(el)) {
        interactive.push(el);
      }
    }
  } catch (_) {}

  try {
    const allElements = Array.from(root.querySelectorAll("*"));
    for (const el of allElements) {
      if (el.shadowRoot && el.shadowRoot.mode === "open") {
        const shadowChildren = collectInteractiveElements(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots, extraSelector);
        for (const child of shadowChildren) {
          interactive.push(child);
        }
      } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
        try {
          const frameDoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
          if (frameDoc && (frameDoc.body || frameDoc.documentElement)) {
            const frameChildren = collectInteractiveElements(frameDoc, maxDepth, currentDepth + 1, seenRoots, extraSelector);
            for (const child of frameChildren) {
              interactive.push(child);
            }
          }
        } catch (e) {
          // Cross-origin SecurityError - ignore safely
        }
      }
    }
  } catch (_) {}

  return interactive;
}

function findDeepElement(root, selector, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
  if (!root || currentDepth > maxDepth || seenRoots.has(root)) return null;
  seenRoots.add(root);

  try {
    const el = root.querySelector(selector);
    if (el) return el;
  } catch (_) {}

  try {
    const all = Array.from(root.querySelectorAll("*"));
    for (const node of all) {
      if (node.shadowRoot && node.shadowRoot.mode === "open") {
        const found = findDeepElement(node.shadowRoot, selector, maxDepth, currentDepth + 1, seenRoots);
        if (found) return found;
      } else if (node.tagName === "IFRAME" || node.tagName === "FRAME") {
        try {
          const frameDoc = node.contentDocument || (node.contentWindow && node.contentWindow.document);
          if (frameDoc) {
            const found = findDeepElement(frameDoc, selector, maxDepth, currentDepth + 1, seenRoots);
            if (found) return found;
          }
        } catch (_) {}
      }
    }
  } catch (_) {}

  return null;
}

function resolveElement(sel, elemId) {
  if (elemId == null) {
    let el = null;
    try {
      el = document.querySelector(sel);
    } catch (_) {}
    if (!el) {
      el = findDeepElement(document, sel);
    }
    if (!el) {
      return {
        element: null,
        errorResult: {
          success: false,
          error: `Element not found for selector: '${sel}'`,
          remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
          allowed_actions: ["get_page_snapshot", "scroll_page"],
        },
      };
    }
    return { element: el, errorResult: null };
  }

  // Fast-Path: Query [data-zelyr-id="${elemId}"]
  const fastSelector = `[data-zelyr-id="${elemId}"]`;
  let fastNode = null;
  try {
    fastNode = document.querySelector(fastSelector);
  } catch (_) {}
  if (!fastNode) {
    fastNode = findDeepElement(document, fastSelector);
  }
  if (fastNode && fastNode.isConnected) {
    return { element: fastNode, errorResult: null };
  }

  // Stale / Detached Fallback:
  const fingerprints = (typeof window !== "undefined" && window.__zelyrFingerprints) ? window.__zelyrFingerprints : {};
  const fp = fingerprints[elemId];
  if (!fp) {
    return {
      element: null,
      errorResult: {
        success: false,
        error: `Element ID ${elemId} not found on page.`,
        remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
        allowed_actions: ["get_page_snapshot", "scroll_page"],
      },
    };
  }

  const matchCandidate = (c) => {
    if (!c || !c.isConnected) return false;

    // Visibility check
    if (!isElementVisible(c)) {
      return false;
    }

    // 1. Tag
    if (fp.tag && c.tagName && c.tagName.toLowerCase() !== fp.tag.toLowerCase()) return false;
    // 2. Role
    const cRole = (typeof c.getAttribute === "function" ? c.getAttribute("role") : c.role) || "";
    if (fp.role && cRole !== fp.role) return false;
    // 3. Type
    const cType = (typeof c.getAttribute === "function" ? c.getAttribute("type") : null) || c.type || "";
    if (fp.type && cType !== fp.type) return false;
    // 4. Name
    const cName = (typeof c.getAttribute === "function" ? c.getAttribute("name") : null) || c.name || "";
    if (fp.name && cName !== fp.name) return false;

    // 5. Matching textSnippet or ariaLabel
    const cAria = (
      (typeof c.getAttribute === "function" ? (c.getAttribute("aria-label") || c.getAttribute("aria-labelledby")) : null) || ""
    ).trim().slice(0, 40);
    const cText = (c.innerText || c.textContent || "").trim().slice(0, 40);

    if (fp.ariaLabel && fp.textSnippet) {
      if (cAria !== fp.ariaLabel && cText !== fp.textSnippet) return false;
    } else if (fp.ariaLabel) {
      if (cAria !== fp.ariaLabel) return false;
    } else if (fp.textSnippet) {
      if (cText !== fp.textSnippet) return false;
    }

    return true;
  };

  const findMatches = () => {
    const extra = (fp && fp.tag) ? fp.tag : "";
    const nodes = collectInteractiveElements(document, 10, 0, new Set(), extra);
    return nodes.filter(matchCandidate);
  };

  let matches = findMatches();

  if (matches.length === 1) {
    const matchedEl = matches[0];
    matchedEl.setAttribute("data-zelyr-id", String(elemId));
    return { element: matchedEl, errorResult: null };
  }

  if (matches.length > 1) {
    return {
      element: null,
      errorResult: {
        success: false,
        error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${elemId}.`,
        remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
        allowed_actions: ["get_page_snapshot"],
      },
    };
  }

  // 0 matches found: Trigger exactly one internal bounded retry: take an inline snapshot refresh, re-attempt fingerprint match once.
  try {
    const freshCandidates = collectInteractiveElements(document);
    let freshIdx = 1;
    for (const fel of freshCandidates) {
      if (freshIdx > 50) break;
      fel.setAttribute("data-zelyr-id", String(freshIdx));
      const fTag = (fel.tagName || "").toLowerCase();
      const fRole = (typeof fel.getAttribute === "function" ? fel.getAttribute("role") : fel.role) || "";
      const fType = (typeof fel.getAttribute === "function" ? fel.getAttribute("type") : null) || fel.type || "";
      const fName = (typeof fel.getAttribute === "function" ? fel.getAttribute("name") : null) || fel.name || "";
      const fAria = (typeof fel.getAttribute === "function" ? (fel.getAttribute("aria-label") || fel.getAttribute("aria-labelledby")) : null) || "";
      const fText = (fel.innerText || fel.textContent || "").trim().slice(0, 40);
      if (typeof window !== "undefined") {
        window.__zelyrFingerprints = window.__zelyrFingerprints || {};
        window.__zelyrFingerprints[freshIdx] = {
          tag: fTag,
          role: fRole,
          type: fType,
          name: fName,
          ariaLabel: fAria,
          textSnippet: fText,
        };
      }
      freshIdx++;
    }
  } catch (_) {}

  matches = findMatches();

  if (matches.length === 1) {
    const matchedEl = matches[0];
    matchedEl.setAttribute("data-zelyr-id", String(elemId));
    return { element: matchedEl, errorResult: null };
  }

  if (matches.length > 1) {
    return {
      element: null,
      errorResult: {
        success: false,
        error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${elemId}.`,
        remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
        allowed_actions: ["get_page_snapshot"],
      },
    };
  }

  return {
    element: null,
    errorResult: {
      success: false,
      error: `Element ID ${elemId} not found on page.`,
      remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
      allowed_actions: ["get_page_snapshot", "scroll_page"],
    },
  };
}

async function handleCommand(msg) {
  const { id, action } = msg;

  try {
    switch (action) {
      case "ping": {
        sendReply({ id: msg.id, action: "pong", success: true });
        return;
      }

      case "list_tabs":
      case "tabs": {
        const tabs = await chrome.tabs.query({});
        const tabList = tabs.map((t) => ({
          id: t.id,
          title: t.title || "",
          url: t.url || "",
          active: Boolean(t.active),
          windowId: t.windowId,
        }));
        sendReply({
          id,
          success: true,
          action: "list_tabs",
          tabs: tabList,
          count: tabList.length,
        });
        break;
      }

      case "switch_tab": {
        const query = (msg.query || msg.filter || msg.target || msg.title || msg.url || msg.text || "").toLowerCase().trim();
        const tabId = msg.tab_id ? Number(msg.tab_id) : null;
        const allTabs = await chrome.tabs.query({});

        let matched = null;
        if (tabId) {
          matched = allTabs.find((t) => t.id === tabId);
        } else if (query) {
          matched = allTabs.find(
            (t) =>
              (t.title && t.title.toLowerCase().includes(query)) ||
              (t.url && t.url.toLowerCase().includes(query))
          );
        }

        if (!matched) {
          const openTabs = allTabs.map((t) => ({
            id: t.id,
            title: t.title || "",
            url: t.url || "",
            active: Boolean(t.active),
          }));
          sendReply({
            id,
            success: false,
            action: "switch_tab",
            error: `No open tab matching query: '${query || tabId}'`,
            remedy_hint: `Call list_tabs to see valid targets, or open_tab with url='${query || "<url>"}' to launch it.`,
            allowed_actions: ["list_tabs", "open_tab"],
            open_tabs: openTabs,
          });
          return;
        }

        await chrome.tabs.update(matched.id, { active: true });
        if (matched.windowId && chrome.windows) {
          try {
            await chrome.windows.update(matched.windowId, { focused: true });
          } catch (_) {}
        }

        sendReply({
          id,
          success: true,
          action: "switch_tab",
          tab: {
            id: matched.id,
            title: matched.title || "",
            url: matched.url || "",
          },
        });
        break;
      }

      case "open_tab": {
        let url = (msg.url || msg.query || "about:blank").trim();
        if (url && !url.includes("://") && !url.startsWith("about:") && !url.startsWith("chrome:")) {
          url = `https://${url}`;
        }
        try {
          const tab = await chrome.tabs.create({ url, active: true });
          sendReply({
            id,
            success: true,
            action: "open_tab",
            tab_id: tab.id,
            title: tab.title || "",
            url: tab.url || url,
          });
        } catch (err) {
          sendReply({
            id,
            success: false,
            action: "open_tab",
            error: `Failed to open tab: ${err.message || String(err)}`,
          });
        }
        break;
      }

      case "navigate_url": {
        let url = (msg.url || msg.query || msg.target || "").trim();
        if (!url) {
          sendReply({ id, success: false, action: "navigate_url", error: "Missing url parameter" });
          return;
        }
        if (!url.includes("://") && !url.startsWith("about:") && !url.startsWith("chrome:")) {
          url = `https://${url}`;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "navigate_url", error: "No target tab available to navigate" });
          return;
        }

        try {
          await chrome.tabs.update(tab.id, { url });

          // Await tab status === "complete" or a 4-second timeout
          await new Promise((resolve) => {
            let timer = null;
            let listener = null;

            const cleanup = () => {
              if (timer) clearTimeout(timer);
              if (listener && chrome.tabs && chrome.tabs.onUpdated && typeof chrome.tabs.onUpdated.removeListener === "function") {
                try { chrome.tabs.onUpdated.removeListener(listener); } catch (_) {}
              }
            };

            timer = setTimeout(() => {
              cleanup();
              resolve();
            }, 4000);

            listener = (updatedTabId, changeInfo) => {
              if (updatedTabId === tab.id && changeInfo.status === "complete") {
                cleanup();
                resolve();
              }
            };

            if (chrome.tabs && chrome.tabs.onUpdated && typeof chrome.tabs.onUpdated.addListener === "function") {
              try {
                chrome.tabs.onUpdated.addListener(listener);
              } catch (_) {
                cleanup();
                resolve();
              }
            } else {
              cleanup();
              resolve();
            }
          });

          // Followed by waitForSettled in the page
          try {
            await chrome.scripting.executeScript({
              target: { tabId: tab.id },
              func: async () => {
                function waitForSettled(options = {}) {
                  const quietWindow = options.quietWindow || 75;
                  const maxTimeout = options.maxTimeout || 800;
                  return new Promise((resolve) => {
                    let timer = null;
                    let maxTimer = null;
                    let observer = null;
                    const cleanup = () => {
                      if (timer) clearTimeout(timer);
                      if (maxTimer) clearTimeout(maxTimer);
                      if (observer) {
                        try { observer.disconnect(); } catch (e) {}
                        observer = null;
                      }
                    };
                    const done = () => {
                      cleanup();
                      const raf = (typeof requestAnimationFrame === "function") ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
                      raf(() => {
                        raf(() => resolve());
                      });
                    };
                    try {
                      const target = (typeof document !== "undefined") ? (document.body || document.documentElement) : null;
                      if (!target || typeof MutationObserver === "undefined") {
                        done();
                        return;
                      }
                      observer = new MutationObserver(() => {
                        clearTimeout(timer);
                        timer = setTimeout(done, quietWindow);
                      });
                      observer.observe(target, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        characterData: true,
                      });
                    } catch (e) {
                      done();
                      return;
                    }
                    timer = setTimeout(done, quietWindow);
                    maxTimer = setTimeout(done, maxTimeout);
                  });
                }
                await waitForSettled();
              },
            });
          } catch (_) {}

          let updatedTab = null;
          try {
            updatedTab = await chrome.tabs.get(tab.id);
          } catch (_) {}

          sendReply({
            id,
            success: true,
            action: "navigate_url",
            tab_id: tab.id,
            title: (updatedTab && updatedTab.title) || tab.title || "",
            url: (updatedTab && updatedTab.url) || url,
          });
        } catch (err) {
          sendReply({
            id,
            success: false,
            action: "navigate_url",
            error: `Failed to navigate tab: ${err.message || String(err)}`,
          });
        }
        break;
      }

      case "capture_tab_screenshot": {
        try {
          if (!chrome.tabs || typeof chrome.tabs.captureVisibleTab !== "function") {
            sendReply({
              id,
              success: false,
              action: "capture_tab_screenshot",
              error: "captureVisibleTab API is not supported in this browser context",
            });
            return;
          }

          const capturePromise = new Promise((resolve, reject) => {
            try {
              const res = chrome.tabs.captureVisibleTab(null, { format: "png" }, (result) => {
                if (chrome.runtime && chrome.runtime.lastError) {
                  reject(new Error(chrome.runtime.lastError.message));
                } else if (result) {
                  resolve(result);
                }
              });
              if (res && typeof res.then === "function") {
                res.then(resolve).catch(reject);
              }
            } catch (e) {
              reject(e);
            }
          });

          const dataUrl = await capturePromise;

          sendReply({
            id,
            success: true,
            action: "capture_tab_screenshot",
            format: "png",
            data_url: dataUrl,
            data_url_prefix: dataUrl ? dataUrl.slice(0, 30) : "",
            length: dataUrl ? dataUrl.length : 0,
          });
        } catch (err) {
          sendReply({
            id,
            success: false,
            action: "capture_tab_screenshot",
            error: `Failed to capture tab screenshot: ${err.message || String(err)}`,
          });
        }
        break;
      }

      case "get_active_tab": {
        try {
          let activeTabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
          if (!activeTabs || activeTabs.length === 0) {
            activeTabs = await chrome.tabs.query({ active: true });
          }
          if (!activeTabs || activeTabs.length === 0) {
            activeTabs = await chrome.tabs.query({});
          }

          if (activeTabs && activeTabs.length > 0) {
            const tab = activeTabs[0];
            sendReply({
              id,
              success: true,
              action: "get_active_tab",
              tab_id: tab.id,
              title: tab.title || "",
              url: tab.url || "",
              active: Boolean(tab.active),
            });
          } else {
            sendReply({
              id,
              success: false,
              action: "get_active_tab",
              error: "No active tab found",
            });
          }
        } catch (err) {
          sendReply({
            id,
            success: false,
            action: "get_active_tab",
            error: `Failed to get active tab: ${err.message || String(err)}`,
          });
        }
        break;
      }

      case "close_tab": {
        const tabId = msg.tab_id ? Number(msg.tab_id) : null;
        const query = (msg.query || msg.filter || msg.target || msg.title || msg.url || "").toLowerCase().trim();
        try {
          const allTabs = await chrome.tabs.query({});
          let target = null;
          if (tabId) {
            target = allTabs.find((t) => t.id === tabId);
          } else if (query) {
            target = allTabs.find(
              (t) =>
                (t.title && t.title.toLowerCase().includes(query)) ||
                (t.url && t.url.toLowerCase().includes(query))
            );
          } else {
            const activeTabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
            if (activeTabs && activeTabs.length > 0) {
              target = activeTabs[0];
            }
          }

          if (!target || !target.id) {
            sendReply({
              id,
              success: false,
              action: "close_tab",
              error: `No tab found to close matching '${query || tabId || "active"}'`,
            });
            return;
          }

          await chrome.tabs.remove(target.id);
          sendReply({
            id,
            success: true,
            action: "close_tab",
            closed_tab_id: target.id,
            title: target.title || "",
            url: target.url || "",
          });
        } catch (err) {
          sendReply({
            id,
            success: false,
            action: "close_tab",
            error: `Failed to close tab: ${err.message || String(err)}`,
          });
        }
        break;
      }

      case "click": {
        const selector = msg.selector;
        if (!selector) {
          sendReply({ id, success: false, action: "click", error: "Missing selector parameter" });
          return;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "click", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: (sel) => {
            const el = document.querySelector(sel);
            if (!el) {
              return { success: false, error: `Element not found for selector: '${sel}'` };
            }
            try {
              el.scrollIntoView({ behavior: "instant", block: "center" });
            } catch (_) {}
            el.click();
            return {
              success: true,
              result: `Clicked <${el.tagName.toLowerCase()}> element (${sel})`,
              tagName: el.tagName,
              id: el.id,
            };
          },
          args: [selector],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Script execution produced no result" };
        sendReply({ id, action: "click", selector, ...execResult });
        break;
      }

      case "type": {
        const selector = msg.selector;
        const text = msg.text || "";
        if (!selector) {
          sendReply({ id, success: false, action: "type", error: "Missing selector parameter" });
          return;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "type", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: (sel, val) => {
            const el = document.querySelector(sel);
            if (!el) {
              return { success: false, error: `Element not found for selector: '${sel}'` };
            }
            try {
              el.focus();
            } catch (_) {}
            if ("value" in el) {
              el.value = val;
              el.dispatchEvent(new Event("input", { bubbles: true }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
            } else {
              el.textContent = val;
            }
            return {
              success: true,
              result: `Typed '${val}' into <${el.tagName.toLowerCase()}>`,
              value: val,
            };
          },
          args: [selector, text],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Script execution produced no result" };
        sendReply({ id, action: "type", selector, ...execResult });
        break;
      }

      case "get_dom":
      case "get_content": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "get_dom", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            return {
              title: document.title,
              url: window.location.href,
              content: document.body ? document.body.innerText.slice(0, 50000) : "",
            };
          },
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Script execution produced no result" };
        sendReply({ id, success: true, action: "get_dom", ...execResult });
        break;
      }

      case "get_page_snapshot": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "get_page_snapshot", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            function isElementVisible(el) {
              if (!el) return false;
              if (typeof el.getBoundingClientRect === "function" && typeof window !== "undefined" && typeof window.getComputedStyle === "function") {
                try {
                  const style = window.getComputedStyle(el);
                  if (style) {
                    if (
                      style.display === "none" ||
                      style.visibility === "hidden" ||
                      parseFloat(style.opacity || "1") === 0
                    ) {
                      return false;
                    }
                  }
                  const rect = el.getBoundingClientRect();
                  if (rect && rect.width === 0 && rect.height === 0 && el.offsetParent === null) {
                    const root = typeof el.getRootNode === "function" ? el.getRootNode() : null;
                    if (!root || root === document) {
                      return false;
                    }
                  }
                } catch (_) {}
              }
              return true;
            }

            function clearOldLabels(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return;
              seenRoots.add(root);
              try {
                const old = Array.from(root.querySelectorAll("[data-zelyr-id]"));
                for (const el of old) el.removeAttribute("data-zelyr-id");
              } catch (_) {}
              try {
                const all = Array.from(root.querySelectorAll("*"));
                for (const el of all) {
                  if (el.shadowRoot && el.shadowRoot.mode === "open") {
                    clearOldLabels(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots);
                  } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                    try {
                      const doc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                      if (doc) clearOldLabels(doc, maxDepth, currentDepth + 1, seenRoots);
                    } catch (_) {}
                  }
                }
              } catch (_) {}
            }

            function collectInteractiveElements(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set(), extraSelector = "") {
              const interactive = [];
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return interactive;
              seenRoots.add(root);

              const baseSelector = [
                "a[href]",
                "button",
                "input",
                "textarea",
                "select",
                '[role="button"]',
                '[role="link"]',
                '[role="checkbox"]',
                '[role="tab"]',
                '[role="menuitem"]',
                "[onclick]",
                '[tabindex]:not([tabindex="-1"])',
              ].join(", ");

              const selector = extraSelector ? `${baseSelector}, ${extraSelector}` : baseSelector;

              try {
                const nodes = Array.from(root.querySelectorAll(selector));
                for (const el of nodes) {
                  if (isElementVisible(el)) {
                    interactive.push(el);
                  }
                }
              } catch (_) {}

              try {
                const allElements = Array.from(root.querySelectorAll("*"));
                for (const el of allElements) {
                  if (el.shadowRoot && el.shadowRoot.mode === "open") {
                    const shadowChildren = collectInteractiveElements(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                    for (const child of shadowChildren) {
                      interactive.push(child);
                    }
                  } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                    try {
                      const frameDoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                      if (frameDoc && (frameDoc.body || frameDoc.documentElement)) {
                        const frameChildren = collectInteractiveElements(frameDoc, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                        for (const child of frameChildren) {
                          interactive.push(child);
                        }
                      }
                    } catch (e) {
                      // Cross-origin SecurityError - ignore safely
                    }
                  }
                }
              } catch (_) {}

              return interactive;
            }

            // Clean up any old labels first
            clearOldLabels(document);

            window.__zelyrFingerprints = {};

            const candidates = collectInteractiveElements(document);
            const items = [];
            let index = 1;

            for (const el of candidates) {
              if (index > 50) break; // Capped at 50 most relevant elements

              if (!isElementVisible(el)) {
                continue;
              }

              // Tag element with transient attribute
              el.setAttribute("data-zelyr-id", String(index));

              const tag = (el.tagName || "").toLowerCase();
              const role = (typeof el.getAttribute === "function" ? el.getAttribute("role") : el.role) || "";
              const type = (typeof el.getAttribute === "function" ? el.getAttribute("type") : null) || el.type || "";
              const name = (typeof el.getAttribute === "function" ? el.getAttribute("name") : null) || el.name || "";
              const ariaLabel = (typeof el.getAttribute === "function" ? (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby")) : null) || "";
              const textSnippet = (el.innerText || el.textContent || "").trim().slice(0, 40);

              window.__zelyrFingerprints[index] = {
                tag: tag,
                role: role,
                type: type,
                name: name,
                ariaLabel: ariaLabel,
                textSnippet: textSnippet,
              };

              const item = {
                id: index,
                tag: tag,
              };

              if (tag === "input" || tag === "textarea") {
                if (type) item.type = type;
                if (el.placeholder) item.placeholder = el.placeholder.slice(0, 100);
                if (el.value) item.value = el.value.slice(0, 100);
                if (name) item.name = name;
              }

              if (ariaLabel) item.aria_label = ariaLabel.trim().slice(0, 100);

              const text = (el.innerText || el.textContent || "").trim().replace(/\s+/g, " ");
              if (text && tag !== "input") {
                item.text = text.slice(0, 100);
              }

              if (tag === "a") {
                const href = (typeof el.getAttribute === "function" ? el.getAttribute("href") : el.href) || "";
                if (href) item.href = href.slice(0, 120);
              }
              if (role) item.role = role;

              items.push(item);
              index++;
            }

            const summaryLines = items.map((it) => {
              const parts = [`[${it.id}] <${it.tag}`];
              if (it.type) parts.push(`type="${it.type}"`);
              if (it.placeholder) parts.push(`placeholder="${it.placeholder}"`);
              if (it.name) parts.push(`name="${it.name}"`);
              if (it.role) parts.push(`role="${it.role}"`);
              parts.push(">");
              if (it.text) parts.push(`"${it.text}"`);
              if (it.value) parts.push(`value="${it.value}"`);
              if (it.aria_label) parts.push(`(aria: "${it.aria_label}")`);
              if (it.href) parts.push(`(href: ${it.href})`);
              return parts.join(" ");
            });

            // Note any restricted cross-origin frames
            try {
              const allFrames = Array.from(document.querySelectorAll("iframe, frame"));
              for (const f of allFrames) {
                try {
                  const doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
                  if (!doc) {
                    summaryLines.push("[iframe: cross-origin restricted]");
                  }
                } catch (_) {
                  summaryLines.push("[iframe: cross-origin restricted]");
                }
              }
            } catch (_) {}

            return {
              title: document.title,
              url: window.location.href,
              elements: items,
              summary: summaryLines.join("\n"),
              count: items.length,
            };
          },
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Snapshot script produced no result" };
        sendReply({ id, success: true, action: "get_page_snapshot", ...execResult });
        break;
      }

      case "click_element": {
        const elementId = msg.element_id != null ? Number(msg.element_id) : null;
        const selector = msg.selector || (elementId != null ? `[data-zelyr-id="${elementId}"]` : null);
        if (!selector) {
          sendReply({ id, success: false, action: "click_element", error: "Missing element_id or selector parameter" });
          return;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "click_element", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: async (sel, elemId) => {
            function waitForSettled(options = {}) {
              const quietWindow = options.quietWindow || 75;
              const maxTimeout = options.maxTimeout || 800;
              return new Promise((resolve) => {
                let timer = null;
                let maxTimer = null;
                let observer = null;
                const cleanup = () => {
                  if (timer) clearTimeout(timer);
                  if (maxTimer) clearTimeout(maxTimer);
                  if (observer) {
                    try { observer.disconnect(); } catch (e) {}
                    observer = null;
                  }
                };
                const done = () => {
                  cleanup();
                  const raf = (typeof requestAnimationFrame === "function") ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
                  raf(() => {
                    raf(() => resolve());
                  });
                };
                try {
                  const target = (typeof document !== "undefined") ? (document.body || document.documentElement) : null;
                  if (!target || typeof MutationObserver === "undefined") {
                    done();
                    return;
                  }
                  observer = new MutationObserver(() => {
                    clearTimeout(timer);
                    timer = setTimeout(done, quietWindow);
                  });
                  observer.observe(target, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    characterData: true,
                  });
                } catch (e) {
                  done();
                  return;
                }
                timer = setTimeout(done, quietWindow);
                maxTimer = setTimeout(done, maxTimeout);
              });
            }

            function isElementVisible(el) {
              if (!el) return false;
              if (typeof el.getBoundingClientRect === "function" && typeof window !== "undefined" && typeof window.getComputedStyle === "function") {
                try {
                  const style = window.getComputedStyle(el);
                  if (style) {
                    if (
                      style.display === "none" ||
                      style.visibility === "hidden" ||
                      parseFloat(style.opacity || "1") === 0
                    ) {
                      return false;
                    }
                  }
                  const rect = el.getBoundingClientRect();
                  if (rect && rect.width === 0 && rect.height === 0 && el.offsetParent === null) {
                    const root = typeof el.getRootNode === "function" ? el.getRootNode() : null;
                    if (!root || root === document) {
                      return false;
                    }
                  }
                } catch (_) {}
              }
              return true;
            }

            function collectInteractiveElements(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set(), extraSelector = "") {
              const interactive = [];
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return interactive;
              seenRoots.add(root);

              const baseSelector = [
                "a[href]",
                "button",
                "input",
                "textarea",
                "select",
                '[role="button"]',
                '[role="link"]',
                '[role="checkbox"]',
                '[role="tab"]',
                '[role="menuitem"]',
                "[onclick]",
                '[tabindex]:not([tabindex="-1"])',
              ].join(", ");

              const selector = extraSelector ? `${baseSelector}, ${extraSelector}` : baseSelector;

              try {
                const nodes = Array.from(root.querySelectorAll(selector));
                for (const el of nodes) {
                  if (isElementVisible(el)) {
                    interactive.push(el);
                  }
                }
              } catch (_) {}

              try {
                const allElements = Array.from(root.querySelectorAll("*"));
                for (const el of allElements) {
                  if (el.shadowRoot && el.shadowRoot.mode === "open") {
                    const shadowChildren = collectInteractiveElements(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                    for (const child of shadowChildren) {
                      interactive.push(child);
                    }
                  } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                    try {
                      const frameDoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                      if (frameDoc && (frameDoc.body || frameDoc.documentElement)) {
                        const frameChildren = collectInteractiveElements(frameDoc, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                        for (const child of frameChildren) {
                          interactive.push(child);
                        }
                      }
                    } catch (e) {
                      // Cross-origin SecurityError - ignore safely
                    }
                  }
                }
              } catch (_) {}

              return interactive;
            }

            function findDeepElement(root, selector, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return null;
              seenRoots.add(root);

              try {
                const el = root.querySelector(selector);
                if (el) return el;
              } catch (_) {}

              try {
                const all = Array.from(root.querySelectorAll("*"));
                for (const node of all) {
                  if (node.shadowRoot && node.shadowRoot.mode === "open") {
                    const found = findDeepElement(node.shadowRoot, selector, maxDepth, currentDepth + 1, seenRoots);
                    if (found) return found;
                  } else if (node.tagName === "IFRAME" || node.tagName === "FRAME") {
                    try {
                      const frameDoc = node.contentDocument || (node.contentWindow && node.contentWindow.document);
                      if (frameDoc) {
                        const found = findDeepElement(frameDoc, selector, maxDepth, currentDepth + 1, seenRoots);
                        if (found) return found;
                      }
                    } catch (_) {}
                  }
                }
              } catch (_) {}

              return null;
            }

            function resolveElement(targetSel, targetElemId) {
              if (targetElemId == null) {
                let el = null;
                try {
                  el = document.querySelector(targetSel);
                } catch (_) {}
                if (!el) {
                  el = findDeepElement(document, targetSel);
                }
                if (!el) {
                  return {
                    element: null,
                    errorResult: {
                      success: false,
                      error: `Element not found for selector: '${targetSel}'`,
                      remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                      allowed_actions: ["get_page_snapshot", "scroll_page"],
                    },
                  };
                }
                return { element: el, errorResult: null };
              }

              // Fast-Path: Query [data-zelyr-id="${targetElemId}"]
              const fastSelector = `[data-zelyr-id="${targetElemId}"]`;
              let fastNode = null;
              try {
                fastNode = document.querySelector(fastSelector);
              } catch (_) {}
              if (!fastNode) {
                fastNode = findDeepElement(document, fastSelector);
              }
              if (fastNode && fastNode.isConnected) {
                return { element: fastNode, errorResult: null };
              }

              // Stale / Detached Fallback:
              const fingerprints = (typeof window !== "undefined" && window.__zelyrFingerprints) ? window.__zelyrFingerprints : {};
              const fp = fingerprints[targetElemId];
              if (!fp) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Element ID ${targetElemId} not found on page.`,
                    remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                    allowed_actions: ["get_page_snapshot", "scroll_page"],
                  },
                };
              }

              const matchCandidate = (c) => {
                if (!c || !c.isConnected) return false;
                if (!isElementVisible(c)) {
                  return false;
                }

                if (fp.tag && c.tagName && c.tagName.toLowerCase() !== fp.tag.toLowerCase()) return false;
                const cRole = (typeof c.getAttribute === "function" ? c.getAttribute("role") : c.role) || "";
                if (fp.role && cRole !== fp.role) return false;
                const cType = (typeof c.getAttribute === "function" ? c.getAttribute("type") : null) || c.type || "";
                if (fp.type && cType !== fp.type) return false;
                const cName = (typeof c.getAttribute === "function" ? c.getAttribute("name") : null) || c.name || "";
                if (fp.name && cName !== fp.name) return false;

                const cAria = (
                  (typeof c.getAttribute === "function" ? (c.getAttribute("aria-label") || c.getAttribute("aria-labelledby")) : null) || ""
                ).trim().slice(0, 40);
                const cText = (c.innerText || c.textContent || "").trim().slice(0, 40);

                if (fp.ariaLabel && fp.textSnippet) {
                  if (cAria !== fp.ariaLabel && cText !== fp.textSnippet) return false;
                } else if (fp.ariaLabel) {
                  if (cAria !== fp.ariaLabel) return false;
                } else if (fp.textSnippet) {
                  if (cText !== fp.textSnippet) return false;
                }

                return true;
              };

              const findMatches = () => {
                const extra = (fp && fp.tag) ? fp.tag : "";
                const nodes = collectInteractiveElements(document, 10, 0, new Set(), extra);
                return nodes.filter(matchCandidate);
              };

              let matches = findMatches();
              if (matches.length === 1) {
                matches[0].setAttribute("data-zelyr-id", String(targetElemId));
                return { element: matches[0], errorResult: null };
              }

              if (matches.length > 1) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${targetElemId}.`,
                    remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
                    allowed_actions: ["get_page_snapshot"],
                  },
                };
              }

              // 0 matches found: Trigger exactly one internal bounded retry
              try {
                const freshCandidates = collectInteractiveElements(document);
                let freshIdx = 1;
                for (const fel of freshCandidates) {
                  if (freshIdx > 50) break;
                  fel.setAttribute("data-zelyr-id", String(freshIdx));
                  const fTag = (fel.tagName || "").toLowerCase();
                  const fRole = (typeof fel.getAttribute === "function" ? fel.getAttribute("role") : fel.role) || "";
                  const fType = (typeof fel.getAttribute === "function" ? fel.getAttribute("type") : null) || fel.type || "";
                  const fName = (typeof fel.getAttribute === "function" ? fel.getAttribute("name") : null) || fel.name || "";
                  const fAria = (typeof fel.getAttribute === "function" ? (fel.getAttribute("aria-label") || fel.getAttribute("aria-labelledby")) : null) || "";
                  const fText = (fel.innerText || fel.textContent || "").trim().slice(0, 40);
                  if (typeof window !== "undefined") {
                    window.__zelyrFingerprints = window.__zelyrFingerprints || {};
                    window.__zelyrFingerprints[freshIdx] = {
                      tag: fTag,
                      role: fRole,
                      type: fType,
                      name: fName,
                      ariaLabel: fAria,
                      textSnippet: fText,
                    };
                  }
                  freshIdx++;
                }
              } catch (_) {}

              matches = findMatches();
              if (matches.length === 1) {
                matches[0].setAttribute("data-zelyr-id", String(targetElemId));
                return { element: matches[0], errorResult: null };
              }

              if (matches.length > 1) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${targetElemId}.`,
                    remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
                    allowed_actions: ["get_page_snapshot"],
                  },
                };
              }

              return {
                element: null,
                errorResult: {
                  success: false,
                  error: `Element ID ${targetElemId} not found on page.`,
                  remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                  allowed_actions: ["get_page_snapshot", "scroll_page"],
                },
              };
            }

            const { element: el, errorResult } = resolveElement(sel, elemId);
            if (!el) {
              return errorResult;
            }

            try {
              el.scrollIntoView({ behavior: "instant", block: "center" });
            } catch (_) {}
            try {
              el.focus();
            } catch (_) {}

            // Visual Action Highlighting: 400ms high-contrast outline
            try {
              const origOutline = el.style.outline;
              const origBoxShadow = el.style.boxShadow;
              el.style.outline = "3px solid #7aa2f7";
              el.style.boxShadow = "0 0 10px rgba(122, 162, 247, 0.8)";
              setTimeout(() => {
                try {
                  el.style.outline = origOutline || "";
                  el.style.boxShadow = origBoxShadow || "";
                } catch (_) {}
              }, 400);
            } catch (_) {}

            el.click();

            // Await DOM settlement before completing
            await waitForSettled();

            return {
              success: true,
              clicked_id: elemId != null ? elemId : (el.getAttribute("data-zelyr-id") ? Number(el.getAttribute("data-zelyr-id")) : null),
              tag: el.tagName,
              text: (el.innerText || el.textContent || "").trim().slice(0, 100),
            };
          },
          args: [selector, elementId],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Click script execution produced no result" };
        sendReply({ id, action: "click_element", ...execResult });
        break;
      }

      case "fill_element": {
        const elementId = msg.element_id != null ? Number(msg.element_id) : null;
        const selector = msg.selector || (elementId != null ? `[data-zelyr-id="${elementId}"]` : null);
        const text = msg.text || "";
        const submit = Boolean(msg.submit);

        if (!selector) {
          sendReply({ id, success: false, action: "fill_element", error: "Missing element_id or selector parameter" });
          return;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "fill_element", error: "No target tab available" });
          return;
        }

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: async (sel, val, doSubmit, elemId) => {
            function waitForSettled(options = {}) {
              const quietWindow = options.quietWindow || 75;
              const maxTimeout = options.maxTimeout || 800;
              return new Promise((resolve) => {
                let timer = null;
                let maxTimer = null;
                let observer = null;
                const cleanup = () => {
                  if (timer) clearTimeout(timer);
                  if (maxTimer) clearTimeout(maxTimer);
                  if (observer) {
                    try { observer.disconnect(); } catch (e) {}
                    observer = null;
                  }
                };
                const done = () => {
                  cleanup();
                  const raf = (typeof requestAnimationFrame === "function") ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
                  raf(() => {
                    raf(() => resolve());
                  });
                };
                try {
                  const target = (typeof document !== "undefined") ? (document.body || document.documentElement) : null;
                  if (!target || typeof MutationObserver === "undefined") {
                    done();
                    return;
                  }
                  observer = new MutationObserver(() => {
                    clearTimeout(timer);
                    timer = setTimeout(done, quietWindow);
                  });
                  observer.observe(target, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    characterData: true,
                  });
                } catch (e) {
                  done();
                  return;
                }
                timer = setTimeout(done, quietWindow);
                maxTimer = setTimeout(done, maxTimeout);
              });
            }

            function isElementVisible(el) {
              if (!el) return false;
              if (typeof el.getBoundingClientRect === "function" && typeof window !== "undefined" && typeof window.getComputedStyle === "function") {
                try {
                  const style = window.getComputedStyle(el);
                  if (style) {
                    if (
                      style.display === "none" ||
                      style.visibility === "hidden" ||
                      parseFloat(style.opacity || "1") === 0
                    ) {
                      return false;
                    }
                  }
                  const rect = el.getBoundingClientRect();
                  if (rect && rect.width === 0 && rect.height === 0 && el.offsetParent === null) {
                    const root = typeof el.getRootNode === "function" ? el.getRootNode() : null;
                    if (!root || root === document) {
                      return false;
                    }
                  }
                } catch (_) {}
              }
              return true;
            }

            function collectInteractiveElements(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set(), extraSelector = "") {
              const interactive = [];
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return interactive;
              seenRoots.add(root);

              const baseSelector = [
                "a[href]",
                "button",
                "input",
                "textarea",
                "select",
                '[role="button"]',
                '[role="link"]',
                '[role="checkbox"]',
                '[role="tab"]',
                '[role="menuitem"]',
                "[onclick]",
                '[tabindex]:not([tabindex="-1"])',
              ].join(", ");

              const selector = extraSelector ? `${baseSelector}, ${extraSelector}` : baseSelector;

              try {
                const nodes = Array.from(root.querySelectorAll(selector));
                for (const el of nodes) {
                  if (isElementVisible(el)) {
                    interactive.push(el);
                  }
                }
              } catch (_) {}

              try {
                const allElements = Array.from(root.querySelectorAll("*"));
                for (const el of allElements) {
                  if (el.shadowRoot && el.shadowRoot.mode === "open") {
                    const shadowChildren = collectInteractiveElements(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                    for (const child of shadowChildren) {
                      interactive.push(child);
                    }
                  } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                    try {
                      const frameDoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                      if (frameDoc && (frameDoc.body || frameDoc.documentElement)) {
                        const frameChildren = collectInteractiveElements(frameDoc, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                        for (const child of frameChildren) {
                          interactive.push(child);
                        }
                      }
                    } catch (e) {
                      // Cross-origin SecurityError - ignore safely
                    }
                  }
                }
              } catch (_) {}

              return interactive;
            }

            function findDeepElement(root, selector, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
              if (!root || currentDepth > maxDepth || seenRoots.has(root)) return null;
              seenRoots.add(root);

              try {
                const el = root.querySelector(selector);
                if (el) return el;
              } catch (_) {}

              try {
                const all = Array.from(root.querySelectorAll("*"));
                for (const node of all) {
                  if (node.shadowRoot && node.shadowRoot.mode === "open") {
                    const found = findDeepElement(node.shadowRoot, selector, maxDepth, currentDepth + 1, seenRoots);
                    if (found) return found;
                  } else if (node.tagName === "IFRAME" || node.tagName === "FRAME") {
                    try {
                      const frameDoc = node.contentDocument || (node.contentWindow && node.contentWindow.document);
                      if (frameDoc) {
                        const found = findDeepElement(frameDoc, selector, maxDepth, currentDepth + 1, seenRoots);
                        if (found) return found;
                      }
                    } catch (_) {}
                  }
                }
              } catch (_) {}

              return null;
            }

            function resolveElement(targetSel, targetElemId) {
              if (targetElemId == null) {
                let el = null;
                try {
                  el = document.querySelector(targetSel);
                } catch (_) {}
                if (!el) {
                  el = findDeepElement(document, targetSel);
                }
                if (!el) {
                  return {
                    element: null,
                    errorResult: {
                      success: false,
                      error: `Element not found for selector: '${targetSel}'`,
                      remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                      allowed_actions: ["get_page_snapshot", "scroll_page"],
                    },
                  };
                }
                return { element: el, errorResult: null };
              }

              // Fast-Path: Query [data-zelyr-id="${targetElemId}"]
              const fastSelector = `[data-zelyr-id="${targetElemId}"]`;
              let fastNode = null;
              try {
                fastNode = document.querySelector(fastSelector);
              } catch (_) {}
              if (!fastNode) {
                fastNode = findDeepElement(document, fastSelector);
              }
              if (fastNode && fastNode.isConnected) {
                return { element: fastNode, errorResult: null };
              }

              // Stale / Detached Fallback:
              const fingerprints = (typeof window !== "undefined" && window.__zelyrFingerprints) ? window.__zelyrFingerprints : {};
              const fp = fingerprints[targetElemId];
              if (!fp) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Element ID ${targetElemId} not found on page.`,
                    remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                    allowed_actions: ["get_page_snapshot", "scroll_page"],
                  },
                };
              }

              const matchCandidate = (c) => {
                if (!c || !c.isConnected) return false;
                if (!isElementVisible(c)) {
                  return false;
                }

                if (fp.tag && c.tagName && c.tagName.toLowerCase() !== fp.tag.toLowerCase()) return false;
                const cRole = (typeof c.getAttribute === "function" ? c.getAttribute("role") : c.role) || "";
                if (fp.role && cRole !== fp.role) return false;
                const cType = (typeof c.getAttribute === "function" ? c.getAttribute("type") : null) || c.type || "";
                if (fp.type && cType !== fp.type) return false;
                const cName = (typeof c.getAttribute === "function" ? c.getAttribute("name") : null) || c.name || "";
                if (fp.name && cName !== fp.name) return false;

                const cAria = (
                  (typeof c.getAttribute === "function" ? (c.getAttribute("aria-label") || c.getAttribute("aria-labelledby")) : null) || ""
                ).trim().slice(0, 40);
                const cText = (c.innerText || c.textContent || "").trim().slice(0, 40);

                if (fp.ariaLabel && fp.textSnippet) {
                  if (cAria !== fp.ariaLabel && cText !== fp.textSnippet) return false;
                } else if (fp.ariaLabel) {
                  if (cAria !== fp.ariaLabel) return false;
                } else if (fp.textSnippet) {
                  if (cText !== fp.textSnippet) return false;
                }

                return true;
              };

              const findMatches = () => {
                const extra = (fp && fp.tag) ? fp.tag : "";
                const nodes = collectInteractiveElements(document, 10, 0, new Set(), extra);
                return nodes.filter(matchCandidate);
              };

              let matches = findMatches();
              if (matches.length === 1) {
                matches[0].setAttribute("data-zelyr-id", String(targetElemId));
                return { element: matches[0], errorResult: null };
              }

              if (matches.length > 1) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${targetElemId}.`,
                    remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
                    allowed_actions: ["get_page_snapshot"],
                  },
                };
              }

              // 0 matches found: Trigger exactly one internal bounded retry
              try {
                const freshCandidates = collectInteractiveElements(document);
                let freshIdx = 1;
                for (const fel of freshCandidates) {
                  if (freshIdx > 50) break;
                  fel.setAttribute("data-zelyr-id", String(freshIdx));
                  const fTag = (fel.tagName || "").toLowerCase();
                  const fRole = (typeof fel.getAttribute === "function" ? fel.getAttribute("role") : fel.role) || "";
                  const fType = (typeof fel.getAttribute === "function" ? fel.getAttribute("type") : null) || fel.type || "";
                  const fName = (typeof fel.getAttribute === "function" ? fel.getAttribute("name") : null) || fel.name || "";
                  const fAria = (typeof fel.getAttribute === "function" ? (fel.getAttribute("aria-label") || fel.getAttribute("aria-labelledby")) : null) || "";
                  const fText = (fel.innerText || fel.textContent || "").trim().slice(0, 40);
                  if (typeof window !== "undefined") {
                    window.__zelyrFingerprints = window.__zelyrFingerprints || {};
                    window.__zelyrFingerprints[freshIdx] = {
                      tag: fTag,
                      role: fRole,
                      type: fType,
                      name: fName,
                      ariaLabel: fAria,
                      textSnippet: fText,
                    };
                  }
                  freshIdx++;
                }
              } catch (_) {}

              matches = findMatches();
              if (matches.length === 1) {
                matches[0].setAttribute("data-zelyr-id", String(targetElemId));
                return { element: matches[0], errorResult: null };
              }

              if (matches.length > 1) {
                return {
                  element: null,
                  errorResult: {
                    success: false,
                    error: `Ambiguous element resolution: ${matches.length} elements match fingerprint for ID ${targetElemId}.`,
                    remedy_hint: "Call get_page_snapshot to re-index unique element IDs.",
                    allowed_actions: ["get_page_snapshot"],
                  },
                };
              }

              return {
                element: null,
                errorResult: {
                  success: false,
                  error: `Element ID ${targetElemId} not found on page.`,
                  remedy_hint: "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                  allowed_actions: ["get_page_snapshot", "scroll_page"],
                },
              };
            }

            const { element: el, errorResult } = resolveElement(sel, elemId);
            if (!el) {
              return errorResult;
            }

            try {
              el.scrollIntoView({ behavior: "instant", block: "center" });
            } catch (_) {}
            try {
              el.focus();
            } catch (_) {}

            const tag = el.tagName.toLowerCase();
            let currentVal = val;

            if (tag === "select") {
              const options = Array.from(el.options || []);
              const lowerVal = String(val).toLowerCase().trim();
              const matchedOpt = options.find((opt) =>
                (opt.value && opt.value.toLowerCase().trim() === lowerVal) ||
                (opt.text && opt.text.toLowerCase().trim() === lowerVal) ||
                (opt.text && opt.text.toLowerCase().trim().includes(lowerVal)) ||
                (opt.value && opt.value.toLowerCase().trim().includes(lowerVal))
              );
              if (matchedOpt) {
                el.value = matchedOpt.value;
                currentVal = el.value;
              } else if (val) {
                el.value = val;
                currentVal = el.value;
              }
              el.dispatchEvent(new Event("input", { bubbles: true }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
            } else if (tag === "input" && (el.type === "checkbox" || el.type === "radio")) {
              const lowerVal = String(val).toLowerCase().trim();
              if (lowerVal === "true" || lowerVal === "1" || lowerVal === "check" || lowerVal === "checked") {
                el.checked = true;
              } else if (lowerVal === "false" || lowerVal === "0" || lowerVal === "uncheck" || lowerVal === "unchecked") {
                el.checked = false;
              } else {
                el.checked = !el.checked;
              }
              currentVal = el.checked;
              el.dispatchEvent(new Event("input", { bubbles: true }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
            } else if ("value" in el) {
              el.value = val;
              el.dispatchEvent(new Event("input", { bubbles: true }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
            } else {
              el.textContent = val;
            }

            let submitted = false;
            if (doSubmit) {
              if (el.form && typeof el.form.requestSubmit === "function") {
                try {
                  el.form.requestSubmit();
                  submitted = true;
                } catch (_) {
                  try {
                    el.form.submit();
                    submitted = true;
                  } catch (_) {}
                }
              } else if (el.form && typeof el.form.submit === "function") {
                try {
                  el.form.submit();
                  submitted = true;
                } catch (_) {}
              }
              if (!submitted) {
                const enterEvent = new KeyboardEvent("keydown", {
                  bubbles: true,
                  cancelable: true,
                  key: "Enter",
                  code: "Enter",
                  keyCode: 13,
                  which: 13,
                });
                el.dispatchEvent(enterEvent);
                submitted = true;
              }
            }

            // Await DOM settlement before completing
            await waitForSettled();

            const id = elemId != null ? elemId : (el.getAttribute("data-zelyr-id") ? Number(el.getAttribute("data-zelyr-id")) : null);
            return {
              success: true,
              element_id: id,
              element_type: tag,
              value: currentVal,
              text: val,
              submitted: submitted,
            };
          },
          args: [selector, text, submit, elementId],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Fill script execution produced no result" };
        sendReply({ id, action: "fill_element", ...execResult });
        break;
      }

      case "scroll_page": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "scroll_page", error: "No target tab available" });
          return;
        }

        const direction = (msg.direction || "down").toLowerCase();
        const amount = Number(msg.amount) || 600;
        const reSnapshot = Boolean(msg.re_snapshot || msg.snapshot);

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: (dir, amt, doSnapshot) => {
            const initialY = window.scrollY;
            if (dir === "top") {
              window.scrollTo({ top: 0, behavior: "instant" });
            } else if (dir === "bottom") {
              window.scrollTo({ top: document.body ? document.body.scrollHeight : 10000, behavior: "instant" });
            } else if (dir === "up") {
              window.scrollBy({ top: -amt, behavior: "instant" });
            } else {
              window.scrollBy({ top: amt, behavior: "instant" });
            }

            const docHeight = Math.max(
              document.body ? document.body.scrollHeight : 0,
              document.documentElement ? document.documentElement.scrollHeight : 0
            );

            const scrollData = {
              success: true,
              scrollY: window.scrollY,
              scrollX: window.scrollX,
              innerHeight: window.innerHeight,
              scrollHeight: docHeight,
              deltaY: window.scrollY - initialY,
              direction: dir,
            };

            if (doSnapshot) {
              function isElementVisible(el) {
                if (!el) return false;
                if (typeof el.getBoundingClientRect === "function" && typeof window !== "undefined" && typeof window.getComputedStyle === "function") {
                  try {
                    const style = window.getComputedStyle(el);
                    if (style) {
                      if (
                        style.display === "none" ||
                        style.visibility === "hidden" ||
                        parseFloat(style.opacity || "1") === 0
                      ) {
                        return false;
                      }
                    }
                    const rect = el.getBoundingClientRect();
                    if (rect && rect.width === 0 && rect.height === 0 && el.offsetParent === null) {
                      const root = typeof el.getRootNode === "function" ? el.getRootNode() : null;
                      if (!root || root === document) {
                        return false;
                      }
                    }
                  } catch (_) {}
                }
                return true;
              }

              function clearOldLabels(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set()) {
                if (!root || currentDepth > maxDepth || seenRoots.has(root)) return;
                seenRoots.add(root);
                try {
                  const old = Array.from(root.querySelectorAll("[data-zelyr-id]"));
                  for (const el of old) el.removeAttribute("data-zelyr-id");
                } catch (_) {}
                try {
                  const all = Array.from(root.querySelectorAll("*"));
                  for (const el of all) {
                    if (el.shadowRoot && el.shadowRoot.mode === "open") {
                      clearOldLabels(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots);
                    } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                      try {
                        const doc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                        if (doc) clearOldLabels(doc, maxDepth, currentDepth + 1, seenRoots);
                      } catch (_) {}
                    }
                  }
                } catch (_) {}
              }

              function collectInteractiveElements(root, maxDepth = 10, currentDepth = 0, seenRoots = new Set(), extraSelector = "") {
                const interactive = [];
                if (!root || currentDepth > maxDepth || seenRoots.has(root)) return interactive;
                seenRoots.add(root);

                const baseSelector = [
                  "a[href]",
                  "button",
                  "input",
                  "textarea",
                  "select",
                  '[role="button"]',
                  '[role="link"]',
                  '[role="checkbox"]',
                  '[role="tab"]',
                  '[role="menuitem"]',
                  "[onclick]",
                  '[tabindex]:not([tabindex="-1"])',
                ].join(", ");

                const selector = extraSelector ? `${baseSelector}, ${extraSelector}` : baseSelector;

                try {
                  const nodes = Array.from(root.querySelectorAll(selector));
                  for (const el of nodes) {
                    if (isElementVisible(el)) {
                      interactive.push(el);
                    }
                  }
                } catch (_) {}

                try {
                  const allElements = Array.from(root.querySelectorAll("*"));
                  for (const el of allElements) {
                    if (el.shadowRoot && el.shadowRoot.mode === "open") {
                      const shadowChildren = collectInteractiveElements(el.shadowRoot, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                      for (const child of shadowChildren) {
                        interactive.push(child);
                      }
                    } else if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
                      try {
                        const frameDoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                        if (frameDoc && (frameDoc.body || frameDoc.documentElement)) {
                          const frameChildren = collectInteractiveElements(frameDoc, maxDepth, currentDepth + 1, seenRoots, extraSelector);
                          for (const child of frameChildren) {
                            interactive.push(child);
                          }
                        }
                      } catch (e) {
                        // Cross-origin SecurityError - ignore safely
                      }
                    }
                  }
                } catch (_) {}

                return interactive;
              }

              clearOldLabels(document);
              window.__zelyrFingerprints = {};

              const candidates = collectInteractiveElements(document);
              const items = [];
              let index = 1;

              for (const el of candidates) {
                if (index > 50) break;
                if (!isElementVisible(el)) continue;

                el.setAttribute("data-zelyr-id", String(index));
                const tag = (el.tagName || "").toLowerCase();
                const role = (typeof el.getAttribute === "function" ? el.getAttribute("role") : el.role) || "";
                const type = (typeof el.getAttribute === "function" ? el.getAttribute("type") : null) || el.type || "";
                const name = (typeof el.getAttribute === "function" ? el.getAttribute("name") : null) || el.name || "";
                const ariaLabel = (typeof el.getAttribute === "function" ? (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby")) : null) || "";
                const textSnippet = (el.innerText || el.textContent || "").trim().slice(0, 40);

                window.__zelyrFingerprints[index] = {
                  tag,
                  role,
                  type,
                  name,
                  ariaLabel,
                  textSnippet,
                };

                const item = { id: index, tag };
                if (tag === "input" || tag === "textarea") {
                  if (type) item.type = type;
                  if (el.placeholder) item.placeholder = el.placeholder.slice(0, 100);
                  if (el.value) item.value = el.value.slice(0, 100);
                  if (name) item.name = name;
                }
                if (ariaLabel) item.aria_label = ariaLabel.trim().slice(0, 100);
                const text = (el.innerText || el.textContent || "").trim().replace(/\s+/g, " ");
                if (text && tag !== "input") item.text = text.slice(0, 100);
                if (tag === "a") {
                  const href = (typeof el.getAttribute === "function" ? el.getAttribute("href") : el.href) || "";
                  if (href) item.href = href.slice(0, 120);
                }
                if (role) item.role = role;
                items.push(item);
                index++;
              }

              const summaryLines = items.map((it) => {
                const parts = [`[${it.id}] <${it.tag}`];
                if (it.type) parts.push(`type="${it.type}"`);
                if (it.placeholder) parts.push(`placeholder="${it.placeholder}"`);
                if (it.name) parts.push(`name="${it.name}"`);
                if (it.role) parts.push(`role="${it.role}"`);
                parts.push(">");
                if (it.text) parts.push(`"${it.text}"`);
                if (it.value) parts.push(`value="${it.value}"`);
                if (it.aria_label) parts.push(`(aria: "${it.aria_label}")`);
                if (it.href) parts.push(`(href: ${it.href})`);
                return parts.join(" ");
              });

              try {
                const allFrames = Array.from(document.querySelectorAll("iframe, frame"));
                for (const f of allFrames) {
                  try {
                    const doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
                    if (!doc) summaryLines.push("[iframe: cross-origin restricted]");
                  } catch (_) {
                    summaryLines.push("[iframe: cross-origin restricted]");
                  }
                }
              } catch (_) {}

              scrollData.elements = items;
              scrollData.summary = summaryLines.join("\n");
              scrollData.count = items.length;
            }

            return scrollData;
          },
          args: [direction, amount, reSnapshot],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Scroll execution produced no result" };
        sendReply({ id, action: "scroll_page", ...execResult });
        break;
      }

      case "extract_page_content": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "extract_page_content", error: "No target tab available" });
          return;
        }

        const mode = (msg.mode || "readable").toLowerCase();
        const maxChars = Number(msg.max_chars) || 4000;

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: (extractMode, maxBudget) => {
            const mainRoot = document.querySelector("article, main, [role='main'], #content, .content, .post-content") || document.body;
            if (!mainRoot) {
              return { success: false, error: "No document body available" };
            }

            const clone = mainRoot.cloneNode(true);
            const noiseSelectors = [
              "script",
              "style",
              "noscript",
              "svg",
              "iframe",
              "nav",
              "footer",
              "header",
              "form",
              "[role='navigation']",
              "[role='banner']",
              "[role='complementary']",
              "[aria-hidden='true']",
              ".cookie",
              ".cookie-banner",
              ".cookies",
              "[id*='cookie']",
              "[class*='cookie']",
              "[id*='consent']",
              "[class*='consent']",
              ".popup",
              ".modal",
              ".advert",
              ".ad",
              "[class*='advertisement']",
            ];
            try {
              const toRemove = clone.querySelectorAll(noiseSelectors.join(", "));
              for (const el of toRemove) {
                el.remove();
              }
            } catch (_) {}

            function domToMarkdown(node) {
              if (node.nodeType === Node.TEXT_NODE) {
                return node.nodeValue.replace(/\s+/g, " ");
              }
              if (node.nodeType !== Node.ELEMENT_NODE) {
                return "";
              }

              const tag = node.tagName.toLowerCase();
              let childText = "";
              for (const child of node.childNodes) {
                childText += domToMarkdown(child);
              }
              childText = childText.trim();

              if (!childText && !["hr", "br", "img"].includes(tag)) {
                return "";
              }

              switch (tag) {
                case "h1":
                  return `\n\n# ${childText}\n\n`;
                case "h2":
                  return `\n\n## ${childText}\n\n`;
                case "h3":
                  return `\n\n### ${childText}\n\n`;
                case "h4":
                case "h5":
                case "h6":
                  return `\n\n#### ${childText}\n\n`;
                case "p":
                  return `\n\n${childText}\n\n`;
                case "li":
                  return `\n- ${childText}`;
                case "ul":
                case "ol":
                  return `\n\n${childText}\n\n`;
                case "pre":
                case "code":
                  if (tag === "pre" || (node.parentElement && node.parentElement.tagName.toLowerCase() !== "pre")) {
                    return `\n\`\`\`\n${node.innerText || childText}\n\`\`\`\n`;
                  }
                  return `\`${childText}\``;
                case "blockquote":
                  return `\n> ${childText.replace(/\n/g, "\n> ")}\n\n`;
                case "a": {
                  const href = node.getAttribute("href");
                  if (href && !href.startsWith("javascript:")) {
                    return `[${childText}](${href})`;
                  }
                  return childText;
                }
                case "strong":
                case "b":
                  return `**${childText}**`;
                case "em":
                case "i":
                  return `*${childText}*`;
                case "br":
                  return "\n";
                case "hr":
                  return "\n---\n";
                default:
                  if (["div", "section", "article", "main", "table", "tr"].includes(tag)) {
                    return `\n${childText}\n`;
                  }
                  return ` ${childText} `;
              }
            }

            let markdown = domToMarkdown(clone);
            markdown = markdown
              .replace(/[ \t]+/g, " ")
              .replace(/\n\s*\n\s*\n+/g, "\n\n")
              .trim();

            let truncated = false;
            if (markdown.length > maxBudget) {
              markdown = markdown.slice(0, maxBudget) + "\n\n[... content truncated for context budget]";
              truncated = true;
            }

            return {
              success: true,
              title: document.title,
              url: window.location.href,
              content: markdown,
              length: markdown.length,
              truncated,
            };
          },
          args: [mode, maxChars],
        });

        const execResult = results && results[0] ? results[0].result : { success: false, error: "Content extraction script produced no result" };
        sendReply({ id, action: "extract_page_content", ...execResult });
        break;
      }

      case "evaluate_script": {
        const expression = msg.expression || msg.script || msg.code || "";
        if (!expression) {
          sendReply({ id, success: false, action: "evaluate_script", error: "Missing expression parameter" });
          return;
        }

        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "evaluate_script", error: "No target tab available" });
          return;
        }

        try {
          const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (codeStr) => {
              try {
                const res = window.eval(codeStr);
                return { success: true, result: res };
              } catch (evalErr) {
                return { success: false, error: evalErr.message || String(evalErr) };
              }
            },
            args: [expression],
          });

          const execResult = results && results[0] ? results[0].result : { success: false, error: "Script execution produced no result" };
          sendReply({ id, action: "evaluate_script", ...execResult });
        } catch (err) {
          sendReply({ id, success: false, action: "evaluate_script", error: `Failed to evaluate script: ${err.message || String(err)}` });
        }
        break;
      }

      case "duplicate_tab": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "duplicate_tab", error: "No target tab found to duplicate" });
          return;
        }
        try {
          const duplicated = await chrome.tabs.duplicate(tab.id);
          sendReply({
            id,
            success: true,
            action: "duplicate_tab",
            tab_id: duplicated.id,
            title: duplicated.title || "",
            url: duplicated.url || "",
          });
        } catch (err) {
          sendReply({ id, success: false, action: "duplicate_tab", error: `Failed to duplicate tab: ${err.message || String(err)}` });
        }
        break;
      }

      case "reload_tab": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "reload_tab", error: "No target tab found to reload" });
          return;
        }
        try {
          const bypassCache = Boolean(msg.bypass_cache || msg.bypassCache);
          await chrome.tabs.reload(tab.id, { bypassCache });
          sendReply({
            id,
            success: true,
            action: "reload_tab",
            tab_id: tab.id,
            title: tab.title || "",
            url: tab.url || "",
          });
        } catch (err) {
          sendReply({ id, success: false, action: "reload_tab", error: `Failed to reload tab: ${err.message || String(err)}` });
        }
        break;
      }

      case "pin_tab": {
        const tab = await getTargetTab(msg);
        if (!tab || !tab.id) {
          sendReply({ id, success: false, action: "pin_tab", error: "No target tab found to pin" });
          return;
        }
        try {
          const pinned = msg.pinned !== undefined ? Boolean(msg.pinned) : !tab.pinned;
          const updated = await chrome.tabs.update(tab.id, { pinned });
          sendReply({
            id,
            success: true,
            action: "pin_tab",
            tab_id: updated.id,
            pinned: Boolean(updated.pinned),
            title: updated.title || "",
          });
        } catch (err) {
          sendReply({ id, success: false, action: "pin_tab", error: `Failed to update tab pin state: ${err.message || String(err)}` });
        }
        break;
      }

      case "manage_browser_tabs": {
        const subaction = (msg.operation || msg.tab_action || msg.subaction || "").toLowerCase().trim();
        if (subaction === "duplicate") {
          return handleCommand({ ...msg, action: "duplicate_tab" });
        } else if (subaction === "reload") {
          return handleCommand({ ...msg, action: "reload_tab" });
        } else if (subaction === "pin" || subaction === "unpin") {
          const pinVal = subaction === "pin";
          return handleCommand({ ...msg, action: "pin_tab", pinned: msg.pinned !== undefined ? msg.pinned : pinVal });
        } else if (subaction === "close") {
          return handleCommand({ ...msg, action: "close_tab" });
        } else if (subaction === "open") {
          return handleCommand({ ...msg, action: "open_tab" });
        } else if (subaction === "switch") {
          return handleCommand({ ...msg, action: "switch_tab" });
        } else if (subaction === "list") {
          return handleCommand({ ...msg, action: "list_tabs" });
        } else {
          sendReply({
            id,
            success: false,
            action: "manage_browser_tabs",
            error: `Unsupported tab operation: '${subaction}'. Supported: 'duplicate', 'reload', 'pin', 'close', 'open', 'switch', 'list'.`,
          });
        }
        break;
      }

      default:
        sendReply({
          id,
          success: false,
          error: `Unsupported action: '${action}'`,
          action,
        });
        break;
    }
  } catch (err) {
    sendReply({
      id,
      success: false,
      error: `Internal extension error: ${err.message || String(err)}`,
      action,
    });
  }
}

// Start connection immediately on extension load
if (typeof chrome !== "undefined" && chrome.runtime) {
  connect();
  if (chrome.runtime.onStartup) {
    chrome.runtime.onStartup.addListener(() => connect());
  }
  if (chrome.runtime.onInstalled) {
    chrome.runtime.onInstalled.addListener(() => connect());
  }
}

// Periodic alarm keep-alive to protect against Manifest V3 worker idle termination
if (typeof chrome !== "undefined" && chrome.alarms) {
  try {
    chrome.alarms.create("zelyr_keep_alive", { periodInMinutes: 0.33 }); // Every ~20s
    chrome.alarms.onAlarm.addListener((alarm) => {
      if (alarm.name === "zelyr_keep_alive") {
        log("Keep-alive alarm triggered: ensuring WebSocket link is active");
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          connect();
        }
      }
    });
  } catch (err) {
    log("Failed to register chrome.alarms keep-alive:", err);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    waitForSettled,
    resolveElement,
    collectInteractiveElements,
    findDeepElement,
    clearOldLabels,
    isElementVisible,
    handleCommand,
    connect,
  };
}
