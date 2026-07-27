/** MedSafe RAG — frontend application logic */

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const input = $("#question-input");
  const askBtn = $("#ask-btn");
  const btnText = askBtn.querySelector(".btn-text");
  const btnSpinner = askBtn.querySelector(".btn-spinner");
  const welcomeState = $("#welcome-state");
  const resultsState = $("#results-state");
  const statusBadge = $("#status-badge");
  const criticalRisks = $("#critical-risks");
  const metricsRow = $("#metrics-row");
  const combinedUseBadge = $("#combined-use-badge");
  const answerBlock = $("#answer-block");
  const brandNotice = $("#brand-notice");
  const queryAnalysis = $("#query-analysis");
  const citedLabels = $("#cited-labels");
  const examplesGrid = $("#examples-grid");
  const copyBtn = $("#copy-btn");
  const exportBtn = $("#export-btn");
  const thumbUp = $("#thumb-up");
  const thumbDown = $("#thumb-down");

  let lastResult = null;
  let examples = [];

  function sanitizeDashes(text) {
    if (!text) return text;
    return String(text)
      .replace(/\s*—\s*/g, ", ")
      .replace(/\s*–\s*/g, ", ")
      .replace(/\s*--\s*/g, ", ");
  }

  function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = sanitizeDashes(text);
    return d.innerHTML;
  }

  function showToast(msg) {
    const existing = $(".toast");
    if (existing) existing.remove();
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2800);
  }

  function setLoading(on) {
    askBtn.disabled = on;
    btnText.classList.toggle("hidden", on);
    btnSpinner.classList.toggle("hidden", !on);
  }

  async function loadExamples() {
    try {
      const res = await fetch("/api/examples");
      examples = await res.json();
    } catch {
      examples = [
        "Is Tylenol safe to take every day?",
        "What are the interactions for warfarin?",
        "Can my mom take warfarin and ibuprofen together?",
        "What are the side effects of sertraline?",
        "What is the normal dose of metformin?",
      ];
    }
    examplesGrid.innerHTML = examples
      .map(
        (q) =>
          `<button type="button" class="example-chip" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`
      )
      .join("");
    $$(".example-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        input.value = chip.dataset.q;
        askQuestion();
      });
    });
  }

  function renderMetrics(metrics) {
    if (!metrics) {
      metricsRow.classList.add("hidden");
      return;
    }
    metricsRow.classList.remove("hidden");
    metricsRow.innerHTML = `
      <div class="metric-card">
        <div class="metric-label">Active Ingredient</div>
        <div class="metric-value">${escapeHtml(metrics.active_ingredient)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value risk-${metrics.risk_level}">${escapeHtml(metrics.risk)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Demographic</div>
        <div class="metric-value">${escapeHtml(metrics.demographic)}</div>
      </div>
    `;
  }

  function renderCombinedUseBadge(data) {
    const combo = data.combined_use_risk;
    if (!combo) {
      combinedUseBadge.classList.add("hidden");
      combinedUseBadge.innerHTML = "";
      return;
    }
    combinedUseBadge.classList.remove("hidden");
    combinedUseBadge.className = `combined-use-badge risk-${combo.risk_level}`;
    combinedUseBadge.innerHTML = `
      <span class="combo-icon">⚠️</span>
      <span>${escapeHtml(combo.message)}</span>`;
  }

  function renderBrandNotice(data) {
    const subs = data.brand_substitutions || [];
    if (!subs.length || data.abstained) {
      brandNotice.classList.add("hidden");
      brandNotice.innerHTML = "";
      return;
    }
    brandNotice.classList.remove("hidden");
    const lines = subs
      .map((s) => `<em>${escapeHtml(s.brand)} → ${escapeHtml(s.generic)}</em>`)
      .join(", ");
    brandNotice.innerHTML = `Brand name resolved: ${lines}`;
  }

  function cardRiskClass(data) {
    if (data.abstained) return "risk-abstain";
    return `risk-${data.structured?.card_risk_level || data.metrics?.risk_level || "low"}`;
  }

  function renderCriticalRisks(structured, metrics) {
    const riskLevel =
      structured?.card_risk_level || metrics?.risk_level || "low";
    // Single source of truth: never show Critical Risks when risk is LOW.
    if (riskLevel === "low" || riskLevel === "abstain" || riskLevel === "unknown") {
      criticalRisks.classList.add("hidden");
      criticalRisks.innerHTML = "";
      return;
    }
    const risks = structured?.critical_risks || [];
    if (!risks.length) {
      criticalRisks.classList.add("hidden");
      criticalRisks.innerHTML = "";
      return;
    }
    criticalRisks.classList.remove("hidden");
    criticalRisks.className = `critical-risks risk-${riskLevel}`;
    const chips = risks
      .map(
        (r) =>
          `<span class="risk-chip severity-${escapeHtml(r.severity || riskLevel)}">${escapeHtml(r.label)}</span>`
      )
      .join("");
    const title = riskLevel === "moderate" ? "Use Caution" : "Critical Risks";
    criticalRisks.innerHTML = `
      <div class="critical-risks-inner">
        <div class="critical-risks-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          ${title}
        </div>
        <div class="risk-chips">${chips}</div>
      </div>`;
  }

  function renderCitationInline(text) {
    const parts = [];
    let lastIndex = 0;
    const re = /\(Source:\s*([^)]+)\)/gi;
    let match;
    while ((match = re.exec(text)) !== null) {
      parts.push(escapeHtml(text.slice(lastIndex, match.index)));
      parts.push(`<span class="inline-citation">(Source: ${escapeHtml(match[1])})</span>`);
      lastIndex = match.index + match[0].length;
    }
    parts.push(escapeHtml(text.slice(lastIndex)));
    return parts.join("");
  }

  function bindTabs(container) {
    const tabs = container.querySelectorAll(".answer-tab");
    const panels = container.querySelectorAll(".tab-panel");
    const activate = (target) => {
      tabs.forEach((t) => {
        const on = t.dataset.tab === target;
        t.classList.toggle("active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach((p) => p.classList.toggle("active", p.dataset.panel === target));
    };
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => activate(tab.dataset.tab));
    });
    container.querySelectorAll(".read-more-clinical").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        activate("clinical");
      });
    });
  }

  function bindTechnicalToggle(container) {
    const btn = container.querySelector(".tech-notes-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const panel = container.querySelector(".tech-notes-panel");
      const open = panel.classList.toggle("open");
      btn.setAttribute("aria-expanded", open);
    });
  }

  function renderAnswer(data) {
    const abstained = data.abstained;
    const s = data.structured || {};
    const riskClass = cardRiskClass(data);
    answerBlock.className = abstained ? "answer-block abstained" : "answer-block";

    if (abstained) {
      let html = `
        <div class="tldr-box risk-abstain">
          <div class="tldr-header">
            <span class="tldr-icon">⚠️</span>
            <div>
              <div class="tldr-label">Quick Summary for Caregivers</div>
              <div class="tldr-verdict">Unable to answer reliably</div>
            </div>
          </div>
        </div>
        <h2>⚠️ Abstention Notice</h2>`;
      if (data.abstention_reason) {
        html += `<p class="answer-paragraph"><strong>Reason:</strong> ${escapeHtml(data.abstention_reason)}</p>`;
      }
      (s.simple_guide || data.paragraphs || []).forEach((p) => {
        html += `<p class="answer-paragraph">${escapeHtml(p)}</p>`;
      });
      if (s.disclaimer || data.disclaimer) {
        html += `<p class="answer-disclaimer">${escapeHtml(s.disclaimer || data.disclaimer)}</p>`;
      }
      answerBlock.innerHTML = html;
      return;
    }

    const verdict = s.verdict || { emoji: "ℹ️", text: "Safety summary", tone: "caution" };
    const bullets = s.tldr_bullets || [];
    const simpleGuide = s.simple_guide || [];
    const clinicalDetails = s.clinical_details || [];
    const techNotes = s.technical_notes || [];

    let html = `
      <div class="tldr-box ${riskClass}">
        <div class="tldr-header">
          <span class="tldr-icon">${verdict.emoji}</span>
          <div>
            <div class="tldr-label">Quick Summary for Caregivers</div>
            <div class="tldr-verdict">${escapeHtml(verdict.text)}</div>
          </div>
        </div>
        <ul class="tldr-bullets">
          ${bullets
            .map(
              (b) =>
                `<li><span class="bullet-icon">${b.icon}</span><span><strong>${escapeHtml(b.label)}:</strong> ${escapeHtml(b.text)}</span></li>`
            )
            .join("")}
        </ul>
      </div>

      <div class="answer-tabs" role="tablist">
        <button type="button" class="answer-tab active" data-tab="simple" role="tab" aria-selected="true">Simple Guide</button>
        <button type="button" class="answer-tab" data-tab="clinical" role="tab" aria-selected="false">Clinical Details</button>
      </div>

      <div class="tab-panels">
        <div class="tab-panel active" data-panel="simple" role="tabpanel">
          <p class="tab-intro">Plain English, action-oriented, no jargon.</p>
          ${simpleGuide
            .map((p) => {
              if (p === "Read more in Clinical Details") {
                return `<p class="answer-paragraph"><a href="#" class="read-more-clinical">${escapeHtml(p)}</a></p>`;
              }
              return `<p class="answer-paragraph">${escapeHtml(p)}</p>`;
            })
            .join("")}
        </div>
        <div class="tab-panel" data-panel="clinical" role="tabpanel">
          <p class="tab-intro">Full FDA-sourced text with exact product references.</p>
          ${clinicalDetails.map((p) => `<p class="answer-paragraph clinical">${renderCitationInline(p)}</p>`).join("")}
        </div>
      </div>`;

    if (techNotes.length) {
      html += `
        <div class="tech-notes">
          <button type="button" class="tech-notes-toggle" aria-expanded="false">
            <span>Technical &amp; Sourcing Notes (${techNotes.length})</span>
            <svg class="tech-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="tech-notes-panel">
            ${techNotes.map((n) => `<p class="tech-note">${escapeHtml(n)}</p>`).join("")}
          </div>
        </div>`;
    }

    if (s.disclaimer || data.disclaimer) {
      html += `<p class="answer-disclaimer">${escapeHtml(s.disclaimer || data.disclaimer)}</p>`;
    }

    answerBlock.innerHTML = html;
    bindTabs(answerBlock);
    bindTechnicalToggle(answerBlock);
  }

  function renderSidebar(data) {
    let qaHtml = `<h3>Query Analysis</h3>`;
    qaHtml += `<div class="query-item"><div class="query-item-label">Original</div><div class="query-item-value">${escapeHtml(data.original_query)}</div></div>`;
    if (data.rewritten_query !== data.original_query) {
      qaHtml += `<div class="query-item"><div class="query-item-label">Rewritten</div><div class="query-item-value">${escapeHtml(data.rewritten_query)}</div></div>`;
    }
    if (data.substitutions && data.substitutions.length) {
      qaHtml += `<div class="query-item"><div class="query-item-label">Brand Resolution</div><div>`;
      (data.brand_substitutions || []).forEach((s) => {
        qaHtml += `<span class="substitution-tag">${escapeHtml(s.display)}</span>`;
      });
      qaHtml += `</div></div>`;
    }
    qaHtml += `<div class="query-item"><div class="query-item-label">Response Time</div><div class="query-item-value">${data.elapsed_seconds}s</div></div>`;
    queryAnalysis.innerHTML = qaHtml;

    if (data.abstained || !data.citations || !data.citations.length) {
      citedLabels.innerHTML = "";
      citedLabels.classList.add("hidden");
      return;
    }
    citedLabels.classList.remove("hidden");

    let citHtml = `<h3>FDA Sources (${data.citations.length})</h3>`;
    data.citations.forEach((cit, i) => {
      const drug = cit.drug.replace(/_/g, " ");
      citHtml += `
        <div class="citation-item" data-idx="${i}">
          <button type="button" class="citation-header" aria-expanded="false">
            <div>
              <div class="citation-title">${escapeHtml(drug)}</div>
              <div class="citation-section">${escapeHtml(cit.section_display)} · ${escapeHtml(cit.full_product_name || "")}</div>
            </div>
            <svg class="citation-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="citation-body">${escapeHtml(cit.snippet || "Excerpt not available.")}</div>
        </div>`;
    });
    citedLabels.innerHTML = citHtml;

    citedLabels.querySelectorAll(".citation-header").forEach((btn) => {
      btn.addEventListener("click", () => {
        const item = btn.closest(".citation-item");
        const open = item.classList.toggle("open");
        btn.setAttribute("aria-expanded", open);
      });
    });
  }

  function renderStatus(data) {
    if (data.abstained) {
      statusBadge.className = "status-badge warning";
      statusBadge.innerHTML = `<span class="dot"></span> Abstained: insufficient label data`;
    } else {
      statusBadge.className = "status-badge success";
      statusBadge.innerHTML = `<span class="dot"></span> Answer generated from ${data.citations?.length || 0} FDA label source(s)`;
    }
  }

  function showResults(data) {
    lastResult = data;
    welcomeState.classList.add("hidden");
    resultsState.classList.remove("hidden");
    renderStatus(data);
    renderCriticalRisks(data.structured, data.metrics);
    renderMetrics(data.metrics);
    renderCombinedUseBadge(data);
    renderAnswer(data);
    renderBrandNotice(data);
    renderSidebar(data);
    thumbUp.classList.remove("active-up");
    thumbDown.classList.remove("active-down");
    $("#results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function askQuestion() {
    const question = input.value.trim();
    if (!question) {
      showToast("Please enter a question.");
      input.focus();
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (data.error) {
        showToast(data.error);
        return;
      }
      showResults(data);
    } catch {
      showToast("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  copyBtn.addEventListener("click", () => {
    if (!lastResult) return;
    navigator.clipboard.writeText(lastResult.answer).then(() => showToast("Copied to clipboard"));
  });

  exportBtn.addEventListener("click", () => {
    if (!lastResult) return;
    const w = window.open("", "_blank");
    const title = lastResult.abstained ? "MedSafe: Abstention Notice" : "MedSafe: Clinical Summary";
    const s = lastResult.structured || {};
    const exportBody = (s.simple_guide || lastResult.paragraphs || [])
      .map((p) => `<p>${escapeHtml(p)}</p>`)
      .join("");
    const exportTech = (s.technical_notes || [])
      .map((n) => `<p style="font-size:0.85rem;color:#64748b">${escapeHtml(n)}</p>`)
      .join("");
    w.document.write(`<!DOCTYPE html><html><head><title>${title}</title>
      <style>body{font-family:Georgia,serif;max-width:720px;margin:40px auto;padding:0 24px;line-height:1.7;color:#1e293b}
      h1{font-size:1.5rem;border-bottom:2px solid #2563eb;padding-bottom:12px}
      .meta{color:#64748b;font-size:0.875rem;margin-bottom:24px}
      .verdict{background:#fef3c7;border:1px solid #fde68a;padding:16px;border-radius:8px;margin-bottom:24px}
      .disclaimer{margin-top:32px;padding-top:16px;border-top:1px solid #e2e8f0;font-size:0.8125rem;color:#64748b;font-style:italic}
      </style></head><body>
      <h1>${title}</h1>
      <p class="meta">Question: ${escapeHtml(lastResult.original_query)}</p>
      ${s.verdict ? `<div class="verdict"><strong>${s.verdict.emoji} ${escapeHtml(s.verdict.text)}</strong></div>` : ""}
      ${exportBody}
      ${exportTech}
      ${lastResult.disclaimer || s.disclaimer ? `<p class="disclaimer">${escapeHtml(lastResult.disclaimer || s.disclaimer)}</p>` : ""}
      </body></html>`);
    w.document.close();
    w.print();
  });

  thumbUp.addEventListener("click", () => {
    thumbUp.classList.toggle("active-up");
    thumbDown.classList.remove("active-down");
    showToast("Thanks for your feedback!");
  });

  thumbDown.addEventListener("click", () => {
    thumbDown.classList.toggle("active-down");
    thumbUp.classList.remove("active-up");
    showToast("Thanks, we'll use this to improve.");
  });

  askBtn.addEventListener("click", askQuestion);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") askQuestion();
  });

  loadExamples();

  /* ── Mode tabs ── */
  const modeAsk = $("#mode-ask");
  const modeChecker = $("#mode-checker");
  $$(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      $$(".mode-tab").forEach((t) => {
        const active = t === tab;
        t.classList.toggle("active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
      });
      modeAsk.classList.toggle("hidden", mode !== "ask");
      modeAsk.classList.toggle("active", mode === "ask");
      modeChecker.classList.toggle("hidden", mode !== "checker");
      modeChecker.classList.toggle("active", mode === "checker");
    });
  });

  /* ── Medication Checker ── */
  const medsInput = $("#meds-input");
  const medChipInput = $("#med-chip-input");
  const addMedBtn = $("#add-med-btn");
  const medChipsEl = $("#med-chips");
  const medInlineMsg = $("#med-inline-msg");
  const suggestionChipsEl = $("#suggestion-chips");
  const pairPreview = $("#pair-preview");
  const checkMedsBtn = $("#check-meds-btn");
  const checkBtnText = checkMedsBtn.querySelector(".btn-text");
  const checkBtnSpinner = checkMedsBtn.querySelector(".btn-spinner");
  const clearMedsBtn = $("#clear-meds-btn");
  const loadDemoBtn = $("#load-demo-btn");
  const checkerEmpty = $("#checker-empty");
  const checkerProgress = $("#checker-progress");
  const progressLabel = $("#progress-label");
  const progressCount = $("#progress-count");
  const progressBar = $("#progress-bar");
  const checkerResolutions = $("#checker-resolutions");
  const checkerSummary = $("#checker-summary");
  const checkerMatrixWrap = $("#checker-matrix-wrap");
  const checkerMatrixTable = $("#checker-matrix-table");
  const checkerDetailsWrap = $("#checker-details-wrap");
  const checkerDetails = $("#checker-details");
  const checkerExportWrap = $("#checker-export-wrap");
  const checkerExportBtn = $("#checker-export-btn");

  const SUGGESTIONS = [
    "warfarin",
    "ibuprofen",
    "metformin",
    "lisinopril",
    "sertraline",
    "Tylenol",
    "Advil",
    "Coumadin",
    "amlodipine",
    "atorvastatin",
    "amoxicillin",
    "naproxen",
  ];
  const DEMO_LIST = ["warfarin", "ibuprofen", "metformin", "lisinopril", "sertraline"];

  /** @type {{ input: string, resolved: string, notice?: string|null }[]} */
  let medList = [];
  let lastCheckerReport = null;
  let addInFlight = false;

  function uniqueResolvedNames() {
    const seen = [];
    const keys = new Set();
    medList.forEach((m) => {
      const key = m.resolved.toLowerCase();
      if (!keys.has(key)) {
        keys.add(key);
        seen.push(m.resolved);
      }
    });
    return seen;
  }

  function showMedInlineMsg(text, kind) {
    if (!medInlineMsg) return;
    if (!text) {
      medInlineMsg.className = "med-inline-msg hidden";
      medInlineMsg.textContent = "";
      return;
    }
    medInlineMsg.className = `med-inline-msg msg-${kind || "error"}`;
    medInlineMsg.textContent = text;
  }

  function riskClass(risk) {
    return String(risk || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  }

  function syncMedsTextarea() {
    const unique = uniqueResolvedNames();
    const n = unique.length;
    const pairs = n < 2 ? 0 : (n * (n - 1)) / 2;
    medsInput.value = unique.join("\n");
    pairPreview.textContent = `${n} med${n === 1 ? "" : "s"} · ${pairs} pair${pairs === 1 ? "" : "s"}`;
    checkMedsBtn.disabled = addInFlight || n < 2;
    if (medList.length >= 2 && n < 2) {
      showMedInlineMsg("Please enter at least 2 different medications", "error");
    }
    renderMedChips();
    renderSuggestions();
  }

  function renderMedChips() {
    medChipsEl.innerHTML = medList
      .map(
        (med, i) => `
      <span class="med-chip">
        ${escapeHtml(med.resolved)}
        <button type="button" data-remove="${i}" aria-label="Remove ${escapeHtml(med.resolved)}">×</button>
      </span>`
      )
      .join("");
    medChipsEl.querySelectorAll("[data-remove]").forEach((btn) => {
      btn.addEventListener("click", () => {
        medList.splice(Number(btn.dataset.remove), 1);
        showMedInlineMsg("", "");
        syncMedsTextarea();
      });
    });
  }

  function renderSuggestions() {
    const lowerResolved = new Set(medList.map((m) => m.resolved.toLowerCase()));
    const lowerInput = new Set(medList.map((m) => m.input.toLowerCase()));
    suggestionChipsEl.innerHTML = SUGGESTIONS.map((name) => {
      const used = lowerResolved.has(name.toLowerCase()) || lowerInput.has(name.toLowerCase());
      return `<button type="button" class="suggestion-chip${used ? " in-list" : ""}" data-add="${escapeHtml(name)}" ${used ? "disabled" : ""}>${escapeHtml(name)}</button>`;
    }).join("");
    suggestionChipsEl.querySelectorAll("[data-add]").forEach((btn) => {
      btn.addEventListener("click", () => addMedication(btn.dataset.add));
    });
  }

  async function addMedication(raw) {
    const name = (raw || "").trim();
    if (!name || addInFlight) return;

    addInFlight = true;
    addMedBtn.disabled = true;
    try {
      const res = await fetch("/api/medications/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ medication: name }),
      });
      const data = await res.json();
      if (data.error) {
        showMedInlineMsg(data.error, "error");
        return;
      }

      const resolved = (data.resolved || name).trim();
      const existing = medList.find((m) => m.resolved.toLowerCase() === resolved.toLowerCase());
      if (existing) {
        showMedInlineMsg(
          `${data.input} is the same as ${existing.resolved}, which is already on your list.`,
          "error"
        );
        medChipInput.value = "";
        syncMedsTextarea();
        return;
      }

      medList.push({
        input: data.input || name,
        resolved,
        notice: data.notice || null,
      });
      medChipInput.value = "";
      if (data.notice) {
        showMedInlineMsg(data.notice, "info");
      } else {
        showMedInlineMsg("", "");
      }
      syncMedsTextarea();
    } catch {
      showMedInlineMsg("Could not resolve that medication. Please try again.", "error");
    } finally {
      addInFlight = false;
      addMedBtn.disabled = false;
      syncMedsTextarea();
    }
  }

  function setCheckerLoading(on) {
    checkMedsBtn.disabled = on || uniqueResolvedNames().length < 2;
    checkBtnText.classList.toggle("hidden", on);
    checkBtnSpinner.classList.toggle("hidden", !on);
    addMedBtn.disabled = on || addInFlight;
    medChipInput.disabled = on;
  }

  function setProgress(current, total, label) {
    checkerProgress.classList.remove("hidden");
    progressLabel.textContent = label;
    progressCount.textContent = total ? `${current} / ${total}` : "";
    const pct = total ? Math.round((current / total) * 100) : 8;
    progressBar.style.width = `${Math.max(4, pct)}%`;
  }

  function buildAnswerHtml(data) {
    const s = data.structured || {};
    if (data.abstained) {
      let html = `<p class="answer-paragraph"><strong>Unable to answer reliably.</strong></p>`;
      if (data.abstention_reason) {
        html += `<p class="answer-paragraph">${escapeHtml(data.abstention_reason)}</p>`;
      }
      (s.simple_guide || data.paragraphs || []).forEach((p) => {
        html += `<p class="answer-paragraph">${escapeHtml(p)}</p>`;
      });
      return html;
    }
    // Medication Checker: strip inline (Source: ...) — table below handles attribution.
    // Ask tab is unchanged (does not use buildAnswerHtml).
    const stripInlineSources = (text) =>
      String(text || "")
        .replace(/\s*\(Source:\s*[^)]+\)/gi, "")
        .replace(/\s{2,}/g, " ")
        .trim();
    const bodyParas = (s.simple_guide || data.paragraphs || s.clinical_details || [])
      .map(stripInlineSources)
      .filter(Boolean);
    let html = bodyParas.map((p) => `<p class="answer-paragraph">${escapeHtml(p)}</p>`).join("");
    if (data.citations && data.citations.length) {
      const rows = data.citations
        .map((cit) => {
          const drug = (cit.drug || "").replace(/_/g, " ");
          const section = (cit.section_display || cit.section || "").replace(/_/g, " ");
          const product = cit.full_product_name || "—";
          return `<tr style="border-top:1px solid #EEEEEE;">
            <td style="padding:6px 12px; width:20%;">${escapeHtml(drug)}</td>
            <td style="padding:6px 12px; width:30%;">${escapeHtml(section)}</td>
            <td style="padding:6px 12px; width:50%;">${escapeHtml(product)}</td>
          </tr>`;
        })
        .join("");
      html += `<div class="pair-citations">
        <h4>FDA Sources referenced</h4>
        <div class="source-table-wrap">
          <table class="source-meta-table" style="width:100%; border-collapse:collapse; font-size:13px;">
            <thead>
              <tr style="background:#F5F5F5;">
                <th style="padding:6px 12px; text-align:left; width:20%; color:#666;">Source</th>
                <th style="padding:6px 12px; text-align:left; width:30%; color:#666;">Section</th>
                <th style="padding:6px 12px; text-align:left; width:50%; color:#666;">Product</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        ${data.limited_sources_note ? `<p class="answer-paragraph"><em>${escapeHtml(data.limited_sources_note)}</em></p>` : ""}
      </div>`;
    } else if (data.limited_sources_note) {
      html += `<div class="pair-citations"><p class="answer-paragraph"><em>${escapeHtml(data.limited_sources_note)}</em></p></div>`;
    } else if (!data.abstained) {
      html += `<div class="pair-citations"><p class="answer-paragraph"><em>Limited FDA label data available for this specific combination.</em></p></div>`;
    }
    if (s.disclaimer || data.disclaimer) {
      html += `<p class="answer-disclaimer">${escapeHtml(s.disclaimer || data.disclaimer)}</p>`;
    }
    return html;
  }

  function applyDetailFilter(filter) {
    $$("#detail-filters .filter-chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.filter === filter);
    });
    checkerDetails.querySelectorAll(".pair-detail-card").forEach((card) => {
      const risk = card.dataset.risk;
      const show = filter === "all" || risk === filter;
      card.classList.toggle("filtered-out", !show);
    });
  }

  function renderCheckerReport(report) {
    lastCheckerReport = report;
    checkerEmpty.classList.add("hidden");

    if (report.resolutions && report.resolutions.length) {
      checkerResolutions.classList.remove("hidden");
      checkerResolutions.innerHTML =
        "<strong>Brand names resolved</strong><br />" +
        report.resolutions.map((r) => escapeHtml(r.display)).join("<br />");
    } else {
      checkerResolutions.classList.add("hidden");
      checkerResolutions.innerHTML = "";
    }

    const summaryClass =
      report.high_count > 0
        ? "risk-high"
        : report.moderate_count > 0
          ? "risk-moderate"
          : report.unknown_count > 0
            ? "risk-unknown"
            : "risk-low";
    const bannerTitle =
      report.high_count > 0
        ? "⚠️ Interactions need attention"
        : report.moderate_count > 0
          ? "⚠️ Some combinations need caution"
          : report.unknown_count > 0
            ? "⚠️ Some pairs could not be fully classified — review below"
            : "✅ No significant interactions found";

    checkerSummary.className = `checker-summary ${summaryClass}`;
    checkerSummary.classList.remove("hidden");
    checkerSummary.innerHTML = `
      <div class="summary-banner-title">${bannerTitle}</div>
      <div class="summary-stats">
        <div class="summary-stat"><div class="summary-stat-label">Medications</div><div class="summary-stat-value">${report.medication_count}</div></div>
        <div class="summary-stat"><div class="summary-stat-label">Pairs checked</div><div class="summary-stat-value">${report.pair_count}</div></div>
        <div class="summary-stat"><div class="summary-stat-label">High risk</div><div class="summary-stat-value risk-stat-high">${report.high_count}</div></div>
        <div class="summary-stat"><div class="summary-stat-label">Moderate</div><div class="summary-stat-value risk-stat-moderate">${report.moderate_count}</div></div>
        <div class="summary-stat"><div class="summary-stat-label">Needs review</div><div class="summary-stat-value risk-stat-unknown">${report.unknown_count || 0}</div></div>
      </div>`;

    const rows = report.pairs
      .map(
        (p, i) => `
      <tr class="pair-row risk-${riskClass(p.pair_risk)}" data-pair-idx="${i}">
        <td><strong>${escapeHtml(p.drug_a)}</strong></td>
        <td><strong>${escapeHtml(p.drug_b)}</strong></td>
        <td><span class="risk-pill ${riskClass(p.pair_risk)}">${escapeHtml(p.pair_risk)}</span></td>
        <td>${escapeHtml(p.key_warning)}</td>
      </tr>`
      )
      .join("");

    checkerMatrixWrap.classList.remove("hidden");
    checkerMatrixTable.innerHTML = `
      <div class="matrix-scroll">
        <table class="matrix-table">
          <thead>
            <tr>
              <th>Drug A</th>
              <th>Drug B</th>
              <th>Risk</th>
              <th>Key Warning</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    checkerDetailsWrap.classList.remove("hidden");
    checkerDetails.innerHTML = report.pairs
      .map((p, i) => {
        const openByDefault = p.pair_risk === "HIGH";
        return `
        <div class="pair-detail-card risk-${riskClass(p.pair_risk)} ${openByDefault ? "open" : ""}" data-detail-idx="${i}" data-risk="${p.pair_risk}">
          <button type="button" class="pair-detail-header" aria-expanded="${openByDefault}">
            <div>
              <div class="pair-detail-title">${escapeHtml(p.drug_a)} + ${escapeHtml(p.drug_b)}</div>
              <div class="pair-detail-sub"><span class="risk-pill ${riskClass(p.pair_risk)}">${escapeHtml(p.pair_risk)}</span> ${escapeHtml(p.key_warning)}</div>
            </div>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="pair-detail-body">${buildAnswerHtml(p)}</div>
        </div>`;
      })
      .join("");

    const order = { HIGH: 0, MODERATE: 1, UNKNOWN: 2, LOW: 3 };
    Array.from(checkerDetails.children)
      .sort((a, b) => {
        const ia = Number(a.dataset.detailIdx);
        const ib = Number(b.dataset.detailIdx);
        return (order[report.pairs[ia].pair_risk] ?? 9) - (order[report.pairs[ib].pair_risk] ?? 9);
      })
      .forEach((card) => checkerDetails.appendChild(card));

    checkerDetails.querySelectorAll(".pair-detail-header").forEach((btn) => {
      btn.addEventListener("click", () => {
        const card = btn.closest(".pair-detail-card");
        const open = card.classList.toggle("open");
        btn.setAttribute("aria-expanded", open);
      });
    });

    checkerMatrixTable.querySelectorAll(".pair-row").forEach((row) => {
      row.addEventListener("click", () => {
        const idx = Number(row.dataset.pairIdx);
        const card = checkerDetails.querySelector(`[data-detail-idx="${idx}"]`);
        if (!card) return;
        applyDetailFilter("all");
        card.classList.add("open");
        card.querySelector(".pair-detail-header")?.setAttribute("aria-expanded", "true");
        card.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    $$("#detail-filters .filter-chip").forEach((chip) => {
      chip.onclick = () => applyDetailFilter(chip.dataset.filter);
    });
    applyDetailFilter("all");

    checkerExportWrap.classList.remove("hidden");
  }

  async function runMedicationCheck() {
    const unique = uniqueResolvedNames();
    if (unique.length < 2) {
      showMedInlineMsg("Please enter at least 2 different medications", "error");
      showToast("Please enter at least 2 different medications");
      return;
    }

    setCheckerLoading(true);
    checkerEmpty.classList.add("hidden");
    setProgress(0, 0, "Preparing medication list…");
    checkerSummary.classList.add("hidden");
    checkerMatrixWrap.classList.add("hidden");
    checkerDetailsWrap.classList.add("hidden");
    checkerExportWrap.classList.add("hidden");
    checkerResolutions.classList.add("hidden");

    try {
      const prepRes = await fetch("/api/medications/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ medications: unique.join("\n") }),
      });
      const prep = await prepRes.json();
      if (prep.error) {
        showMedInlineMsg(prep.error, "error");
        showToast(prep.error);
        checkerProgress.classList.add("hidden");
        checkerEmpty.classList.remove("hidden");
        return;
      }

      if (prep.warning) showToast(prep.warning);

      if (prep.resolutions && prep.resolutions.length) {
        checkerResolutions.classList.remove("hidden");
        checkerResolutions.innerHTML =
          "<strong>Brand names resolved</strong><br />" +
          prep.resolutions.map((r) => escapeHtml(r.display)).join("<br />");
      }

      const pairs = prep.pairs || [];
      const results = [];
      for (let i = 0; i < pairs.length; i++) {
        const pair = pairs[i];
        setProgress(i + 1, pairs.length, `Checking ${pair.drug_a} + ${pair.drug_b}`);

        // Safety net: skip same-drug pairs without calling the API.
        if (String(pair.drug_a).toLowerCase() === String(pair.drug_b).toLowerCase()) {
          results.push({
            drug_a: pair.drug_a,
            drug_b: pair.drug_b,
            pair_risk: "N/A",
            key_warning: "Same medication — no interaction check needed",
            abstained: false,
            answer: "Same medication — no interaction check needed",
            paragraphs: ["Same medication — no interaction check needed"],
            structured: {
              simple_guide: ["Same medication — no interaction check needed"],
              clinical_details: ["Same medication — no interaction check needed"],
              disclaimer: "",
            },
            citations: [],
          });
          continue;
        }

        const res = await fetch("/api/medications/check-pair", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ drug_a: pair.drug_a, drug_b: pair.drug_b }),
        });
        const data = await res.json();
        if (data.error) {
          results.push({
            drug_a: pair.drug_a,
            drug_b: pair.drug_b,
            pair_risk: "UNKNOWN",
            key_warning: data.error,
            abstained: true,
            answer: data.error,
            paragraphs: [data.error],
            structured: {},
            citations: [],
          });
        } else {
          results.push(data);
        }
      }

      const highCount = results.filter((r) => r.pair_risk === "HIGH").length;
      const moderateCount = results.filter((r) => r.pair_risk === "MODERATE").length;
      const unknownCount = results.filter((r) => r.pair_risk === "UNKNOWN").length;
      const lowCount = results.filter((r) => r.pair_risk === "LOW").length;

      checkerProgress.classList.add("hidden");
      renderCheckerReport({
        medications: prep.medications,
        resolutions: prep.resolutions || [],
        medication_count: prep.medication_count,
        pair_count: prep.pair_count,
        high_count: highCount,
        moderate_count: moderateCount,
        unknown_count: unknownCount,
        low_count: lowCount,
        interaction_count: highCount + moderateCount,
        pairs: results,
      });
      checkerSummary.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch {
      showToast("Network error. Please try again.");
      checkerProgress.classList.add("hidden");
      checkerEmpty.classList.remove("hidden");
    } finally {
      setCheckerLoading(false);
    }
  }

  checkerExportBtn.addEventListener("click", () => {
    if (!lastCheckerReport) return;
    const r = lastCheckerReport;
    const matrixRows = r.pairs
      .map(
        (p) =>
          `<tr><td>${escapeHtml(p.drug_a)}</td><td>${escapeHtml(p.drug_b)}</td><td>${escapeHtml(p.pair_risk)}</td><td>${escapeHtml(p.key_warning)}</td></tr>`
      )
      .join("");
    const details = r.pairs
      .map((p) => {
        const body = (p.structured?.simple_guide || p.paragraphs || [])
          .map((x) => `<p>${escapeHtml(x)}</p>`)
          .join("");
        return `<h2>${escapeHtml(p.drug_a)} + ${escapeHtml(p.drug_b)} (${escapeHtml(p.pair_risk)})</h2>${body}`;
      })
      .join("");
    const resolutions = (r.resolutions || [])
      .map((x) => `<p>${escapeHtml(x.display)}</p>`)
      .join("");
    const w = window.open("", "_blank");
    w.document.write(`<!DOCTYPE html><html><head><title>MedSafe Medication Report</title>
      <style>
        body{font-family:Georgia,serif;max-width:860px;margin:40px auto;padding:0 24px;line-height:1.6;color:#1e293b}
        h1{font-size:1.6rem;border-bottom:2px solid #2563eb;padding-bottom:12px}
        table{width:100%;border-collapse:collapse;margin:20px 0}
        th,td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;font-size:0.9rem}
        th{background:#f1f5f9}
        .summary{padding:14px;border-radius:8px;background:#f8fafc;margin:16px 0}
      </style></head><body>
      <h1>MedSafe Medication Schedule Report</h1>
      <div class="summary">${r.medication_count} medications · ${r.pair_count} pairs · ${r.interaction_count} interactions (${r.high_count} high, ${r.moderate_count} moderate)</div>
      ${resolutions}
      <h2>Interaction Matrix</h2>
      <table><thead><tr><th>Drug A</th><th>Drug B</th><th>Risk</th><th>Key Warning</th></tr></thead><tbody>${matrixRows}</tbody></table>
      <h2>Detailed Results</h2>
      ${details}
      <p style="margin-top:32px;font-size:0.85rem;color:#64748b;font-style:italic">This is not medical advice. Please consult a doctor or pharmacist before making medication decisions.</p>
      </body></html>`);
    w.document.close();
    w.print();
  });

  addMedBtn.addEventListener("click", () => addMedication(medChipInput.value));
  medChipInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addMedication(medChipInput.value);
    }
  });
  clearMedsBtn.addEventListener("click", () => {
    medList = [];
    showMedInlineMsg("", "");
    syncMedsTextarea();
    showToast("Medication list cleared.");
  });
  loadDemoBtn.addEventListener("click", () => {
    medList = DEMO_LIST.map((name) => ({ input: name, resolved: name, notice: null }));
    showMedInlineMsg("", "");
    syncMedsTextarea();
    showToast("Demo list loaded. Click Check My Medications.");
  });
  checkMedsBtn.addEventListener("click", runMedicationCheck);

  syncMedsTextarea();
})();
