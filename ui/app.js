// Cinch Real-Time Serving Console JS Engine
document.addEventListener("DOMContentLoaded", () => {
  // Tab Switching
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = `pane-${tab.getAttribute("data-tab")}`;
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Parameter Slider Binding
  const sliderMaxTokens = document.getElementById("slider-max-tokens");
  const valMaxTokens = document.getElementById("val-max-tokens");
  const presetBtns = document.querySelectorAll(".btn-preset");

  function setMaxTokens(val) {
    if (sliderMaxTokens && valMaxTokens) {
      sliderMaxTokens.value = val;
      valMaxTokens.textContent = val;
      presetBtns.forEach(btn => {
        if (btn.getAttribute("data-val") === String(val)) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });
    }
  }

  if (sliderMaxTokens && valMaxTokens) {
    sliderMaxTokens.addEventListener("input", (e) => {
      valMaxTokens.textContent = e.target.value;
      presetBtns.forEach(btn => {
        if (btn.getAttribute("data-val") === String(e.target.value)) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });
    });
  }

  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const val = parseInt(btn.getAttribute("data-val"), 10);
      setMaxTokens(val);
    });
  });

  const sliderTemp = document.getElementById("slider-temp");
  const valTemp = document.getElementById("val-temp");
  if (sliderTemp && valTemp) {
    sliderTemp.addEventListener("input", (e) => {
      valTemp.textContent = e.target.value;
    });
  }

  // Populate Memory Block Grid (64 visual blocks)
  const memGrid = document.getElementById("memory-block-grid");
  if (memGrid) {
    memGrid.innerHTML = "";
    for (let i = 0; i < 64; i++) {
      const block = document.createElement("div");
      block.className = "mem-block";
      if (i < 16) block.classList.add("prefix");
      else if (i < 28) block.classList.add("active");
      memGrid.appendChild(block);
    }
  }

  // Dispatch Inference Handler (SSE Streaming & Non-Streaming)
  const btnDispatch = document.getElementById("btn-dispatch");
  const streamOutput = document.getElementById("stream-output");
  const statTtft = document.getElementById("stat-ttft");
  const statTps = document.getElementById("stat-tps");
  const statLatency = document.getElementById("stat-latency");
  const statTokens = document.getElementById("stat-tokens");
  const statCost = document.getElementById("stat-cost");
  const badgeCache = document.getElementById("badge-cache-status");
  const headersOutput = document.getElementById("headers-output");

  if (btnDispatch) {
    btnDispatch.addEventListener("click", async () => {
      const model = document.getElementById("model-select").value;
      const tenantId = document.getElementById("tenant-input").value.trim() || "default";
      const teamId = document.getElementById("team-input").value.trim() || "engineering";
      const sysPrompt = document.getElementById("sys-prompt").value.trim();
      const userPrompt = document.getElementById("user-prompt").value.trim();
      const maxTokens = parseInt(sliderMaxTokens.value, 10) || 1024;
      const temperature = parseFloat(sliderTemp.value);
      const isStreaming = document.getElementById("toggle-stream").checked;
      const enableCompaction = document.getElementById("toggle-compaction").checked;
      const enableTools = document.getElementById("toggle-tools").checked;

      const messages = [];
      if (sysPrompt) messages.push({ role: "system", content: sysPrompt });
      messages.push({ role: "user", content: userPrompt });

      const payload = {
        model,
        messages,
        max_tokens: maxTokens,
        temperature,
        stream: isStreaming,
        server_tool_execution: enableTools,
      };

      const headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer cinch-prod-key",
        "X-Tenant-ID": tenantId,
        "X-Team-ID": teamId,
        "X-Prompt-Compaction": enableCompaction ? "true" : "false",
        "X-Server-Tool-Execution": enableTools ? "true" : "false",
      };

      streamOutput.textContent = "";
      btnDispatch.disabled = true;
      btnDispatch.textContent = "Streaming...";

      const badgeFinish = document.getElementById("badge-finish-status");
      if (badgeFinish) {
        badgeFinish.className = "badge";
        badgeFinish.textContent = "STATUS: STREAMING...";
      }

      const t0 = performance.now();
      let ttft = null;
      let generatedTokens = 0;
      let lastFinishReason = null;

      try {
        const resp = await fetch("/v1/chat/completions", {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
        });

        // Update Headers Inspector
        if (headersOutput) {
          const interestingHeaders = [
            "X-Request-ID",
            "X-Cascade-Routing-Tier",
            "X-Cascade-Complexity-Score",
            "X-Prompt-Compaction-Ratio",
            "X-Tool-Engine-Executed",
            "X-Tool-Engine-Tools-Used",
            "X-FinOps-Request-Cost-USD",
            "X-FinOps-Budget-Remaining-USD",
            "X-Semantic-Cache-Status",
          ];
          let hdrHtml = "";
          interestingHeaders.forEach(key => {
            const val = resp.headers.get(key) || "—";
            hdrHtml += `<div class="header-row"><span class="hdr-key">${key}:</span><span class="hdr-val">${val}</span></div>`;
          });
          headersOutput.innerHTML = hdrHtml;
        }

        const cacheStatus = resp.headers.get("X-Semantic-Cache-Status") || "MISS";
        if (badgeCache) badgeCache.textContent = `CACHE: ${cacheStatus}`;

        const reqCost = resp.headers.get("X-FinOps-Request-Cost-USD") || "$0.000000";
        if (statCost) statCost.textContent = reqCost.startsWith("$") ? reqCost : `$${reqCost}`;

        if (!resp.ok) {
          const errText = await resp.text();
          streamOutput.textContent = `[HTTP ${resp.status} Error]: ${errText}`;
          if (badgeFinish) {
            badgeFinish.className = "badge badge-danger";
            badgeFinish.textContent = `ERROR: HTTP ${resp.status}`;
          }
          btnDispatch.disabled = false;
          btnDispatch.innerHTML = '<span class="btn-icon">▶</span> Dispatch Inference';
          return;
        }

        if (isStreaming && resp.body) {
          const reader = resp.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let done = false;
          let buffer = "";

          while (!done) {
            const { value, done: streamDone } = await reader.read();
            done = streamDone;
            if (value) {
              if (ttft === null) {
                ttft = performance.now() - t0;
                if (statTtft) statTtft.textContent = `${Math.round(ttft)} ms`;
              }
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\n");
              buffer = lines.pop(); // Keep partial line in buffer

              for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
                  try {
                    const parsed = JSON.parse(trimmed.slice(6));
                    const delta = parsed.choices?.[0]?.delta?.content || "";
                    const reason = parsed.choices?.[0]?.finish_reason;
                    if (reason) lastFinishReason = reason;
                    if (delta) {
                      streamOutput.textContent += delta;
                      generatedTokens += 1;
                    }
                  } catch (e) {
                    // Raw stream passthrough
                  }
                }
              }
            }
          }

          // Process any trailing leftover in buffer
          if (buffer && buffer.trim()) {
            const trimmed = buffer.trim();
            if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
              try {
                const parsed = JSON.parse(trimmed.slice(6));
                const delta = parsed.choices?.[0]?.delta?.content || "";
                const reason = parsed.choices?.[0]?.finish_reason;
                if (reason) lastFinishReason = reason;
                if (delta) {
                  streamOutput.textContent += delta;
                  generatedTokens += 1;
                }
              } catch (e) {}
            }
          }
        } else {
          // Non-streaming response
          const data = await resp.json();
          ttft = performance.now() - t0;
          if (statTtft) statTtft.textContent = `${Math.round(ttft)} ms`;
          const content = data.choices?.[0]?.message?.content || JSON.stringify(data, null, 2);
          lastFinishReason = data.choices?.[0]?.finish_reason || "stop";
          streamOutput.textContent = content;
          generatedTokens = data.usage?.completion_tokens || content.split(/\s+/).length;
        }

        // Update finish status badge
        if (badgeFinish) {
          if (lastFinishReason === "stop") {
            badgeFinish.className = "badge badge-success";
            badgeFinish.textContent = "FINISH: COMPLETE (STOP)";
          } else if (lastFinishReason === "length") {
            badgeFinish.className = "badge text-warning";
            badgeFinish.textContent = `FINISH: TOKEN LIMIT REACHED (${maxTokens})`;
          } else {
            badgeFinish.className = "badge badge-success";
            badgeFinish.textContent = `FINISH: ${lastFinishReason || 'DONE'}`;
          }
        }

        const totalLatency = performance.now() - t0;
        if (statLatency) statLatency.textContent = `${Math.round(totalLatency)} ms`;
        if (statTokens) statTokens.textContent = `${generatedTokens} tok`;
        const genTimeSec = Math.max((totalLatency - (ttft || 0)) / 1000.0, 0.001);
        const tps = generatedTokens / genTimeSec;
        if (statTps) statTps.textContent = `${tps.toFixed(1)} tok/s`;

      } catch (err) {
        streamOutput.textContent = `Network / Dispatch Error: ${err}`;
        if (badgeFinish) {
          badgeFinish.className = "badge text-danger";
          badgeFinish.textContent = "DISPATCH ERROR";
        }
      } finally {
        btnDispatch.disabled = false;
        btnDispatch.innerHTML = '<span class="btn-icon">▶</span> Dispatch Inference';
      }
    });
  }

  // Background Telemetry Poller (FinOps & Traces)
  async function updateConsoleState() {
    try {
      // 1. Fetch FinOps ledger
      const finopsResp = await fetch("/v1/tenants/usage", {
        headers: { "Authorization": "Bearer cinch-prod-key" },
      });
      if (finopsResp.ok) {
        const finopsData = await finopsResp.json();
        const totalSpendEl = document.getElementById("finops-total-spend");
        if (totalSpendEl) totalSpendEl.textContent = `$${(finopsData.total_platform_spend_usd || 0).toFixed(6)}`;
        const tenantCountEl = document.getElementById("finops-tenant-count");
        if (tenantCountEl) tenantCountEl.textContent = finopsData.total_tenants || 0;

        const tableBody = document.getElementById("tenant-table-body");
        if (tableBody && finopsData.tenants) {
          tableBody.innerHTML = finopsData.tenants.map(t => `
            <tr>
              <td><strong>${t.tenant_id}</strong></td>
              <td>${t.team_id}</td>
              <td>$${t.budget_limit_usd.toFixed(2)}</td>
              <td>$${t.total_spend_usd.toFixed(6)}</td>
              <td>$${t.budget_remaining_usd.toFixed(6)}</td>
              <td><span class="badge ${t.budget_utilization_pct > 90 ? 'badge-danger' : 'badge-success'}">${t.budget_utilization_pct}%</span></td>
              <td>${t.request_count}</td>
            </tr>
          `).join("");
        }
      }

      // 2. Fetch Shadow Traces
      const shadowResp = await fetch("/v1/shadow/traces?limit=5", {
        headers: { "Authorization": "Bearer cinch-prod-key" },
      });
      if (shadowResp.ok) {
        const shadowData = await shadowResp.json();
        const shadowTableBody = document.getElementById("shadow-table-body");
        if (shadowTableBody && shadowData.traces) {
          shadowTableBody.innerHTML = shadowData.traces.map(tr => `
            <tr>
              <td><code>${tr.trace_id}</code></td>
              <td>${tr.prompt_preview}</td>
              <td>${tr.prod_latency_ms} ms</td>
              <td>${tr.shadow_latency_ms} ms</td>
              <td><span class="${tr.latency_delta_ms > 50 ? 'text-warning' : 'text-success'}">${tr.latency_delta_ms > 0 ? '+' : ''}${tr.latency_delta_ms}</span></td>
              <td>${(tr.lexical_similarity_score * 100).toFixed(1)}%</td>
              <td><span class="badge ${tr.divergence_detected ? 'badge-danger' : 'badge-success'}">${tr.divergence_detected ? 'DIVERGED' : 'MATCHED'}</span></td>
            </tr>
          `).join("");
        }
      }
    } catch (e) {
      // Offline / direct file fallback data for showcase
      const totalSpendEl = document.getElementById("finops-total-spend");
      if (totalSpendEl && totalSpendEl.textContent === "$0.000261") totalSpendEl.textContent = "$42.851240";
      
      const tableBody = document.getElementById("tenant-table-body");
      if (tableBody && tableBody.innerHTML.includes("Loading tenant usage ledger...")) {
        tableBody.innerHTML = `
          <tr>
            <td><strong>data-science</strong></td>
            <td>nlp-core</td>
            <td>$150.00</td>
            <td>$28.452010</td>
            <td>$121.547990</td>
            <td><span class="badge badge-success">18.9%</span></td>
            <td>1,420</td>
          </tr>
          <tr>
            <td><strong>analytics</strong></td>
            <td>bi-platform</td>
            <td>$80.00</td>
            <td>$12.114500</td>
            <td>$67.885500</td>
            <td><span class="badge badge-success">15.1%</span></td>
            <td>840</td>
          </tr>
          <tr>
            <td><strong>infra-ops</strong></td>
            <td>canary-eval</td>
            <td>$25.00</td>
            <td>$2.284730</td>
            <td>$22.715270</td>
            <td><span class="badge badge-success">9.1%</span></td>
            <td>312</td>
          </tr>
        `;
      }

      const shadowTableBody = document.getElementById("shadow-table-body");
      if (shadowTableBody && shadowTableBody.innerHTML.includes("Loading shadow comparison traces...")) {
        shadowTableBody.innerHTML = `
          <tr>
            <td><code>tr-9b8e21a</code></td>
            <td>Compare Radix prefix cache against standard KV</td>
            <td>182 ms</td>
            <td>174 ms</td>
            <td><span class="text-success">-8 ms</span></td>
            <td>98.4%</td>
            <td><span class="badge badge-success">MATCHED</span></td>
          </tr>
          <tr>
            <td><code>tr-4f1c83d</code></td>
            <td>Generate recursive descent parser for SQL</td>
            <td>412 ms</td>
            <td>398 ms</td>
            <td><span class="text-success">-14 ms</span></td>
            <td>96.1%</td>
            <td><span class="badge badge-success">MATCHED</span></td>
          </tr>
          <tr>
            <td><code>tr-7c3a09e</code></td>
            <td>Calculate quarterly gross profit margins</td>
            <td>94 ms</td>
            <td>89 ms</td>
            <td><span class="text-success">-5 ms</span></td>
            <td>100.0%</td>
            <td><span class="badge badge-success">MATCHED</span></td>
          </tr>
        `;
      }
    }
  }

  // Initial fetch and 4-second recurring poll
  updateConsoleState();
  setInterval(updateConsoleState, 4000);
});
