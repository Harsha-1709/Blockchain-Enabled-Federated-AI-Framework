/**
 * Blockchain-Enabled Federated AI Framework
 * Front-end Dashboard Controller & Telemetry Visualization
 */

// State tracking
let convergenceChart = null;
let currentRoundData = {
  labels: ["Round 0 (Baseline)"],
  accuracy: [0.0],
  loss: [0.0]
};

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  fetchInitialStatus();
  setupEventListeners();
});

/**
 * Initializes Chart.js Real-time Convergence Line Graph
 */
function initChart() {
  const ctx = document.getElementById("convergenceChart").getContext("2d");
  convergenceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: currentRoundData.labels,
      datasets: [
        {
          label: "Global Accuracy (%)",
          data: currentRoundData.accuracy,
          borderColor: "#00f2fe",
          backgroundColor: "rgba(0, 242, 254, 0.1)",
          borderWidth: 2.5,
          pointBackgroundColor: "#00f2fe",
          pointRadius: 4,
          pointHoverRadius: 7,
          tension: 0.35,
          yAxisID: "y"
        },
        {
          label: "Cross-Entropy Loss",
          data: currentRoundData.loss,
          borderColor: "#8b5cf6",
          backgroundColor: "rgba(139, 92, 246, 0.05)",
          borderWidth: 2,
          borderDash: [5, 5],
          pointBackgroundColor: "#8b5cf6",
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(12, 18, 34, 0.95)",
          titleColor: "#00f2fe",
          bodyColor: "#f1f5f9",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          titleFont: { family: "'Plus Jakarta Sans', sans-serif", weight: "bold" },
          bodyFont: { family: "'JetBrains Mono', monospace" }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { family: "'JetBrains Mono', monospace", size: 10 } }
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          min: 0,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: {
            color: "#00f2fe",
            font: { family: "'JetBrains Mono', monospace" },
            callback: (v) => `${v}%`
          }
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: {
            color: "#8b5cf6",
            font: { family: "'JetBrains Mono', monospace" }
          }
        }
      }
    }
  });
}

/**
 * Event Listeners for User Interactivity
 */
function setupEventListeners() {
  // Attack toggle panel visibility
  const toggleAttack = document.getElementById("toggleAdversaryAttack");
  const attackPanel = document.getElementById("attackConfigPanel");
  toggleAttack.addEventListener("change", (e) => {
    attackPanel.style.display = e.target.checked ? "flex" : "none";
  });

  // Action Buttons
  document.getElementById("btnRunSingleRound").addEventListener("click", () => triggerRound(1));
  document.getElementById("btnRunFiveRounds").addEventListener("click", () => runMultipleRounds(3));
  document.getElementById("btnVerifyChain").addEventListener("click", verifyBlockchainIntegrity);
  document.getElementById("btnTamperTest").addEventListener("click", testTamperDetection);
  document.getElementById("btnResetSystem").addEventListener("click", resetSystem);
  document.getElementById("btnExportAudit").addEventListener("click", exportAuditReport);
  document.getElementById("btnRefreshMatrix").addEventListener("click", fetchInitialStatus);

  // Modal close
  document.getElementById("btnCloseModal").addEventListener("click", () => {
    document.getElementById("blockModal").style.display = "none";
  });
  window.addEventListener("click", (e) => {
    const modal = document.getElementById("blockModal");
    if (e.target === modal) modal.style.display = "none";
  });
}

/**
 * Fetches initial consortium status from the backend
 */
async function fetchInitialStatus() {
  try {
    const res = await fetch("/api/status");
    const json = await res.json();
    if (json.success) {
      updateSystemUI(json.data);
      fetchBlockchain();
    }
  } catch (err) {
    showToast("Error connecting to consortium server: " + err.message, "error");
  }
}

/**
 * Updates UI Components with System State
 */
function updateSystemUI(data) {
  const metrics = data.metrics || {};

  // Metrics Bar
  document.getElementById("metricAccuracy").innerText = `${metrics.accuracy || 0}%`;
  document.getElementById("metricF1").innerText = `${metrics.f1_score || 0}%`;
  document.getElementById("metricLoss").innerText = metrics.loss !== undefined ? metrics.loss : "--";
  document.getElementById("currentRoundDisplay").innerText = `Round #${data.current_round}`;

  // Update Baseline in Chart if Round 0
  if (data.current_round === 0) {
    currentRoundData.labels = ["Baseline (R0)"];
    currentRoundData.accuracy = [metrics.accuracy || 0];
    currentRoundData.loss = [metrics.loss || 0];
    convergenceChart.data.labels = currentRoundData.labels;
    convergenceChart.data.datasets[0].data = currentRoundData.accuracy;
    convergenceChart.data.datasets[1].data = currentRoundData.loss;
    convergenceChart.update();
  }

  // Render Client Silos
  renderClientsList(data.clients);

  // Render Class Attack Bars
  if (metrics.per_class_accuracy) {
    renderClassAccuracyBars(metrics.per_class_accuracy);
  }
}

/**
 * Renders Client Silo Cards
 */
function renderClientsList(clients) {
  const container = document.getElementById("clientsList");
  container.innerHTML = "";

  Object.entries(clients).forEach(([cid, client]) => {
    const card = document.createElement("div");
    card.className = "client-item-card";

    const topAttacksHtml = (client.attack_profile || [])
      .map(att => `<span class="threat-tag">${att}</span>`)
      .join("");

    card.innerHTML = `
      <div class="client-top-row">
        <div class="client-name-group">
          <span class="dot ${cid.includes('Hospital') ? 'hospital' : cid.includes('Bank') ? 'bank' : 'iot'}"></span>
          <div>
            <div class="client-name">${client.name}</div>
            <div class="client-org">${client.org_type}</div>
          </div>
        </div>
        <span class="badge badge-emerald">Online / Verified</span>
      </div>
      <div class="client-stats-row">
        <span>Dataset: <strong>${client.sample_count} flows</strong></span>
        <span>Participated: <strong>${client.rounds_participated} rounds</strong></span>
      </div>
      <div class="client-threat-tags">
        ${topAttacksHtml}
      </div>
    `;
    container.appendChild(card);
  });
}

/**
 * Renders Per-Class Detection Rates (15 Classes)
 */
function renderClassAccuracyBars(perClassMap) {
  const container = document.getElementById("attackBarsGrid");
  container.innerHTML = "";

  Object.entries(perClassMap).forEach(([className, rate]) => {
    const pct = Math.round(rate * 100);
    const item = document.createElement("div");
    item.className = "attack-bar-item";
    item.innerHTML = `
      <div class="attack-bar-labels">
        <span class="attack-name">${className}</span>
        <span class="attack-pct">${pct}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(item);
  });
}

/**
 * Triggers a Federated Training Round via POST /api/train-round
 */
async function triggerRound(epochsCount = 1) {
  const btn = document.getElementById("btnRunSingleRound");
  btn.disabled = true;
  btn.innerHTML = `<span class="pulse-dot"></span> Training & Aggregating...`;

  const payload = {
    epochs: parseInt(document.getElementById("inputEpochs").value),
    lr: parseFloat(document.getElementById("inputLr").value),
    dp_enabled: document.getElementById("toggleDifferentialPrivacy").checked,
    simulate_attack: document.getElementById("toggleAdversaryAttack").checked,
    attack_type: document.getElementById("selectAttackType").value,
    byzantine_method: document.getElementById("selectByzantineMethod").value
  };

  try {
    const res = await fetch("/api/train-round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.detail || "Round execution failed");

    const r = data.round;
    const metrics = r.metrics;

    // Update charts
    currentRoundData.labels.push(`Round ${r.round_number}`);
    currentRoundData.accuracy.push(metrics.accuracy);
    currentRoundData.loss.push(metrics.loss);
    convergenceChart.update();

    // Update KPI counters
    document.getElementById("metricAccuracy").innerText = `${metrics.accuracy}%`;
    document.getElementById("metricF1").innerText = `${metrics.f1_score}%`;
    document.getElementById("metricLoss").innerText = metrics.loss;
    document.getElementById("currentRoundDisplay").innerText = `Round #${r.round_number}`;

    // Update DP Budget
    if (r.privacy_accounting && r.privacy_accounting.epsilon) {
      document.getElementById("metricEpsilon").innerText = `ε = ${r.privacy_accounting.epsilon}`;
    }

    // Refresh class bars
    if (metrics.per_class_accuracy) {
      renderClassAccuracyBars(metrics.per_class_accuracy);
    }

    // Check for Byzantine rejections
    const rejected = r.byzantine_defense.rejected_clients || [];
    if (rejected.length > 0) {
      showToast(
        `🛡️ Byzantine Shield Active: Slashed & Dropped [${rejected.join(", ")}] on-chain!`,
        "warning"
      );
      highlightAdversaryClient(rejected);
    } else {
      showToast(`✅ Round #${r.round_number} completed. Block #${r.block.index} sealed!`, "success");
    }

    // Refresh blockchain explorer
    fetchBlockchain();

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Execute Round`;
  }
}

/**
 * Runs multiple sequential rounds
 */
async function runMultipleRounds(count = 3) {
  const btn = document.getElementById("btnRunFiveRounds");
  btn.disabled = true;
  btn.innerText = `Running ${count} Rounds...`;

  for (let i = 1; i <= count; i++) {
    showToast(`Initiating Round ${i} of ${count}...`, "info");
    await triggerRound();
    // Brief delay between rounds
    await new Promise(res => setTimeout(res, 500));
  }

  btn.disabled = false;
  btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="13 17 18 12 13 7"></polyline><polyline points="6 17 11 12 6 7"></polyline></svg> Run 3 Rounds`;
}

/**
 * Highlights adversary node in client list
 */
function highlightAdversaryClient(rejectedClients) {
  const container = document.getElementById("clientsList");
  // Check if adversary card already rendered
  if (!document.getElementById("adversaryCard")) {
    const card = document.createElement("div");
    card.id = "adversaryCard";
    card.className = "client-item-card adversary";
    card.innerHTML = `
      <div class="client-top-row">
        <div class="client-name-group">
          <span class="dot" style="background: #ef4444; box-shadow: 0 0 6px #ef4444;"></span>
          <div>
            <div class="client-name text-red">${rejectedClients[0]}</div>
            <div class="client-org">Unverified / Adversarial</div>
          </div>
        </div>
        <span class="badge badge-danger">Slashed & Filtered</span>
      </div>
      <div class="client-stats-row">
        <span>Anomaly: <strong>Cosine Outlier (< 0.2)</strong></span>
        <span>Status: <strong>Excluded from FedAvg</strong></span>
      </div>
    `;
    container.appendChild(card);
  }
}

/**
 * Fetches and displays the Blockchain Ledger blocks
 */
async function fetchBlockchain() {
  try {
    const res = await fetch("/api/blockchain");
    const data = await res.json();
    if (!data.success) return;

    document.getElementById("blockCountDisplay").innerText = `${data.total_blocks} Blocks`;

    const chainView = document.getElementById("blockchainChain");
    chainView.innerHTML = "";

    data.blocks.forEach((block) => {
      const card = document.createElement("div");
      card.className = "block-card";
      card.onclick = () => openBlockModal(block);

      card.innerHTML = `
        <div class="block-header">
          <span class="block-index-title">BLOCK #${block.index}</span>
          <span class="block-timestamp">${block.time_formatted.split(" ")[1]}</span>
        </div>
        <div class="block-row">
          <span class="block-field-label">Block Hash</span>
          <span class="block-hash-code" title="${block.block_hash}">${block.block_hash}</span>
        </div>
        <div class="block-row">
          <span class="block-field-label">Prev Hash</span>
          <span class="block-hash-code" title="${block.previous_hash}">${block.previous_hash}</span>
        </div>
        <div class="block-row">
          <span class="block-field-label">Merkle Root</span>
          <span class="block-hash-code" title="${block.merkle_root}">${block.merkle_root}</span>
        </div>
        <div class="block-footer">
          <span>Txs: <strong>${block.transaction_count}</strong></span>
          <span>Acc: <strong>${block.metrics.accuracy || 0}%</strong></span>
          <span class="text-cyan">Inspect →</span>
        </div>
      `;
      chainView.appendChild(card);
    });

    // Auto-scroll to latest mined block
    chainView.scrollLeft = chainView.scrollWidth;

  } catch (err) {
    console.error("Blockchain fetch error:", err);
  }
}

/**
 * Opens Block Inspection Modal
 */
function openBlockModal(block) {
  document.getElementById("modalBlockTitle").innerText = `Block #${block.index} (Round ${block.round_number})`;
  const content = document.getElementById("modalBlockContent");

  const txHtml = (block.transactions || []).map(tx => `
    <div style="padding: 0.6rem; margin-bottom: 0.5rem; background: rgba(0,0,0,0.25); border-radius: 6px; border-left: 3px solid ${tx.status === 'ACCEPTED' ? '#10b981' : '#ef4444'};">
      <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
        <strong>${tx.client_id}</strong>
        <span class="${tx.status === 'ACCEPTED' ? 'text-emerald' : 'text-red'}">${tx.status}</span>
      </div>
      <div style="font-family: monospace; font-size: 0.68rem; color: #94a3b8; margin-top: 0.2rem;">
        Tx Hash: ${tx.tx_hash}<br>
        Weight Hash: ${tx.weight_hash}<br>
        Signature: ${tx.signature}
      </div>
    </div>
  `).join("");

  content.innerHTML = `
    <div class="modal-section">
      <span class="block-field-label">Full Cryptographic Block Hash</span>
      <div class="modal-code-block">${block.block_hash}</div>
    </div>

    <div class="modal-section">
      <span class="block-field-label">Previous Block Pointer</span>
      <div class="modal-code-block">${block.previous_hash}</div>
    </div>

    <div class="modal-section">
      <span class="block-field-label">Merkle Root & Validator Signature</span>
      <div class="modal-code-block">
        Merkle Root: ${block.merkle_root}<br>
        Validator ID: ${block.validator_id}<br>
        PoA Sig: ${block.validator_signature ? block.validator_signature.substring(0, 64) + '...' : 'GENESIS_SEALED'}
      </div>
    </div>

    <div class="modal-section">
      <span class="block-field-label">Transactions (${block.transaction_count})</span>
      <div style="max-height: 200px; overflow-y: auto;">
        ${txHtml}
      </div>
    </div>
  `;

  document.getElementById("blockModal").style.display = "flex";
}

/**
 * Cryptographically validates all blocks in the ledger
 */
async function verifyBlockchainIntegrity() {
  try {
    const res = await fetch("/api/blockchain/verify", { method: "POST" });
    const data = await res.json();
    if (data.success && data.report.is_valid) {
      showToast("🛡️ Audit Passed: All SHA-256 block hashes, Merkle roots, and ECDSA signatures are valid!", "success");
      document.getElementById("ledgerStatusText").innerText = "Cryptographically Valid";
      document.getElementById("ledgerStatusText").className = "status-value text-emerald";
    } else {
      showToast(`⚠️ Integrity Alert: ${data.report.reason}`, "error");
      document.getElementById("ledgerStatusText").innerText = "TAMPER DETECTED";
      document.getElementById("ledgerStatusText").className = "status-value text-red";
    }
  } catch (err) {
    showToast(`Verification error: ${err.message}`, "error");
  }
}

/**
 * Simulates a block tamper attempt to prove cryptographic detection
 */
async function testTamperDetection() {
  try {
    const res = await fetch("/api/blockchain/tamper-test", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      showToast("⚠️ Tamper Simulated in Block #1! Blockchain integrity check triggered...", "warning");
      document.getElementById("ledgerStatusText").innerText = "TAMPER DETECTED";
      document.getElementById("ledgerStatusText").className = "status-value text-red";
      fetchBlockchain();
    } else {
      showToast(data.detail || "Tamper test failed", "error");
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

/**
 * Resets the consortium system to genesis
 */
async function resetSystem() {
  if (!confirm("Are you sure you want to reset the consortium and reinitialize genesis?")) return;
  try {
    const res = await fetch("/api/reset", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      showToast("Consortium & Blockchain Ledger re-initialized to Genesis.", "info");
      currentRoundData = {
        labels: ["Baseline (R0)"],
        accuracy: [0.0],
        loss: [0.0]
      };
      convergenceChart.data.labels = currentRoundData.labels;
      convergenceChart.data.datasets[0].data = currentRoundData.accuracy;
      convergenceChart.data.datasets[1].data = currentRoundData.loss;
      convergenceChart.update();
      document.getElementById("adversaryCard")?.remove();
      document.getElementById("ledgerStatusText").innerText = "Cryptographically Valid";
      document.getElementById("ledgerStatusText").className = "status-value text-emerald";
      fetchInitialStatus();
    }
  } catch (err) {
    showToast(`Reset error: ${err.message}`, "error");
  }
}

/**
 * Exports full forensic audit report
 */
function exportAuditReport() {
  window.open("/api/export-audit", "_blank");
  showToast("📥 Exported GDPR/HIPAA Forensic Audit Log (consortium_audit_report.json)", "success");
}

/**
 * Toast Notification Utility
 */
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'warning' ? '⚠️' : type === 'error' ? '❌' : 'ℹ️'}</span>
    <div>${message}</div>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
