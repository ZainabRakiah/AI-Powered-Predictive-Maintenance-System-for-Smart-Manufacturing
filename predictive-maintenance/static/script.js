async function predict() {
  const resultEl = document.getElementById("result");
  const detailEl = document.getElementById("detail");
  const errorEl = document.getElementById("error");

  errorEl.textContent = "";
  detailEl.textContent = "";

  const data = {
    type: document.getElementById("type").value,
    air_temp: document.getElementById("air_temp").value,
    process_temp: document.getElementById("process_temp").value,
    rpm: document.getElementById("rpm").value,
    torque: document.getElementById("torque").value,
    tool_wear: document.getElementById("tool_wear").value,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await response.json().catch(() => ({}));

    if (!response.ok) {
      resultEl.textContent = "";
      errorEl.textContent = result.error || `Request failed (${response.status})`;
      return;
    }

    resultEl.textContent = result.prediction ?? "";
    if (typeof result.failure_probability === "number") {
      detailEl.textContent = `Failure probability: ${(result.failure_probability * 100).toFixed(1)}%`;
    }
  } catch (e) {
    resultEl.textContent = "";
    errorEl.textContent = "Network error — is the server running?";
  }
}
