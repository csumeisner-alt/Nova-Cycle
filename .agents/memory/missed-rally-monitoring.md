---
name: Missed-rally monitoring
description: Rally detection must remain meaningful when the signal pipeline produces no actionable rows
---

Treat a window with no BUY/SELL signals as one measurable HOLD gap rather than returning an empty missed-rally result. Detect distinct qualifying rally episodes within that gap, while keeping the rate gap-based.

**Why:** A strong VOO rally occurred while the signal tables were empty, and the prior detector returned zero by design. That made a complete signal-pipeline failure look like a clean performance period.

**How to apply:** Keep market-outcome monitoring independent from trade-cycle generation. Do not weaken signal quality gates to improve missed-rally statistics; use the statistics to expose degraded or stale prediction behavior.