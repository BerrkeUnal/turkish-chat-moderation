from __future__ import annotations

APP_CSS = """
:root {
    --app-background: #f5f7fb;
    --panel-background: #ffffff;
    --panel-border: #dfe6ef;
    --primary-color: #2563eb;
    --heading-color: #0f172a;
    --text-color: #475569;
    --muted-color: #64748b;
}

body,
.gradio-container {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

body {
    background: var(--app-background) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 16px 20px 24px !important;
}

.page-header {
    margin-bottom: 8px;
    padding: 2px 2px 12px;
    border-bottom: 1px solid var(--panel-border);
}

.page-header .eyebrow {
    margin-bottom: 4px;
    color: var(--primary-color);
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.12em;
}

.header-row {
    display: flex;
    gap: 28px;
    align-items: baseline;
    justify-content: space-between;
}

.page-header h1 {
    flex: 0 0 auto;
    margin: 0;
    color: var(--heading-color);
    font-size: 1.72rem;
    font-weight: 750;
    letter-spacing: -0.035em;
}

.page-header p {
    max-width: 650px;
    margin: 0;
    color: var(--text-color);
    font-size: 0.86rem;
    line-height: 1.4;
    text-align: right;
}

.tab-shell {
    gap: 12px !important;
    padding-top: 4px !important;
}

.tab-heading {
    display: flex;
    gap: 18px;
    align-items: baseline;
    padding: 2px 1px 4px;
}

.tab-heading h2 {
    flex: 0 0 auto;
    margin: 0;
    color: var(--heading-color);
    font-size: 1.08rem;
    font-weight: 700;
}

.tab-heading p {
    margin: 0;
    color: var(--text-color);
    font-size: 0.84rem;
}

.moderation-controls {
    display: grid !important;
    width: 100% !important;
    max-width: none !important;
    align-self: stretch !important;
    grid-template-columns:
        minmax(320px, 0.8fr)
        minmax(0, 2.2fr);
    gap: 12px !important;
    align-items: end !important;
}

.moderation-controls > *,
.moderation-workspace > *,
.compare-controls > * {
    width: auto !important;
    min-width: 0 !important;
}

.compact-actions {
    gap: 10px !important;
}

#primary-action,
#compare-action {
    font-weight: 650 !important;
}

.moderation-workspace {
    display: grid !important;
    grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr);
    gap: 12px !important;
    align-items: stretch !important;
}

.workspace-panel {
    gap: 9px !important;
    padding: 13px !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 13px !important;
    background: var(--panel-background) !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
}

.section-label {
    color: var(--heading-color);
    font-size: 0.82rem;
    font-weight: 700;
}

#chat-window {
    height: 220px !important;
    min-height: 220px !important;
    max-height: 220px !important;
    border-color: var(--panel-border) !important;
    border-radius: 10px !important;
}

.result-card {
    min-height: 116px;
    padding: 15px;
    border: 1px solid var(--panel-border);
    border-radius: 11px;
    background: #f8fafc;
}

.result-card.allowed {
    border-color: #bbf7d0;
    background: #f0fdf4;
}

.result-card.blocked {
    border-color: #fecaca;
    background: #fef2f2;
}

.result-card.warning {
    border-color: #fde68a;
    background: #fffbeb;
}

.result-card-header {
    display: flex;
    gap: 9px;
    align-items: center;
    margin-bottom: 8px;
}

.status-badge {
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.035em;
}

.status-badge.allowed {
    background: #dcfce7;
    color: #047857;
}

.status-badge.blocked {
    background: #fee2e2;
    color: #b91c1c;
}

.result-title {
    color: var(--heading-color);
    font-size: 0.98rem;
    font-weight: 700;
}

.result-card p {
    margin: 0 0 10px;
    color: var(--text-color);
    font-size: 0.83rem;
    line-height: 1.4;
}

.prediction-metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 7px;
    margin-bottom: 8px;
}

.prediction-metrics span {
    padding: 7px 8px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.72);
    color: var(--muted-color);
    font-size: 0.72rem;
}

.prediction-metrics strong {
    display: block;
    color: var(--heading-color);
    font-size: 0.82rem;
}

.result-model,
.loaded-models {
    color: var(--muted-color);
    font-size: 0.76rem;
    line-height: 1.4;
}

.compare-controls {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 180px;
    gap: 12px !important;
    align-items: end !important;
}

#compare-action {
    min-height: 46px;
    margin-bottom: 1px;
}

.comparison-band {
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    min-height: 48px;
    padding: 10px 13px;
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    background: #f8fafc;
    color: var(--muted-color);
    font-size: 0.8rem;
}

.comparison-band strong {
    color: var(--heading-color);
}

.comparison-band.complete {
    border-color: #bfdbfe;
    background: #eff6ff;
}

#comparison-table table,
#detailed-results-table table {
    font-family: inherit !important;
    font-size: 0.8rem !important;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 11px;
}

.kpi-card {
    padding: 14px 15px;
    border: 1px solid var(--panel-border);
    border-radius: 11px;
    background: var(--panel-background);
}

.kpi-label {
    margin-bottom: 6px;
    color: var(--muted-color);
    font-size: 0.72rem;
    font-weight: 650;
}

.kpi-value {
    margin-bottom: 3px;
    color: var(--heading-color);
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: -0.03em;
}

.kpi-caption {
    color: var(--muted-color);
    font-size: 0.74rem;
}

.deployment-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    align-items: center;
    padding: 10px 12px;
    border: 1px solid #bfdbfe;
    border-radius: 9px;
    background: #eff6ff;
    color: var(--text-color);
    font-size: 0.8rem;
}

.deployment-label {
    padding: 3px 7px;
    border-radius: 999px;
    background: var(--primary-color);
    color: #ffffff;
    font-size: 0.64rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.benchmark-table-wrapper {
    overflow-x: auto;
    border: 1px solid var(--panel-border);
    border-radius: 11px;
    background: var(--panel-background);
}

.benchmark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.79rem;
}

.benchmark-table th {
    padding: 9px 11px;
    border-bottom: 1px solid var(--panel-border);
    background: #f8fafc;
    color: var(--muted-color);
    text-align: left;
    font-size: 0.66rem;
    font-weight: 750;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    white-space: nowrap;
}

.benchmark-table td {
    padding: 10px 11px;
    border-bottom: 1px solid #eef2f7;
    color: #334155;
    white-space: nowrap;
}

.benchmark-table tr:last-child td {
    border-bottom: none;
}

.benchmark-table tbody tr:hover {
    background: #f8fafc;
}

.benchmark-table .deployment-row,
.benchmark-table .deployment-row:hover {
    background: #eff6ff;
}

.model-name {
    color: var(--heading-color);
    font-weight: 650;
}

.model-badges {
    margin-top: 3px;
}

.model-badge {
    display: inline-block;
    margin-right: 4px;
    padding: 2px 6px;
    border-radius: 999px;
    font-size: 0.61rem;
    font-weight: 750;
}

.best-badge {
    background: #ecfdf5;
    color: #047857;
}

.deployment-badge {
    background: #dbeafe;
    color: #1d4ed8;
}

.metric-note {
    margin-top: 12px;
    padding: 11px 13px;
    border-left: 3px solid var(--primary-color);
    border-radius: 7px;
    background: #eff6ff;
}

.about-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(0, 1.45fr);
    gap: 12px;
}

.about-card,
.fact-card,
.responsible-use {
    border: 1px solid var(--panel-border);
    border-radius: 11px;
    background: var(--panel-background);
}

.about-card {
    padding: 18px;
}

.about-label {
    margin-bottom: 6px;
    color: var(--primary-color);
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.about-card h3,
.responsible-use h3 {
    margin: 0 0 8px;
    color: var(--heading-color);
    font-size: 1rem;
}

.about-card p {
    margin: 0;
    color: var(--text-color);
    font-size: 0.82rem;
    line-height: 1.55;
}

.fact-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.fact-card {
    padding: 13px 14px;
}

.fact-card span,
.fact-card strong {
    display: block;
}

.fact-card span {
    margin-bottom: 5px;
    color: var(--muted-color);
    font-size: 0.68rem;
}

.fact-card strong {
    color: var(--heading-color);
    font-size: 0.82rem;
}

.responsible-use {
    display: grid;
    grid-template-columns: minmax(220px, 0.7fr) minmax(0, 1.5fr);
    gap: 20px;
    margin-top: 12px;
    padding: 16px 18px;
}

.responsible-use ul {
    display: grid;
    gap: 6px;
    margin: 0;
    padding-left: 18px;
    color: var(--text-color);
    font-size: 0.78rem;
    line-height: 1.45;
}

.footer-note {
    margin-top: 14px;
    color: #94a3b8;
    text-align: center;
    font-size: 0.72rem;
}

@media (max-width: 820px) {
    .header-row,
    .tab-heading {
        display: block;
    }

    .page-header p {
        margin-top: 5px;
        text-align: left;
    }

    .tab-heading p {
        margin-top: 4px;
    }

    .moderation-controls,
    .moderation-workspace,
    .compare-controls,
    .about-layout,
    .responsible-use {
        grid-template-columns: 1fr;
    }

    #compare-action {
        margin: 0;
    }
}

@media (max-width: 620px) {
    .gradio-container {
        padding: 14px 12px 22px !important;
    }

    .page-header h1 {
        font-size: 1.46rem;
    }

    .kpi-grid,
    .fact-grid,
    .prediction-metrics {
        grid-template-columns: 1fr;
    }
}
"""
