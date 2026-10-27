import json
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView

CHART_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #1e1e2e; color: #cdd6f4; margin: 0; padding: 10px; font-family: sans-serif; }
        canvas { max-height: 100%; max-width: 100%; }
    </style>
</head>
<body>
    <canvas id="myChart"></canvas>
    <script>
        const ctx = document.getElementById('myChart');
        const spec = SPEC_JSON_PLACEHOLDER;
        
        // Simple adaptation for Chart.js
        const config = {
            type: spec.type || 'bar',
            data: spec.data || {
                labels: spec.labels || [],
                datasets: spec.datasets || []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                color: '#cdd6f4',
                plugins: {
                    legend: { labels: { color: '#cdd6f4' } }
                },
                scales: {
                    x: { ticks: { color: '#a6adc8' }, grid: { color: '#313244' } },
                    y: { ticks: { color: '#a6adc8' }, grid: { color: '#313244' } }
                }
            }
        };
        
        new Chart(ctx, config);
    </script>
</body>
</html>
"""

BASIC_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { background-color: #1e1e2e; color: #cdd6f4; margin: 0; padding: 20px; font-family: sans-serif; }
        pre { background: #11111b; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h3>Widget: WIDGET_TYPE_PLACEHOLDER</h3>
    <pre>SPEC_JSON_PLACEHOLDER</pre>
</body>
</html>
"""

class SandboxContainer(QWidget):
    def __init__(self, widget_type: str, spec: dict, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Simple header for closing or interacting
        self.close_btn = QPushButton("✖")
        self.close_btn.setObjectName("sandbox_close")
        self.close_btn.clicked.connect(self.hide)
        
        header_layout.addStretch()
        header_layout.addWidget(self.close_btn)
        
        self.view = QWebEngineView()
        
        layout.addLayout(header_layout)
        layout.addWidget(self.view)
        
        self.render_widget(widget_type, spec)
        
    def render_widget(self, widget_type: str, spec: dict):
        spec_json = json.dumps(spec)
        
        if widget_type == 'chart':
            html = CHART_TEMPLATE.replace("SPEC_JSON_PLACEHOLDER", spec_json)
        else:
            html = BASIC_TEMPLATE.replace("WIDGET_TYPE_PLACEHOLDER", widget_type)
            html = html.replace("SPEC_JSON_PLACEHOLDER", json.dumps(spec, indent=2))
            
        self.view.setHtml(html)
