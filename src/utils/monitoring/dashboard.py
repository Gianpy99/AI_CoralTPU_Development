"""
Simple web dashboard for monitoring the trading system
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DashboardServer:
    """Simple web dashboard for real-time monitoring"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.app = None
        self.server = None
        self.websocket_connections = []
        self.trading_stats = {}
        self.is_running = False
        
        if FASTAPI_AVAILABLE:
            self._setup_app()
        else:
            logger.warning("FastAPI not available, dashboard disabled")
    
    def _setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(title="Coral TPU Trading Dashboard")
        
        # Setup routes
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/stats")
        async def get_stats():
            return self.trading_stats
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coral TPU Trading Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #0f1419;
                    color: #ffffff;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .stat-card {
                    background: #1e2328;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #333;
                    text-align: center;
                }
                .stat-value {
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .stat-label {
                    color: #888;
                    font-size: 0.9em;
                }
                .positive { color: #00d4aa; }
                .negative { color: #ff6b6b; }
                .neutral { color: #ffd93d; }
                .chart-container {
                    background: #1e2328;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #333;
                    margin-bottom: 20px;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 10px;
                }
                .status-online { background-color: #00d4aa; }
                .status-offline { background-color: #ff6b6b; }
                .log-container {
                    background: #1e2328;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #333;
                    max-height: 300px;
                    overflow-y: auto;
                }
                .log-entry {
                    margin: 5px 0;
                    padding: 5px;
                    border-left: 3px solid #667eea;
                    padding-left: 10px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Coral TPU Crypto Trading System</h1>
                <div>
                    <span class="status-indicator" id="statusIndicator"></span>
                    <span id="connectionStatus">Connecting...</span>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Portfolio Value</div>
                    <div class="stat-value" id="portfolioValue">$0.00</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total P&L</div>
                    <div class="stat-value" id="totalPnl">$0.00</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Active Positions</div>
                    <div class="stat-value" id="activePositions">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Win Rate</div>
                    <div class="stat-value" id="winRate">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Signals Generated</div>
                    <div class="stat-value" id="signalsGenerated">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">TPU Status</div>
                    <div class="stat-value" id="tpuStatus">-</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Portfolio Performance</h3>
                <canvas id="portfolioChart" width="400" height="200"></canvas>
            </div>
            
            <div class="log-container">
                <h3>Recent Activity</h3>
                <div id="logEntries"></div>
            </div>
            
            <script>
                // WebSocket connection
                let ws = null;
                let chart = null;
                let portfolioData = [];
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = function() {
                        document.getElementById('connectionStatus').textContent = 'Connected';
                        document.getElementById('statusIndicator').className = 'status-indicator status-online';
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    ws.onclose = function() {
                        document.getElementById('connectionStatus').textContent = 'Disconnected';
                        document.getElementById('statusIndicator').className = 'status-indicator status-offline';
                        // Attempt to reconnect after 5 seconds
                        setTimeout(connectWebSocket, 5000);
                    };
                }
                
                function updateDashboard(data) {
                    // Update stats
                    document.getElementById('portfolioValue').textContent = `$${data.total_portfolio_value?.toFixed(2) || '0.00'}`;
                    
                    const totalPnl = data.total_pnl || 0;
                    const pnlElement = document.getElementById('totalPnl');
                    pnlElement.textContent = `$${totalPnl.toFixed(2)}`;
                    pnlElement.className = `stat-value ${totalPnl >= 0 ? 'positive' : 'negative'}`;
                    
                    document.getElementById('activePositions').textContent = data.active_positions || 0;
                    document.getElementById('winRate').textContent = `${((data.win_rate || 0) * 100).toFixed(1)}%`;
                    document.getElementById('signalsGenerated').textContent = data.signals_generated || 0;
                    document.getElementById('tpuStatus').textContent = data.tpu_status || 'Unknown';
                    
                    // Update chart
                    if (data.portfolio_history) {
                        updateChart(data.portfolio_history);
                    }
                    
                    // Update logs
                    if (data.recent_logs) {
                        updateLogs(data.recent_logs);
                    }
                }
                
                function updateChart(portfolioHistory) {
                    if (!chart) {
                        const ctx = document.getElementById('portfolioChart').getContext('2d');
                        chart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [{
                                    label: 'Portfolio Value',
                                    data: [],
                                    borderColor: '#667eea',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    tension: 0.4
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: {
                                            color: '#ffffff'
                                        }
                                    }
                                },
                                scales: {
                                    x: {
                                        ticks: {
                                            color: '#888'
                                        },
                                        grid: {
                                            color: '#333'
                                        }
                                    },
                                    y: {
                                        ticks: {
                                            color: '#888'
                                        },
                                        grid: {
                                            color: '#333'
                                        }
                                    }
                                }
                            }
                        });
                    }
                    
                    // Update chart data
                    chart.data.labels = portfolioHistory.map(p => p.timestamp);
                    chart.data.datasets[0].data = portfolioHistory.map(p => p.value);
                    chart.update();
                }
                
                function updateLogs(logs) {
                    const logContainer = document.getElementById('logEntries');
                    logContainer.innerHTML = '';
                    
                    logs.forEach(log => {
                        const logElement = document.createElement('div');
                        logElement.className = 'log-entry';
                        logElement.innerHTML = `<span style="color: #888;">${log.timestamp}</span> ${log.message}`;
                        logContainer.appendChild(logElement);
                    });
                }
                
                // Initialize
                connectWebSocket();
                
                // Fetch initial data
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => updateDashboard(data))
                    .catch(console.error);
            </script>
        </body>
        </html>
        """
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        
        try:
            while True:
                # Keep connection alive
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast data to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update trading statistics"""
        self.trading_stats = {
            **stats,
            'last_update': datetime.now().isoformat(),
            'tpu_status': 'Connected',  # This would come from TPU status check
            'recent_logs': [
                {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': f"Portfolio: ${stats.get('total_portfolio_value', 0):.2f}"
                },
                {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': f"Active positions: {stats.get('active_positions', 0)}"
                }
            ]
        }
        
        # Broadcast to WebSocket clients
        if self.is_running:
            asyncio.create_task(self.broadcast_update(self.trading_stats))
    
    async def start(self):
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE:
            logger.warning("Cannot start dashboard: FastAPI not available")
            return
        
        logger.info(f"Starting dashboard server on port {self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning"  # Reduce uvicorn logging
        )
        
        self.server = uvicorn.Server(config)
        self.is_running = True
        
        # Start server in background
        asyncio.create_task(self.server.serve())
        
        logger.success(f"Dashboard available at http://localhost:{self.port}")
    
    async def stop(self):
        """Stop the dashboard server"""
        if self.server:
            self.is_running = False
            await self.server.shutdown()
            logger.info("Dashboard server stopped")


# Standalone dashboard for testing
if __name__ == "__main__":
    import sys
    import random
    
    async def test_dashboard():
        """Test the dashboard with dummy data"""
        dashboard = DashboardServer(8000)
        await dashboard.start()
        
        # Generate dummy data updates
        try:
            for i in range(100):
                dummy_stats = {
                    'total_portfolio_value': 10000 + random.uniform(-1000, 2000),
                    'total_pnl': random.uniform(-500, 1000),
                    'active_positions': random.randint(0, 5),
                    'win_rate': random.uniform(0.4, 0.8),
                    'signals_generated': i * 2,
                    'successful_trades': random.randint(0, 20),
                    'failed_trades': random.randint(0, 10)
                }
                
                dashboard.update_stats(dummy_stats)
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Stopping dashboard test...")
        finally:
            await dashboard.stop()
    
    if FASTAPI_AVAILABLE:
        asyncio.run(test_dashboard())
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
