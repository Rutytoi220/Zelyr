import React from 'react';
import TerminalMirror from './TerminalMirror';
import TelemetrySidebar from './TelemetrySidebar';
import SwarmVisualizer from './SwarmVisualizer';
import AutomationDashboard from './components/AutomationDashboard';

const App: React.FC = () => {
  return (
    <div className="flex h-screen w-screen bg-zelyr-bg text-zelyr-text overflow-hidden">
      {/* Main Terminal Area */}
      <div className="flex-1 flex flex-col p-6 pr-0">
        <header className="mb-4">
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Zelyr <span className="text-zelyr-accent text-sm bg-zelyr-accent/10 px-2 py-0.5 rounded border border-zelyr-accent/20">v2.0 Terminal</span>
          </h1>
          <p className="text-zelyr-muted text-sm mt-1">Live inference logs and system orchestration</p>
        </header>
        
        <main className="flex-1 min-h-0 pb-6 pr-6 flex flex-col">
          <AutomationDashboard wsUrl="ws://127.0.0.1:49103/ws/events" />
          <SwarmVisualizer wsUrl="ws://127.0.0.1:49103/ws/events" />
          <TerminalMirror wsUrl="ws://127.0.0.1:49103/ws/events" />
        </main>
      </div>

      {/* Sidebar */}
      <TelemetrySidebar />
    </div>
  );
};

export default App;
