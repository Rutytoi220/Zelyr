import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Database, Server, Zap } from 'lucide-react';

interface TelemetryData {
  model: string;
  cpu: string;
  ram: string;
  status: 'Online' | 'Offline' | 'Processing';
  uptime: string;
}

const TelemetrySidebar: React.FC = () => {
  const [telemetry, setTelemetry] = useState<TelemetryData>({
    model: 'Connecting...',
    cpu: '0%',
    ram: '0 GB',
    status: 'Offline',
    uptime: '00:00:00',
  });

  const [rawCpu, setRawCpu] = useState(0);
  const [rawRam, setRawRam] = useState(0);

  useEffect(() => {
    let uptimeSeconds = 0;
    
    // Uptime tick
    const uptimeInterval = setInterval(() => {
      uptimeSeconds++;
      const h = Math.floor(uptimeSeconds / 3600);
      const m = Math.floor((uptimeSeconds % 3600) / 60);
      const s = uptimeSeconds % 60;
      setTelemetry((prev) => ({
        ...prev,
        uptime: `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`,
      }));
    }, 1000);

    // Backend polling
    const fetchTelemetry = async () => {
      try {
        const res = await fetch('http://127.0.0.1:49103/api/status');
        if (!res.ok) throw new Error('Status fetch failed');
        const data = await res.json();
        
        setRawCpu(data.cpu_percent || 0);
        setRawRam(data.ram_percent || 0);

        setTelemetry((prev) => ({
          ...prev,
          model: data.model || 'Unknown',
          cpu: `${(data.cpu_percent || 0).toFixed(1)}%`,
          ram: `${(data.ram_percent || 0).toFixed(1)}% Usage`,
          status: data.engine_running ? 'Online' : 'Processing',
        }));
      } catch (err) {
        setTelemetry((prev) => ({
          ...prev,
          status: 'Offline',
          cpu: '0%',
          ram: '0% Usage',
        }));
        setRawCpu(0);
        setRawRam(0);
      }
    };

    // Initial fetch and interval setup
    fetchTelemetry();
    const pollInterval = setInterval(fetchTelemetry, 2000);

    return () => {
      clearInterval(uptimeInterval);
      clearInterval(pollInterval);
    };
  }, []);

  return (
    <div className="w-80 h-full bg-zelyr-panel border-l border-zelyr-border p-6 flex flex-col gap-6 shadow-xl">
      <div className="flex items-center gap-3 border-b border-zelyr-border pb-4">
        <Zap className="text-zelyr-accent w-6 h-6" />
        <h2 className="text-xl font-bold tracking-wider text-zelyr-text">Zelyr V2</h2>
      </div>

      <div className="flex flex-col gap-4">
        <div className="bg-black/50 p-4 rounded-lg border border-zelyr-border/50 flex flex-col gap-1">
          <div className="text-xs text-zelyr-muted uppercase tracking-widest font-semibold flex items-center gap-2">
            <Server className="w-4 h-4" /> System Status
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div className={`w-2 h-2 rounded-full ${telemetry.status === 'Online' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-yellow-500'}`} />
            <span className="text-sm font-medium">{telemetry.status}</span>
          </div>
          <div className="text-xs text-zelyr-muted mt-2 font-mono">Uptime: {telemetry.uptime}</div>
        </div>

        <div className="bg-black/50 p-4 rounded-lg border border-zelyr-border/50 flex flex-col gap-2">
          <div className="text-xs text-zelyr-muted uppercase tracking-widest font-semibold flex items-center gap-2 mb-1">
            <Cpu className="w-4 h-4" /> Inference Engine
          </div>
          <div className="text-sm font-mono text-zelyr-accent">{telemetry.model}</div>
        </div>

        <div className="bg-black/50 p-4 rounded-lg border border-zelyr-border/50 flex flex-col gap-4">
          <div className="text-xs text-zelyr-muted uppercase tracking-widest font-semibold flex items-center gap-2">
            <Activity className="w-4 h-4" /> Hardware Telemetry
          </div>
          
          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-xs">
              <span className="text-zelyr-muted">CPU Usage</span>
              <span className="font-mono">{telemetry.cpu}</span>
            </div>
            <div className="w-full bg-zinc-900 rounded-full h-1.5">
              <div className="bg-zelyr-accent h-1.5 rounded-full transition-all duration-500" style={{ width: `${rawCpu}%` }}></div>
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-xs">
              <span className="text-zelyr-muted flex items-center gap-1"><Database className="w-3 h-3"/> Memory</span>
              <span className="font-mono">{telemetry.ram}</span>
            </div>
            <div className="w-full bg-zinc-900 rounded-full h-1.5">
              <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${rawRam}%` }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TelemetrySidebar;
