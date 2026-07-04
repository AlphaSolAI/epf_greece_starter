import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { TimeSeriesPoint, TargetVariable } from '../types';
import { MODEL_COLORS } from '../constants';
import { Eye, EyeOff, ZoomIn, Maximize } from 'lucide-react';

interface PredictionChartProps {
  data: TimeSeriesPoint[];
  target: TargetVariable;
}

export const PredictionChart: React.FC<PredictionChartProps> = ({ data, target }) => {
  // Extract all model keys from the first data point (excluding timestamp and actual)
  const allKeys = data.length > 0 
    ? Object.keys(data[0]).filter(k => k !== 'timestamp' && k !== 'actual')
    : [];

  // State to track visible lines. 'Actual' is always visible initially.
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set(['Actual', ...allKeys]));
  
  // State for Y-Axis scaling mode (Auto-scale vs Zero-based)
  const [autoScale, setAutoScale] = useState<boolean>(true);

  const toggleVisibility = (key: string) => {
    const newSet = new Set(visibleKeys);
    if (newSet.has(key)) {
      newSet.delete(key);
    } else {
      newSet.add(key);
    }
    setVisibleKeys(newSet);
  };

  const formatXAxis = (tick: string) => {
    if (!tick) return '';
    const date = new Date(tick);
    return `${date.getHours()}:00`;
  };

  // Custom Tick Formatter for Y-Axis to handle large numbers (e.g. 5000 -> 5k)
  const formatYAxis = (value: number) => {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
    return value.toString();
  };

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {/* Chart Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-2 p-4 bg-slate-850 rounded-lg border border-slate-800">
        
        {/* Left Side: Model Toggles */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => toggleVisibility('Actual')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all border ${
              visibleKeys.has('Actual') 
                ? 'bg-slate-700 border-slate-600 text-white' 
                : 'bg-transparent border-slate-800 text-slate-500 opacity-50'
            }`}
          >
            {visibleKeys.has('Actual') ? <Eye size={12} /> : <EyeOff size={12} />}
            ACTUAL
          </button>
          {allKeys.map(key => (
            <button
              key={key}
              onClick={() => toggleVisibility(key)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all border ${
                visibleKeys.has(key)
                  ? 'bg-slate-800 border-slate-700 text-slate-200 shadow-sm'
                  : 'bg-transparent border-slate-800 text-slate-600 opacity-60'
              }`}
            >
              <span 
                className="w-2 h-2 rounded-full" 
                style={{ backgroundColor: MODEL_COLORS[key] || '#ccc' }}
              />
              {key}
            </button>
          ))}
        </div>

        {/* Right Side: Zoom/Scale Controls */}
        <div className="flex items-center gap-2 border-l border-slate-700 pl-4">
          <span className="text-xs font-semibold uppercase text-slate-500 mr-1">Y-Axis:</span>
          <button
            onClick={() => setAutoScale(!autoScale)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all border ${
              autoScale
                ? 'bg-indigo-600/20 border-indigo-500/50 text-indigo-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
            title={autoScale ? "Switch to Zero-based Scale" : "Switch to Auto-Scale (Zoom)"}
          >
            {autoScale ? <ZoomIn size={14} /> : <Maximize size={14} />}
            {autoScale ? 'FOCUSED' : 'FULL RANGE'}
          </button>
        </div>
      </div>

      {/* The Chart */}
      <div className="h-[500px] w-full bg-slate-900 rounded-lg border border-slate-800 p-4 shadow-inner">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={formatXAxis} 
              stroke="#475569" 
              tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              minTickGap={30}
            />
            
            <YAxis 
              stroke="#475569" 
              tickFormatter={formatYAxis}
              tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              /* 
                 CRITICAL FIX: using 'dataMin' and 'dataMax' explicitly forces Recharts 
                 to crop the empty space below the lines, creating a true "Zoom" effect.
              */
              domain={autoScale ? ['dataMin', 'dataMax'] : [0, 'auto']}
              label={{ 
                value: target === TargetVariable.PRICE ? 'EUR/MWh' : 'MW', 
                angle: -90, 
                position: 'insideLeft',
                style: { fill: '#94a3b8', fontSize: 12, fontWeight: 500 } 
              }}
              width={60}
            />
            
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f1f5f9', fontSize: '12px' }}
              itemStyle={{ fontFamily: 'JetBrains Mono', padding: 0 }}
              labelFormatter={(label) => new Date(label).toLocaleString()}
              formatter={(value: number) => [value.toFixed(2), '']}
            />
            
            <Legend 
              wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} 
              iconType="circle"
            />

            {/* Brush Component for Zooming/Panning along X-Axis */}
            <Brush 
              dataKey="timestamp" 
              height={30} 
              stroke="#475569" 
              fill="#1e293b" 
              tickFormatter={() => ''}
              travellerWidth={10}
            />

            {/* Actual Value Line (Prominent) */}
            {visibleKeys.has('Actual') && (
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual Value"
                stroke="#ffffff"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 6, fill: '#fff' }}
                isAnimationActive={false} // Disable animation for smoother zooming
              />
            )}

            {/* Model Lines - Reduced stroke width for separation */}
            {allKeys.map(key => visibleKeys.has(key) && (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={key}
                stroke={MODEL_COLORS[key] || '#8884d8'}
                strokeWidth={1.5} 
                strokeOpacity={0.9}
                strokeDasharray={key.toLowerCase().includes('baseline') ? "4 4" : "0"}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};