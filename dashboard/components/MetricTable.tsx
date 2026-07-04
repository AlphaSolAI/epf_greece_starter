import React from 'react';
import { ModelMetrics } from '../types';
import { MODEL_COLORS } from '../constants';

interface MetricTableProps {
  metrics: ModelMetrics[];
}

export const MetricTable: React.FC<MetricTableProps> = ({ metrics }) => {
  return (
    <div className="w-full overflow-hidden rounded-lg border border-slate-700 bg-slate-850 shadow-xl">
      <div className="border-b border-slate-700 bg-slate-900 px-6 py-4">
        <h3 className="text-lg font-semibold text-slate-100 font-mono tracking-tight">
          Performance Metrics (Sorted by MAE)
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-400">
          <thead className="bg-slate-900 text-xs uppercase text-slate-500">
            <tr>
              <th scope="col" className="px-6 py-3 font-medium">Model</th>
              <th scope="col" className="px-6 py-3 font-medium text-right">MAE</th>
              <th scope="col" className="px-6 py-3 font-medium text-right">RMSE</th>
              <th scope="col" className="px-6 py-3 font-medium text-right">MAPE (%)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {metrics.map((metric) => (
              <tr 
                key={metric.modelName} 
                className="hover:bg-slate-800/50 transition-colors"
              >
                <td className="px-6 py-4 font-medium text-slate-200 flex items-center gap-2">
                  <span 
                    className="w-3 h-3 rounded-full shadow-sm"
                    style={{ backgroundColor: MODEL_COLORS[metric.modelName] || '#64748b' }}
                  ></span>
                  {metric.modelName}
                  {metric.isBaseline && (
                    <span className="ml-2 inline-flex items-center rounded-md bg-slate-400/10 px-2 py-1 text-xs font-medium text-slate-400 ring-1 ring-inset ring-slate-400/20">
                      Baseline
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-right font-mono text-cyan-400">
                  {metric.mae.toFixed(4)}
                </td>
                <td className="px-6 py-4 text-right font-mono text-violet-400">
                  {metric.rmse.toFixed(4)}
                </td>
                <td className="px-6 py-4 text-right font-mono text-emerald-400">
                  {metric.mape.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
