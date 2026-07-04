import React from 'react';
import { TargetVariable, PredictionTechnique } from '../types';
import { Activity, TrendingUp, Cpu, Upload } from 'lucide-react';

interface SidebarProps {
  currentTarget: TargetVariable;
  currentTechnique: PredictionTechnique;
  onSelectTarget: (t: TargetVariable) => void;
  onSelectTechnique: (t: PredictionTechnique) => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTarget,
  currentTechnique,
  onSelectTarget,
  onSelectTechnique,
  onFileUpload
}) => {
  return (
    <aside className="w-72 h-screen bg-slate-950 border-r border-slate-800 flex flex-col fixed left-0 top-0 overflow-y-auto">
      {/* Header */}
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-indigo-500/10 rounded-lg">
            <Cpu className="text-indigo-400" size={24} />
          </div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">EPF Thesis</h1>
        </div>
        <p className="text-xs text-slate-500 leading-relaxed">
          Energy Price & Load Forecasting<br />
          <span className="opacity-70">Comparative Analysis</span>
        </p>
      </div>

      {/* Target Selector */}
      <div className="p-6 pb-2">
        <h3 className="text-xs font-semibold uppercase text-slate-500 mb-4 tracking-wider">Target Variable</h3>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onSelectTarget(TargetVariable.PRICE)}
            className={`flex flex-col items-center justify-center p-3 rounded-lg border transition-all ${
              currentTarget === TargetVariable.PRICE
                ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600'
            }`}
          >
            <TrendingUp size={20} className="mb-2" />
            <span className="text-xs font-medium">Price</span>
          </button>
          <button
            onClick={() => onSelectTarget(TargetVariable.LOAD)}
            className={`flex flex-col items-center justify-center p-3 rounded-lg border transition-all ${
              currentTarget === TargetVariable.LOAD
                ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600'
            }`}
          >
            <Activity size={20} className="mb-2" />
            <span className="text-xs font-medium">Load</span>
          </button>
        </div>
      </div>

      {/* Technique Selector */}
      <div className="p-6">
        <h3 className="text-xs font-semibold uppercase text-slate-500 mb-4 tracking-wider">Forecasting Technique</h3>
        <div className="space-y-2">
          {Object.values(PredictionTechnique).map((tech) => (
            <button
              key={tech}
              onClick={() => onSelectTechnique(tech)}
              className={`w-full text-left px-4 py-3 rounded-md text-sm transition-all border ${
                currentTechnique === tech
                  ? 'bg-slate-800 border-slate-600 text-white shadow-md'
                  : 'bg-transparent border-transparent text-slate-400 hover:bg-slate-900 hover:text-slate-300'
              }`}
            >
              {tech}
            </button>
          ))}
        </div>
      </div>

      {/* Data Upload Area */}
      <div className="mt-auto p-6 border-t border-slate-800 bg-slate-900/50">
        <label className="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-slate-700 rounded-lg cursor-pointer hover:bg-slate-800 hover:border-slate-500 transition-all group">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <Upload className="w-6 h-6 mb-2 text-slate-500 group-hover:text-indigo-400" />
            <p className="text-xs text-slate-500 group-hover:text-slate-300">Upload JSON Result</p>
          </div>
          <input type="file" className="hidden" accept=".json" onChange={onFileUpload} />
        </label>
      </div>
    </aside>
  );
};