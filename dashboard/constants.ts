import { ExperimentData, ModelMetrics, PredictionTechnique, TargetVariable, TimeSeriesPoint } from './types';

// Scientific Color Palette for Models
export const MODEL_COLORS: Record<string, string> = {
  'Actual': '#ffffff', // White for ground truth
  'LGBM': '#06b6d4', // Cyan 500
  'XGBoost': '#8b5cf6', // Violet 500
  'Random Forest': '#10b981', // Emerald 500
  'MLP': '#f59e0b', // Amber 500
  'SVR': '#ec4899', // Pink 500
  // Baselines (Muted colors)
  'Naive Baseline': '#64748b',
  'Seasonal Naive': '#94a3b8',
  'Moving Average': '#475569',
  'Exponential Smoothing': '#334155'
};

// Helper to generate a mock date string
const addHours = (date: Date, h: number) => {
  const d = new Date(date);
  d.setTime(d.getTime() + (h * 60 * 60 * 1000));
  return d.toISOString();
}

/**
 * GENERATES SCIENTIFIC MOCK DATA
 * This simulates the structure your Python script should output to JSON.
 */
export const generateMockData = (target: TargetVariable, technique: PredictionTechnique): ExperimentData => {
  const dataPoints: TimeSeriesPoint[] = [];
  const startDate = new Date('2023-01-01T00:00:00');
  
  // Generate 48 hours of data
  for (let i = 0; i < 48; i++) {
    const baseValue = target === TargetVariable.LOAD ? 6000 : 150; // Load vs Price scale
    const noise = () => (Math.random() - 0.5) * (baseValue * 0.1);
    const timeTrend = Math.sin(i / 8) * (baseValue * 0.3); // Sinusoidal daily pattern

    const actual = baseValue + timeTrend + noise();

    dataPoints.push({
      timestamp: addHours(startDate, i),
      actual: parseFloat(actual.toFixed(2)),
      'LGBM': parseFloat((actual + (Math.random() - 0.5) * 20).toFixed(2)),
      'XGBoost': parseFloat((actual + (Math.random() - 0.5) * 25).toFixed(2)),
      'SVR': parseFloat((actual + (Math.random() - 0.5) * 40).toFixed(2)),
      'Random Forest': parseFloat((actual + (Math.random() - 0.5) * 35).toFixed(2)),
      'MLP': parseFloat((actual + (Math.random() - 0.5) * 50).toFixed(2)),
      'Naive Baseline': parseFloat((actual + (Math.random() - 0.5) * 80).toFixed(2)),
      'Seasonal Naive': parseFloat((actual + (Math.random() - 0.5) * 90).toFixed(2)),
    });
  }

  // Calculate Mock Metrics based on the generated data logic
  const metrics: ModelMetrics[] = [
    { modelName: 'LGBM', mae: 12.45, rmse: 18.2, mape: 4.2, isBaseline: false },
    { modelName: 'XGBoost', mae: 14.10, rmse: 19.5, mape: 4.8, isBaseline: false },
    { modelName: 'Random Forest', mae: 15.30, rmse: 21.0, mape: 5.1, isBaseline: false },
    { modelName: 'SVR', mae: 18.90, rmse: 25.4, mape: 6.3, isBaseline: false },
    { modelName: 'MLP', mae: 22.10, rmse: 28.9, mape: 7.5, isBaseline: false },
    { modelName: 'Naive Baseline', mae: 45.00, rmse: 55.2, mape: 15.2, isBaseline: true },
    { modelName: 'Seasonal Naive', mae: 42.10, rmse: 51.5, mape: 14.8, isBaseline: true },
  ];

  // Sort by MAE as requested
  metrics.sort((a, b) => a.mae - b.mae);

  return {
    target,
    technique,
    data: dataPoints,
    metrics
  };
};
