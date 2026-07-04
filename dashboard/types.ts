// Enum for the main prediction targets
export enum TargetVariable {
  LOAD = 'Load',
  PRICE = 'Price'
}

// Enum for the prediction techniques
export enum PredictionTechnique {
  RECURSIVE_CLOSED = 'Recursive Closed Loop (Real Data)',
  RECURSIVE_OPEN = 'Recursive Open Loop (Generated)',
  MIMO = 'MIMO (Direct)',
  DIRECT_MULTISTEP = 'Direct Multistep'
}

// Represents a single point in time for the chart
export interface TimeSeriesPoint {
  timestamp: string;
  actual: number; // The ground truth
  [modelName: string]: number | string; // Dynamic keys for model predictions
}

// Metrics for the table
export interface ModelMetrics {
  modelName: string;
  mae: number; // Mean Absolute Error
  rmse: number; // Root Mean Square Error
  mape: number; // Mean Absolute Percentage Error
  isBaseline: boolean;
}

// The structure of the uploaded JSON file
export interface ExperimentData {
  target: TargetVariable;
  technique: PredictionTechnique;
  data: TimeSeriesPoint[];
  metrics: ModelMetrics[];
}
