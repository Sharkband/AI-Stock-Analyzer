import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Brain, Database, AlertCircle, CheckCircle, Clock, Trash2, Play, BarChart3 } from 'lucide-react';



const API_BASE_URL = 'http://localhost:8000';

// Prediction Panel Component
  const PredictionPanel = ({predictForm,setPredictForm,loading,setLoading,makePrediction, predictions}) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <TrendingUp className="mr-2" />
        Make Predictions
      </h2>
      
      <form onSubmit={async (e) => {
        e.preventDefault();
        try {
          await makePrediction(predictForm);
        } catch (err) {
          console.error('Prediction failed:', err);
        }
      }}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Stock Symbol</label>
            <input
              type="text"
              value={predictForm.symbol}
              onChange={(e) =>  setPredictForm({...predictForm, symbol: e.target.value.toUpperCase()})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., AAPL"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Type</label>
            <select
              value={predictForm.model_type}
              onChange={(e) => setPredictForm({...predictForm, model_type: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="cnn_lstm">CNN-LSTM</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Days Ahead</label>
            <input
              type="number"
              value={predictForm.days_ahead}
              onChange={(e) => setPredictForm({...predictForm, days_ahead: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1"
              max="30"
            />
          </div>
        </div>
        
        <button
          type="submit"
          disabled={loading || !predictForm.symbol}
          className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400 transition-colors"
        >
          {loading ? 'Predicting...' : 'Make Prediction'}
        </button>
      </form>

      {/* Recent Predictions */}
      {predictions.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Recent Predictions</h3>
          <div className="space-y-3">
            {predictions.slice(0, 5).map((pred, idx) => (
              <div key={idx} className="bg-gray-50 p-4 rounded-md">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold text-lg">{pred.symbol}</h4>
                    <p className="text-sm text-gray-600">Model: {pred.model_type}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Current: ${pred.current_price.toFixed(2)}</p>
                    <p className="text-lg font-bold">Predicted: ${pred.predicted_price.toFixed(2)}</p>
                    <p className={`text-sm font-semibold ${pred.price_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {pred.price_change >= 0 ? '+' : ''}${pred.price_change.toFixed(2)} ({pred.percent_change.toFixed(2)}%)
                    </p>
                  </div>
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Confidence: {(pred.confidence * 100).toFixed(1)}% | {new Date(pred.timestamp).toLocaleString()}
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Data Last Updated: {(pred.last_updated)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );


  // Training Panel Component
  const TrainingPanel = ({trainForm,setTrainForm,trainModel,loading,trainingStatus,setLoading}) => (
    
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <Brain className="mr-2" />
        Train New Model
      </h2>
      
      <div onSubmit={async (e) => {
        e.preventDefault();
        try {
          await trainModel(trainForm);
          setTrainForm({ ...trainForm, symbol: '' }); // Clear symbol but keep other settings
        } catch (err) {
          console.error('Training failed:', err);
        }
      }}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Stock Symbol</label>
            <input
              type="text"
              value={trainForm.symbol}
              onChange={(e) => setTrainForm({...trainForm, symbol: e.target.value.toUpperCase()})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., AAPL"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Type</label>
            <select
              value={trainForm.model_type}
              onChange={(e) => setTrainForm({...trainForm, model_type: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="cnn_lstm">CNN-LSTM</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Sequence Length</label>
            <input
              type="number"
              value={trainForm.sequence_length}
              onChange={(e) => setTrainForm({...trainForm, sequence_length: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="10"
              max="200"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Epochs</label>
            <input
              type="number"
              value={trainForm.epochs}
              onChange={(e) => setTrainForm({...trainForm, epochs: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="10"
              max="1000"
            />
          </div>
        </div>
        
        <button
          onClick={async (e) => {
            e.preventDefault();
            try {
              await trainModel(trainForm);
              setTrainForm({ ...trainForm, symbol: '' });
            } catch (err) {
              console.error('Training failed:', err);
            }
          }}
          disabled={loading || !trainForm.symbol}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          {loading ? 'Starting Training...' : 'Start Training'}
        </button>
      </div>

      {/* Training Status */}
      {Object.keys(trainingStatus).length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Training Status</h3>
          {Object.entries(trainingStatus).map(([key, status]) => (
            <div key={key} className="bg-gray-50 p-3 rounded-md mb-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">{key.replace('_', ' - ')}</span>
                <div className="flex items-center">
                  {status.status === 'training' && <Clock className="text-yellow-500 mr-1" size={16} />}
                  {status.status === 'completed' && <CheckCircle className="text-green-500 mr-1" size={16} />}
                  {status.status === 'failed' && <AlertCircle className="text-red-500 mr-1" size={16} />}
                  <span className={`text-sm ${
                    status.status === 'training' ? 'text-yellow-600' :
                    status.status === 'completed' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {status.status}
                  </span>
                </div>
              </div>
              {status.train_rmse && (
                <div className="text-sm text-gray-600 mt-1">
                  Train RMSE: {status.train_rmse.toFixed(4)}, Val RMSE: {status.val_rmse.toFixed(4)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );


  



const StockAnalyzerApp = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stockData, setStockData] = useState([]);
  const [marketOverview, setMarketOverview] = useState({});
  const [trainedModels, setTrainedModels] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [selectedPeriod, setSelectedPeriod] = useState('1y');
  const [trainingStatus, setTrainingStatus] = useState({});
  

  // Form states
  const [trainForm, setTrainForm] = useState({
    symbol: '',
    model_type: 'lstm',
    sequence_length: 60,
    epochs: 100,
    batch_size: 32
  });

  const [predictForm, setPredictForm] = useState({
    symbol: '',
    model_type: 'lstm',
    days_ahead: 1
  });

  const [stockSymbols, setStockSymbols] = useState([
    { value: "AAPL", label: "Apple (AAPL)" },
    { value: "GOOGL", label: "Google (GOOGL)" },
    { value: "MSFT", label: "Microsoft (MSFT)" },
    { value: "AMZN", label: "Amazon (AMZN)" },
    { value: "TSLA", label: "Tesla (TSLA)" },
    { value: "VFV.TO", label: "ETF (VFV.TO)" }
  ]);


  // Fetch functions
  const fetchStockData = async (symbol, period = '1y') => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/stock-data/${symbol}?period=${period}`);
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Failed to fetch stock data');
      
      setStockData(data.data);
      setError('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchMarketOverview = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/market-overview`);
      const data = await response.json();
      setMarketOverview(data);
    } catch (err) {
      console.error('Failed to fetch market overview:', err);
    }
  };

  const fetchTrainedModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      const data = await response.json();
      setTrainedModels(data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const trainModel = async (formData) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Failed to start training');
      
      // Start polling training status
      pollTrainingStatus(formData.symbol, formData.model_type);
      
      setError('');
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const makePrediction = async (formData) => {
    try {
      
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Failed to make prediction');
      
      setPredictions(prev => [data, ...prev.slice(0, 9)]); // Keep last 10 predictions
      setError('');
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const pollTrainingStatus = async (symbol, modelType) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/training-status/${symbol}/${modelType}`);
        const data = await response.json();
        
        setTrainingStatus(prev => ({
          ...prev,
          [`${symbol}_${modelType}`]: data
        }));

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollInterval);
          fetchTrainedModels(); // Refresh models list
        }
      } catch (err) {
        console.error('Failed to poll training status:', err);
        clearInterval(pollInterval);
      }
    }, 2000);

    return pollInterval;
  };

  

  const deleteModel = async (symbol, modelType) => {
    try {
      const response = await fetch(`${API_BASE_URL}/models/${symbol}/${modelType}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) throw new Error('Failed to delete model');
      
      fetchTrainedModels(); // Refresh models list
    } catch (err) {
      setError(err.message);
    }
  };

  // Load initial data
  useEffect(() => {
    fetchStockData(selectedSymbol, selectedPeriod);
    fetchMarketOverview();
    fetchTrainedModels();
  }, []);

  // Market Overview Component
  const MarketOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {Object.entries(marketOverview).map(([name, data]) => (
        <div key={name} className="bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-800">{name}</h3>
              <p className="text-2xl font-bold text-gray-900">
                {data.current_price ? data.current_price.toLocaleString() : 'N/A'}
              </p>
            </div>
            <div className="text-right">
              {data.change && (
                <>
                  <div className={`flex items-center ${data.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {data.change >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                    <span className="ml-1 font-semibold">{data.change.toFixed(2)}</span>
                  </div>
                  <p className={`text-sm ${data.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {data.change_percent.toFixed(2)}%
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  // Stock Chart Component
const StockChart = () => {
  
  const [newSymbol, setNewSymbol] = useState("");

  const formattedChartData = useMemo(() => {
    if (!stockData || stockData.length === 0) return [];
    return stockData.map(item => ({
      ...item,
      date: new Date(item.date).toLocaleDateString()
    }));
  }, [stockData]);
  
  const addSymbol = () => {
    if (newSymbol.trim()) {
      const symbolExists = stockSymbols.some(stock => 
        stock.value.toLowerCase() === newSymbol.trim().toLowerCase()
      );
      
      if (!symbolExists) {
        const symbol = newSymbol.trim().toUpperCase();
        const newStock = {
          value: symbol,
          label: `${symbol} (${symbol})`
        };
        setStockSymbols([...stockSymbols, newStock]);
        setNewSymbol("");
      } else {
        alert("This symbol already exists in the list!");
      }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800 mb-2 md:mb-0">Stock Price Chart</h2>
        <div className="flex flex-col md:flex-row gap-2">
          <select 
            value={selectedSymbol} 
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {stockSymbols.map(stock => (
              <option key={stock.value} value={stock.value}>
                {stock.label}
              </option>
            ))}
          </select>
          <select 
            value={selectedPeriod} 
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="1mo">1 Month</option>
            <option value="3mo">3 Months</option>
            <option value="6mo">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="2y">2 Years</option>
          </select>
          <button 
            onClick={() => fetchStockData(selectedSymbol, selectedPeriod)}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Update Chart'}
          </button>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-2 mb-4">
        <input
          type="text"
          value={newSymbol}
          onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
          placeholder="e.g., NVDA"
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addSymbol();
            }
          }}
        />
        <button 
          onClick={addSymbol}
          className="px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
        >
          Add Symbol
        </button>
      </div>
      
      {formattedChartData.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={formattedChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="close" stroke="#2563eb" strokeWidth={2} name="Close Price" />
            <Line type="monotone" dataKey="volume" stroke="#dc2626" strokeWidth={1} name="Volume" yAxisId="right" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

 
  // Models Management Component
  const ModelsManagement = () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <Database className="mr-2" />
        Trained Models
      </h2>
      
      <div className="space-y-3">
        {trainedModels.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No trained models found. Train a model first.</p>
        ) : (
          trainedModels.map((model, idx) => (
            <div key={idx} className="border border-gray-200 rounded-md p-4 hover:bg-gray-50 transition-colors">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-lg">{model.symbol} - {model.model_type}</h3>
                  <p className="text-sm text-gray-600">
                    Trained: {new Date(model.trained_at).toLocaleString()}
                  </p>
                  <p className="text-sm text-gray-600">
                    Train RMSE: {model.train_rmse.toFixed(4)} | Val RMSE: {model.val_rmse.toFixed(4)}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    model.status === 'loaded' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                  }`}>
                    {model.status}
                  </span>
                  <button
                    onClick={() => deleteModel(model.symbol, model.model_type)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-md transition-colors"
                    title="Delete Model"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
      
      <button
        onClick={fetchTrainedModels}
        className="mt-4 w-full bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700 transition-colors"
      >
        Refresh Models List
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">AI Stock Analyzer</h1>
            </div>
            <nav className="flex space-x-4">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: Activity },
                { id: 'train', label: 'Train Models', icon: Brain },
                { id: 'predict', label: 'Predictions', icon: TrendingUp },
                { id: 'models', label: 'Models', icon: Database }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === id 
                      ? 'bg-blue-100 text-blue-700' 
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <Icon size={16} className="mr-1" />
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <p className="text-red-800">{error}</p>
              <button 
                onClick={() => setError('')}
                className="ml-auto text-red-400 hover:text-red-600"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <MarketOverview />
            <StockChart />
          </div>
        )}

        {activeTab === 'train' && <TrainingPanel 
          trainForm = {trainForm}
          setTrainForm = {setTrainForm}
          trainModel = {trainModel}
          loading = {loading}
          setLoading={setLoading}
          trainingStatus = {trainingStatus}
          />
        }
        {activeTab === 'predict' && <PredictionPanel 
          predictForm={predictForm}
          setPredictForm={setPredictForm}
          loading = {loading}
          setLoading = {setLoading}
          makePrediction = {makePrediction}
          predictions = {predictions}
          />
        } 
        {activeTab === 'models' && <ModelsManagement />}
      </main>
    </div>
  );
};

export default StockAnalyzerApp;