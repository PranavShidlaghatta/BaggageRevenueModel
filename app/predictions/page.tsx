"use client"
import { motion } from 'framer-motion'

export default function PredictionsPage() {
  return (
    <div className="min-h-screen bg-[#030303] py-12 relative overflow-hidden">
      {/* Background gradient effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.03] via-transparent to-rose-500/[0.03] blur-3xl" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Model Predictions
          </h1>
          <p className="text-lg text-white/60">
            View and compare predictions from different forecasting models
          </p>
        </motion.div>

        {/* Model Selection Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="bg-white/[0.03] backdrop-blur-sm rounded-lg mb-8 border border-white/[0.08]"
        >
          <div className="border-b border-white/[0.08]">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              <button className="border-b-2 border-indigo-400 py-4 px-1 text-sm font-medium text-indigo-400">
                SARIMAX
              </button>
              <button className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                XGBoost
              </button>
              <button className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                Baseline Models
              </button>
            </nav>
          </div>
        </motion.div>

        {/* Prediction Cards Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* SARIMAX - No Exogenous Variables */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-white/[0.03] backdrop-blur-sm rounded-lg p-6 border border-white/[0.08] hover:bg-white/[0.05] hover:border-white/[0.15] transition-all"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">SARIMAX (No Exogenous)</h2>
              <span className="px-3 py-1 bg-indigo-500/20 text-indigo-400 text-xs font-medium rounded-full border border-indigo-500/30">
                Active
              </span>
            </div>
            <div className="space-y-4">
              <div className="h-64 bg-white/[0.02] rounded-lg flex items-center justify-center border border-white/[0.05]">
                <p className="text-white/40">Forecast Chart Placeholder</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-white">-</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-white">-</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* SARIMAX - With Exogenous Variables */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="bg-white/[0.03] backdrop-blur-sm rounded-lg p-6 border border-white/[0.08] hover:bg-white/[0.05] hover:border-white/[0.15] transition-all"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">SARIMAX (With Exogenous)</h2>
              <span className="px-3 py-1 bg-rose-500/20 text-rose-400 text-xs font-medium rounded-full border border-rose-500/30">
                Active
              </span>
            </div>
            <div className="space-y-4">
              <div className="h-64 bg-white/[0.02] rounded-lg flex items-center justify-center border border-white/[0.05]">
                <p className="text-white/40">Forecast Chart Placeholder</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-white">-</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-white">-</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* XGBoost Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-white/[0.03] backdrop-blur-sm rounded-lg p-6 mb-8 border border-white/[0.08] hover:bg-white/[0.05] hover:border-white/[0.15] transition-all"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-white">XGBoost Regression</h2>
            <span className="px-3 py-1 bg-violet-500/20 text-violet-400 text-xs font-medium rounded-full border border-violet-500/30">
              Active
            </span>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="h-64 bg-white/[0.02] rounded-lg flex items-center justify-center border border-white/[0.05]">
              <p className="text-white/40">Prediction Chart Placeholder</p>
            </div>
            <div className="space-y-4">
              <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                <p className="text-sm text-white/60 mb-2">Model Performance</p>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">R² Score</span>
                    <span className="font-semibold text-white">-</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">RMSE</span>
                    <span className="font-semibold text-white">-</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">MAE</span>
                    <span className="font-semibold text-white">-</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                <p className="text-sm text-white/60 mb-2">SHAP Values</p>
                <div className="h-32 bg-white/[0.02] rounded flex items-center justify-center border border-white/[0.05]">
                  <p className="text-white/40 text-sm">SHAP Plot Placeholder</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Model Comparison Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="bg-white/[0.03] backdrop-blur-sm rounded-lg p-6 border border-white/[0.08]"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Model Comparison</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-white/[0.08]">
              <thead className="bg-white/[0.02]">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white/60 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white/60 uppercase tracking-wider">
                    RMSE
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white/60 uppercase tracking-wider">
                    MAE
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white/60 uppercase tracking-wider">
                    MAPE
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white/60 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white/[0.01] divide-y divide-white/[0.08]">
                <tr className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                    SARIMAX (No Exog)
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                      Ready
                    </span>
                  </td>
                </tr>
                <tr className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                    SARIMAX (With Exog)
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">
                      Ready
                    </span>
                  </td>
                </tr>
                <tr className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                    XGBoost Regression
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">-</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-violet-500/20 text-violet-400 border border-violet-500/30">
                      Ready
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
