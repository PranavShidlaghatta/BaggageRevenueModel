"use client"
import { motion } from 'framer-motion'
import Image from 'next/image'
import Link from 'next/link'

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
              <Link href="/predictions" className="border-b-2 border-indigo-400 py-4 px-1 text-sm font-medium text-indigo-400">
                SARIMAX
              </Link>
              <Link href="/predictions/xgboost" className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                XGBoost
              </Link>
              <Link href="/predictions/comparison" className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                Comparison
              </Link>
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
              <div className="relative w-full h-64 bg-white/[0.02] rounded-lg border border-white/[0.05] overflow-hidden">
                <Image
                  src="/results/plots/Southwest_no_exog_forecast.png"
                  alt="SARIMAX Forecast without Exogenous Variables"
                  fill
                  className="object-contain"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-white">2,792.68</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-white">1,937.63</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAPE</p>
                  <p className="text-2xl font-bold text-white">10.97%</p>
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
              <div className="relative w-full h-64 bg-white/[0.02] rounded-lg border border-white/[0.05] overflow-hidden">
                <Image
                  src="/results/plots/Southwest_jetfuel_cost_forecast.png"
                  alt="SARIMAX Forecast with Jet Fuel Cost Exogenous Variable"
                  fill
                  className="object-contain"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-white">1,839.49</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-white">1,449.18</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                  <p className="text-sm text-white/60 mb-1">MAPE</p>
                  <p className="text-2xl font-bold text-white">8.26%</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

      </div>
    </div>
  )
}
