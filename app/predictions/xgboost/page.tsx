"use client"
import { motion } from 'framer-motion'
import Image from 'next/image'
import Link from 'next/link'

export default function XGBoostPage() {
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
            XGBoost Regression
          </h1>
          <p className="text-lg text-white/60">
            Gradient boosting model with SHAP interpretability
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
              <Link href="/predictions" className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                SARIMAX
              </Link>
              <Link href="/predictions/xgboost" className="border-b-2 border-violet-400 py-4 px-1 text-sm font-medium text-violet-400">
                XGBoost
              </Link>
              <Link href="/predictions/comparison" className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                Comparison
              </Link>
            </nav>
          </div>
        </motion.div>

        {/* XGBoost Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="bg-white/[0.03] backdrop-blur-sm rounded-lg p-6 mb-8 border border-white/[0.08] hover:bg-white/[0.05] hover:border-white/[0.15] transition-all"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-white">XGBoost Regression</h2>
            <span className="px-3 py-1 bg-violet-500/20 text-violet-400 text-xs font-medium rounded-full border border-violet-500/30">
              Active
            </span>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="relative w-full h-64 bg-white/[0.02] rounded-lg border border-white/[0.05] overflow-hidden">
              <Image
                src="/results/plots/Southwest_all_exog_forecast.png"
                alt="XGBoost Forecast with All Exogenous Variables"
                fill
                className="object-contain"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </div>
            <div className="space-y-4">
              <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                <p className="text-sm text-white/60 mb-2">Model Performance</p>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">RMSE</span>
                    <span className="font-semibold text-white">5,085.10</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">MAE</span>
                    <span className="font-semibold text-white">4,809.23</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">MAPE</span>
                    <span className="font-semibold text-white">27.54%</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/[0.02] rounded-lg p-4 border border-white/[0.05]">
                <p className="text-sm text-white/60 mb-2">Exogenous Variables P-values</p>
                <div className="relative w-full h-48 bg-white/[0.02] rounded border border-white/[0.05] overflow-hidden">
                  <Image
                    src="/results/plots/Southwest_exogenous_pvalues.png"
                    alt="Exogenous Variables Statistical Significance"
                    fill
                    className="object-contain"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
