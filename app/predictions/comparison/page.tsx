"use client"
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function ComparisonPage() {
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
            Model Comparison
          </h1>
          <p className="text-lg text-white/60">
            Compare performance metrics across all forecasting models
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
              <Link href="/predictions/xgboost" className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-white/60 hover:text-white/80 hover:border-white/30">
                XGBoost
              </Link>
              <Link href="/predictions/comparison" className="border-b-2 border-amber-400 py-4 px-1 text-sm font-medium text-amber-400">
                Comparison
              </Link>
            </nav>
          </div>
        </motion.div>

        {/* Model Comparison Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">2,792.68</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">1,937.63</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">10.97%</td>
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">1,839.49</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">1,449.18</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">8.26%</td>
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">5,085.10</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">4,809.23</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white/60">27.54%</td>
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
