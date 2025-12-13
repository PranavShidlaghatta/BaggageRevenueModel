"use client"
import Link from 'next/link'
import { HeroGeometric } from '@/components/ui/shape-landing-hero'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export default function Home() {
  const fadeUpVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: 1,
        delay: 0.5 + i * 0.2,
        ease: [0.25, 0.4, 0.25, 1],
      },
    }),
  };

  return (
    <div className="min-h-screen bg-[#030303]">
      {/* Hero Section with Geometric Shapes */}
      <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#030303]">
        <HeroGeometric
          title1="Baggage Revenue"
          title2="Prediction Dashboard"
        />
      </div>

      {/* Features Section */}
      <div className="relative bg-[#030303] py-20">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.02] via-transparent to-rose-500/[0.02] blur-3xl" />
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Model Capabilities</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Our dashboard leverages multiple forecasting models to provide accurate predictions
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* SARIMAX Card */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="bg-white/[0.03] backdrop-blur-sm rounded-xl p-8 hover:bg-white/[0.05] transition-all border border-white/[0.08] hover:border-white/[0.15]"
            >
              <div className="w-12 h-12 bg-indigo-500/20 rounded-lg flex items-center justify-center mb-4 border border-indigo-500/30">
                <svg className="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">SARIMAX</h3>
              <p className="text-white/60">
                Seasonal AutoRegressive Integrated Moving Average with eXogenous variables. 
                Captures seasonal patterns and external factors affecting baggage revenue.
              </p>
            </motion.div>

            {/* XGBoost Card */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="bg-white/[0.03] backdrop-blur-sm rounded-xl p-8 hover:bg-white/[0.05] transition-all border border-white/[0.08] hover:border-white/[0.15]"
            >
              <div className="w-12 h-12 bg-rose-500/20 rounded-lg flex items-center justify-center mb-4 border border-rose-500/30">
                <svg className="w-6 h-6 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">XGBoost Regression</h3>
              <p className="text-white/60">
                Gradient boosting model that handles complex non-linear relationships. 
                Includes SHAP values for model interpretability.
              </p>
            </motion.div>

            {/* Time Series Card */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="bg-white/[0.03] backdrop-blur-sm rounded-xl p-8 hover:bg-white/[0.05] transition-all border border-white/[0.08] hover:border-white/[0.15]"
            >
              <div className="w-12 h-12 bg-violet-500/20 rounded-lg flex items-center justify-center mb-4 border border-violet-500/30">
                <svg className="w-6 h-6 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Time Series Forecasting</h3>
              <p className="text-white/60">
                Multiple baseline models including naive seasonal approaches and ARIMA variants 
                for comprehensive forecasting analysis.
              </p>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Data Sources Section */}
      <div className="relative bg-[#030303] py-20">
        <div className="absolute inset-0 bg-gradient-to-br from-rose-500/[0.02] via-transparent to-indigo-500/[0.02] blur-3xl" />
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="bg-white/[0.03] backdrop-blur-sm rounded-xl p-8 border border-white/[0.08]"
          >
            <h2 className="text-2xl font-bold text-white mb-6">Data Sources</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-white mb-2">Bureau of Transportation Statistics</h3>
                <p className="text-white/60 text-sm">
                  Quarterly baggage revenue data from the last 10 years
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Economic Indicators</h3>
                <p className="text-white/60 text-sm">
                  Jet fuel prices, unemployment rates, and GDP per capita data
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-[#030303] border-t border-white/[0.08] py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-white/40">
            Baggage Revenue Prediction Dashboard © 2025
          </p>
        </div>
      </footer>
    </div>
  )
}
