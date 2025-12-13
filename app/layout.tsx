import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Link from 'next/link'
import { Navigation } from '@/components/navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Baggage Revenue Prediction Dashboard',
  description: 'Forecasting quarterly baggage revenue for Southwest Airlines',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-[#030303] border-b border-white/[0.08] backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16 items-center">
              <div className="flex items-center">
                <Link href="/" className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-rose-400 hover:from-indigo-300 hover:to-rose-300 transition-all">
                  Baggage Revenue Dashboard
                </Link>
              </div>
              <Navigation />
            </div>
          </div>
        </nav>
        <main className="min-h-screen bg-[#030303]">
          {children}
        </main>
      </body>
    </html>
  )
}
