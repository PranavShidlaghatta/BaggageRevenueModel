"use client"
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Home } from "lucide-react"
import { ChevronRightIcon } from "@radix-ui/react-icons"

export function Navigation() {
  const pathname = usePathname()
  const isHome = pathname === '/'
  const isPredictions = pathname === '/predictions'

  return (
    <Breadcrumb>
      <BreadcrumbList className="rounded-lg border border-white/[0.08] bg-white/[0.03] backdrop-blur-sm px-3 py-2 shadow-sm">
        <BreadcrumbItem>
          {isHome ? (
            <BreadcrumbPage className="text-white">
              <Home size={16} strokeWidth={2} aria-hidden="true" className="inline mr-1.5" />
              <span>Home</span>
            </BreadcrumbPage>
          ) : (
            <BreadcrumbLink asChild>
              <Link href="/" className="text-white/70 hover:text-white">
                <Home size={16} strokeWidth={2} aria-hidden="true" className="inline mr-1.5" />
                <span>Home</span>
              </Link>
            </BreadcrumbLink>
          )}
        </BreadcrumbItem>
        <BreadcrumbSeparator>
          <ChevronRightIcon width={16} height={16} className="text-white/40" />
        </BreadcrumbSeparator>
        <BreadcrumbItem>
          {isPredictions ? (
            <BreadcrumbPage className="text-white">Predictions</BreadcrumbPage>
          ) : (
            <BreadcrumbLink asChild>
              <Link href="/predictions" className="text-white/70 hover:text-white">
                Predictions
              </Link>
            </BreadcrumbLink>
          )}
        </BreadcrumbItem>
      </BreadcrumbList>
    </Breadcrumb>
  )
}
