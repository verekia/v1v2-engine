import { useEffect, useRef } from 'react'

import Head from 'next/head'

export default function GamePage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    import('../src/main.ts').then(m => m.startDemo(canvas))
  }, [])

  return (
    <>
      <Head>
        <title>Mana Engine</title>
        <style>{`
          * { margin: 0; padding: 0; }
          html, body { overflow: hidden; width: 100%; height: 100%; }
        `}</style>
      </Head>
      <canvas ref={canvasRef} style={{ width: '100vw', height: '100vh', display: 'block' }} />
    </>
  )
}
