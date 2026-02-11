// KTX2 loader with Basis Universal transcoder

export interface KTX2Texture {
  width: number
  height: number
  data: Uint8Array
  format: 'rgba8unorm'
}

let basisPromise: Promise<any> | null = null

function loadBasisTranscoder(transcoderPath: string): Promise<any> {
  if (basisPromise) return basisPromise
  basisPromise = new Promise<any>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = transcoderPath + 'basis_transcoder.js'
    script.onload = () => {
      const factory = (globalThis as any).BASIS
      factory({ locateFile: (f: string) => transcoderPath + f })
        .then((module: any) => {
          module.initializeBasis()
          resolve(module)
        })
        .catch(reject)
    }
    script.onerror = () => reject(new Error('Failed to load Basis transcoder'))
    document.head.appendChild(script)
  })
  return basisPromise
}

export async function loadKTX2(url: string, transcoderPath = '/basis-1.50/'): Promise<KTX2Texture> {
  const basis = await loadBasisTranscoder(transcoderPath)
  const buf = await (await fetch(url)).arrayBuffer()
  const data = new Uint8Array(buf)

  const ktx2File = new basis.KTX2File(data)
  const width = ktx2File.getWidth()
  const height = ktx2File.getHeight()

  if (!ktx2File.startTranscoding()) {
    ktx2File.close()
    ktx2File.delete()
    throw new Error('KTX2 transcoding failed to start')
  }

  // cTFRGBA32 = 13 — transcode to RGBA8 for max compatibility
  const cTFRGBA32 = 13
  const imageSize = ktx2File.getImageTranscodedSizeInBytes(0, 0, 0, cTFRGBA32)
  const rgba = new Uint8Array(imageSize)

  if (!ktx2File.transcodeImage(rgba, 0, 0, 0, cTFRGBA32, 0, -1, -1)) {
    ktx2File.close()
    ktx2File.delete()
    throw new Error('KTX2 image transcoding failed')
  }

  ktx2File.close()
  ktx2File.delete()

  return { width, height, data: rgba, format: 'rgba8unorm' }
}
