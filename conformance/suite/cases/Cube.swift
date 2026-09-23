import Metal
import Foundation

// A cube texture bound to a compute kernel.
//
// A cube is six square faces a sampler addresses by direction. A device that
// stages it has three ways to be wrong that a 2D case cannot see: the faces in
// the wrong layers, one face standing in for all six, and a filter that stops
// at a face's edge instead of reaching into its neighbour. Each face here is a
// different uniform colour, so the first two read as a neighbour's colour, and
// the edge lane reads as one face's colour rather than the two faces' blend.

/// The colour of face `face` (Metal's order: +X -X +Y -Y +Z -Z).
private func cubeFaceColour(_ face: Int) -> (UInt8, UInt8, UInt8, UInt8) {
    (UInt8(0x10 + face * 0x20), UInt8(0xF0 - face * 0x20), UInt8(0x05 + face * 0x11), 0xFF)
}

private func makeCube(_ side: Int, usage: MTLTextureUsage) -> MTLTexture? {
    let d = MTLTextureDescriptor.textureCubeDescriptor(
        pixelFormat: .rgba8Unorm, size: side, mipmapped: false)
    d.usage = usage
    return dev.makeTexture(descriptor: d)
}

/// Sample the centre of each face with a nearest filter, and the midpoint of
/// the +X/+Z edge with a linear one.
func cubeSampleCase(_ side: Int) {
    let label = "compute_cube_sample_\(side)x\(side)"
    let edgeLabel = "compute_cube_sample_edge_linear_\(side)x\(side)"
    guard let cube = makeCube(side, usage: [.shaderRead]) else {
        report(label, false, "makeTexture nil for a \(side)x\(side) cube")
        skipDependent(edgeLabel, label)
        return
    }
    for face in 0..<6 {
        let (r, g, b, a) = cubeFaceColour(face)
        let texels = [UInt8]((0..<(side * side)).flatMap { _ in [r, g, b, a] })
        texels.withUnsafeBytes { raw in
            cube.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0,
                         slice: face, withBytes: raw.baseAddress!,
                         bytesPerRow: side * 4, bytesPerImage: side * side * 4)
        }
    }
    let lanes = 7
    let out = dev.makeBuffer(length: lanes * 4, options: .storageModeShared)!
    memset(out.contents(), 0xEE, lanes * 4)
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline("sample_cube_faces"))
    enc.setTexture(cube, index: 0)
    enc.setBuffer(out, offset: 0, index: 0)
    enc.setBuffer(ran, offset: 0, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 8, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 0 {
        refused(label)
        skipDependent(edgeLabel, label)
        return
    }
    let got = Array(UnsafeBufferPointer(
        start: out.contents().bindMemory(to: UInt32.self, capacity: lanes), count: lanes))

    var wrong: [String] = []
    for face in 0..<6 {
        let (r, g, b, a) = cubeFaceColour(face)
        let want = pack(r, g, b, a)
        if got[face] != want { wrong.append("face\(face) want=\(hex(want)) got=\(hex(got[face]))") }
    }
    report(label, wrong.isEmpty,
           wrong.isEmpty ? "six faces in +X -X +Y -Y +Z -Z order" : wrong.joined(separator: " "))

    // The +X/+Z edge: a filter that crosses the seam blends the two faces
    // equally; one that clamps at the face edge returns one face's colour.
    let px = cubeFaceColour(0), pz = cubeFaceColour(4)
    let blend = [(px.0, pz.0), (px.1, pz.1), (px.2, pz.2), (px.3, pz.3)]
        .map { UInt8((Int($0.0) + Int($0.1) + 1) / 2) }
    let e = got[6]
    let channels = [UInt8(e & 0xFF), UInt8((e >> 8) & 0xFF), UInt8((e >> 16) & 0xFF), UInt8(e >> 24)]
    let near = zip(channels, blend).allSatisfy { abs(Int($0.0) - Int($0.1)) <= 2 }
    let want = pack(blend[0], blend[1], blend[2], blend[3])
    report(edgeLabel, near,
           near ? "the +X/+Z edge blends both faces"
                : "want≈\(hex(want)) (+X=\(hex(pack(px.0, px.1, px.2, px.3))) +Z=\(hex(pack(pz.0, pz.1, pz.2, pz.3)))) got=\(hex(e))")
}

/// Write every texel of every face from a kernel, then read each face back
/// with a blit, independently of any sampling path.
func cubeWriteCase(_ side: Int) {
    let label = "compute_cube_write_\(side)x\(side)"
    guard let cube = makeCube(side, usage: [.shaderWrite, .shaderRead]) else {
        report(label, false, "makeTexture nil for a \(side)x\(side) cube"); return
    }
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let faceBytes = side * side * 4
    let readback = dev.makeBuffer(length: 6 * faceBytes, options: .storageModeShared)!
    memset(readback.contents(), 0xEE, 6 * faceBytes)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline("write_cube_faces"))
    enc.setTexture(cube, index: 0)
    var s = UInt32(side)
    enc.setBytes(&s, length: 4, index: 1)
    enc.setBuffer(ran, offset: 0, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: (side + 3) / 4, height: (side + 3) / 4, depth: 6),
                             threadsPerThreadgroup: MTLSize(width: 4, height: 4, depth: 1))
    enc.endEncoding()
    let blit = cb.makeBlitCommandEncoder()!
    for face in 0..<6 {
        blit.copy(from: cube, sourceSlice: face, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: side, height: side, depth: 1),
                  to: readback, destinationOffset: face * faceBytes,
                  destinationBytesPerRow: side * 4, destinationBytesPerImage: faceBytes)
    }
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 0 {
        refused(label); return
    }
    let bytes = readback.contents().bindMemory(to: UInt8.self, capacity: 6 * faceBytes)
    var bad: [String] = []
    for face in 0..<6 {
        var faceBad = 0
        var first = ""
        for y in 0..<side {
            for x in 0..<side {
                let o = face * faceBytes + (y * side + x) * 4
                let want = [UInt8(x * 40 + 10), UInt8(y * 40 + 10), UInt8(face * 40 + 20), 0xFF]
                let got = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
                if got != want {
                    if faceBad == 0 { first = " first=(\(x),\(y)) want=\(want) got=\(got)" }
                    faceBad += 1
                }
            }
        }
        if faceBad > 0 { bad.append("face\(face) bad=\(faceBad)\(first)") }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "six faces written and read back in order" : bad.joined(separator: " "))
}
