# 60-second explainer video

`dna_storage_markers.mp4` (1920x1080, 30 fps, 60 s, no audio): file → bits → DNA letters → strands → synthesis and
storage → sequencing with errors → DNAFormer → synchronization markers (our results) → decoded file.

Rebuild: `node render.js dna_storage_markers.mp4` (needs Playwright's Chromium and ffmpeg).
Edit texts, timings or colours in `scene.html` (one function per scene; `SCENES` lists start and end times).
`node render.js x --stills` writes still frames to `stills/` for a quick check.
