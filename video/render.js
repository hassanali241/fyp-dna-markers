// Renders scene.html frame by frame into an MP4.  node render.js [out.mp4] [--stills]
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs');
(async () => {
  const out = process.argv[2] || 'dna_storage_markers.mp4';
  const stills = process.argv.includes('--stills');
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('file://' + path.join(__dirname, 'scene.html'));
  await p.evaluate(() => document.fonts.ready);
  const fps = await p.evaluate(() => window.FPS), dur = await p.evaluate(() => window.DUR);
  if (stills) {
    fs.mkdirSync('stills', { recursive: true });
    for (const t of [2.5, 9, 18, 24, 31, 39.5, 47, 54.5, 58.2, 59.5]) {
      await p.evaluate(t => render(t), t);
      const d = await p.evaluate(() => document.getElementById('c').toDataURL('image/jpeg', 0.85));
      fs.writeFileSync(`stills/t${t}.jpg`, Buffer.from(d.split(',')[1], 'base64'));
    }
    await b.close(); return;
  }
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', '-movflags', '+faststart', out]);
  const n = Math.round(fps * dur);
  for (let i = 0; i < n; i++) {
    const d = await p.evaluate(t => { render(t); return document.getElementById('c').toDataURL('image/jpeg', 0.95); }, i / fps);
    if (!ff.stdin.write(Buffer.from(d.split(',')[1], 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log('frame', i, '/', n);
  }
  ff.stdin.end(); await new Promise(r => ff.on('close', r)); await b.close(); console.log('wrote', out);
})();
