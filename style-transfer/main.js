const displayCanvas = document.getElementById('displayCanvas');
const video = document.createElement('video');
const canvas = document.createElement('canvas');
const canvasContext = canvas.getContext('2d');
canvas.width = displayCanvas.width;
canvas.height = displayCanvas.height;
const modelPath = '/build/mosaic_36.onnx';
const options = { executionProviders: ['wasm'], executionMode: 'parallel', graphOptimizationLevel: 'all'};

window.timestamps = Array(25).fill(performance.now());

const renderFrames = async () => {
  canvasContext.drawImage(video, 0, 0, window.width, window.height, 0, 0, canvas.width, canvas.height);
  const input = await ort.Tensor.fromImage(canvasContext.getImageData(0,0, canvas.width, canvas.height));
  const {output} = await window.session.run({input: input});
  displayCanvas.getContext('2d').putImageData(output.toImageData({format: 'RGB', norm: {mean: 1} } ), 0, 0);

  window.timestamps.shift();
  window.timestamps[24] = performance.now();
  document.getElementById('fps').innerHTML = 1000 / (window.timestamps[24] - window.timestamps[0]) * 25
  requestAnimationFrame(renderFrames);
};

window.onload = () => {
  navigator.mediaDevices.getUserMedia({video: true, audio: false}).then( async stream => {
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = 6;

    const session = await ort.InferenceSession.create(modelPath, options);
    window.session = session;
    video.srcObject = stream;
    video.play();
    window.width = stream.getVideoTracks()[0].getSettings().width;
    window.height = stream.getVideoTracks()[0].getSettings().height;

    renderFrames();
  }).catch(e => console.error(e));
}