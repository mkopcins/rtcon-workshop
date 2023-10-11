const { readFileSync, readFile } = require('fs');
const https = require('https');


const options = {
    key: readFileSync("./certs/myCA.key"),
    cert: readFileSync("./certs/myCA.crt"),
    passphrase: 'kappa'
}

const host = '0.0.0.0';
const port = 8000;
const typeMap = {
  'html': 'text/html',
  'js': 'text/javascrsipt',
  'wasm': 'application/wasm',
};

const requestListener = function (req, res) {
    let url = req.url;
    if (url === "/") url = "/index.html";
    for (const [key, value] of Object.entries(typeMap)) {
      if (url.endsWith(key)) res.setHeader("Content-Type", value);
    }
    readFile(__dirname + url, (err, contents) => { 
      if (err) {
        console.warn("error: ", url);
        console.warn("Error", err.stack);
        console.warn("Error", err.name);
        console.warn("Error", err.message);
        res.writeHead(500);
        res.end();
        return;
      }
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.writeHead(200);
      res.end(contents);
    });
  };

const server = https.createServer(options, requestListener);
server.listen(port, host, () => {
  console.log(`Server is running on https://${host}:${port}`);
});