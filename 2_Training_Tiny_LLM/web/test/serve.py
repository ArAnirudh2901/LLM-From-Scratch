# Local static server for test/bench.html with correct MIME types (.mjs, .wasm).
import http.server, socketserver, sys
H = http.server.SimpleHTTPRequestHandler
H.extensions_map.update({'.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'})


def do_POST(self):   # bench.html posts its results here; written to test/result.txt
    body = self.rfile.read(int(self.headers['Content-Length'])).decode()
    open('test/result.txt', 'a').write(body + '\n')
    self.send_response(204); self.end_headers()


H.do_POST = do_POST
_end = H.end_headers


def end_headers(self):   # cross-origin isolation: lets the browser use threads (SharedArrayBuffer)
    self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
    self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
    self.send_header('Cache-Control', 'no-store')
    _end(self)


H.end_headers = end_headers
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H) as s:
    s.serve_forever()
