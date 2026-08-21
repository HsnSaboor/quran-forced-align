import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

class RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler supporting Byte-Range requests for seamless audio scrubbing."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path) or os.path.isdir(path):
            return super().do_GET()

        range_header = self.headers.get('Range')
        if not range_header or not range_header.startswith('bytes='):
            return super().do_GET()

        file_size = os.path.getsize(path)
        ctype = self.guess_type(path)
        
        range_val = range_header.split('=')[1].strip()
        parts = range_val.split('-')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        
        if start >= file_size or end >= file_size or start > end:
            self.send_error(416, 'Requested Range Not Satisfiable')
            return

        content_length = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.send_header('Content-Length', str(content_length))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()

        with open(path, 'rb') as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(64 * 1024, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except (ConnectionResetError, BrokenPipeError):
                    break
                remaining -= len(data)

def run(port=8000):
    os.chdir('/home/saboor/code/quran-forced-align')
    server = HTTPServer(('0.0.0.0', port), RangeRequestHandler)
    print(f"🚀 Quran Forced-Alignment Web Player Server running at http://localhost:{port}/web-player/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
