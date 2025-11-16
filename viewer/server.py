#!/usr/bin/env python3
"""Simple HTTP server with Jinja template rendering for DZI viewer"""

import os
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from jinja2 import Template


class DZIServerHandler(SimpleHTTPRequestHandler):
  """HTTP handler with Jinja template rendering"""

  def do_GET(self):
    if self.path == '/' or self.path == '/index.html':
      self.send_index()
    else:
      # Serve static files
      super().do_GET()

  def send_index(self):
    """Render and send index.html from template"""
    # Scan DZI files on each request
    dzi_list = scan_dzi_files()

    # Read template
    template_path = Path('template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
      template_content = f.read()

    # Render template with DZI list
    template = Template(template_content)
    html = template.render(dzi_list=dzi_list)

    # Send response
    self.send_response(200)
    self.send_header('Content-Type', 'text/html; charset=utf-8')
    self.end_headers()
    self.wfile.write(html.encode('utf-8'))


def scan_dzi_files():
  """Scan dzi directory for available DZI files"""
  dzi_dir = Path('dzi')
  dzi_files = []

  if dzi_dir.exists():
    # Find all .dzi files in dzi/<name>/*.dzi pattern
    for dzi_file in dzi_dir.glob('*/*.dzi'):
      # Get relative path from viewer root
      rel_path = str(dzi_file)
      # Extract name from .dzi file itself (not folder name)
      # e.g., dzi/25-0452_1/sample.dzi -> sample
      name = dzi_file.stem
      dzi_files.append({
        'name': name,
        'path': rel_path
      })

  # Sort by name
  dzi_files.sort(key=lambda x: x['name'])
  return dzi_files


def run_server(port=8000):
  """Run the server"""
  server_address = ('', port)
  httpd = HTTPServer(server_address, DZIServerHandler)
  print(f'Server running at http://localhost:{port}/')
  print(f'Open http://localhost:{port}/ in your browser')
  print('DZI files will be scanned on each request')
  httpd.serve_forever()


if __name__ == '__main__':
  import sys
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

  # Change to viewer directory
  script_dir = Path(__file__).parent
  os.chdir(script_dir)

  run_server(port)
