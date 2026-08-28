#!/usr/bin/env python3

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from dotenv import load_dotenv

    load_dotenv() 
except ImportError:
    pass

from agent.agent import ReasoningAgent

_agent = ReasoningAgent()


class SolveHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/solve":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body or b"{}")
            question = payload.get("question", "")
        except (json.JSONDecodeError, ValueError):
            self._send(400, {"error": "invalid JSON body"})
            return

        result = _agent.solve(question)
        self._send(200, result)

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args): 
        pass


def main() -> None:
    port = 8000
    httpd = HTTPServer(("0.0.0.0", port), SolveHandler)
    print(f"Serving /solve on http://localhost:{port}  (POST {{'question': '...'}})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
