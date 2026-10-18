"""Original-key streaming adapter with bounded bodies and per-request random seeds."""
import hmac,http.client,http.server,json,pathlib,secrets,os
from request_body import read_body,RequestError

KEY=None
BACKEND_PORT=int(os.environ.get('BACKEND_PORT','8889'))

def randomize_seed(body,method,path):
    route=path.split('?',1)[0].rstrip('/')
    if method!='POST' or not (route.endswith('/completions') or route in ('/v1/responses','/responses')):
        return body
    try:
        payload=json.loads(body)
    except (ValueError,UnicodeDecodeError):
        return body
    if not isinstance(payload,dict) or payload.get('seed') is not None:
        return body
    payload['seed']=secrets.randbits(63)
    return json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode('utf-8')

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def refuse(self,status,message,kind):
        body=json.dumps({'error':{'message':message,'type':kind}}).encode()
        self.close_connection=True
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close')
        self.end_headers();self.wfile.write(body)
    def forward(self):
        if self.path not in ('/health','/metrics') and not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+KEY):
            return self.refuse(401,'Unauthorized','authentication_error')
        conn=http.client.HTTPConnection('127.0.0.1',BACKEND_PORT,timeout=1800)
        try:
            body=read_body(self,limit=96*1024**2)
            body=randomize_seed(body,self.command,self.path)
            headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','authorization','connection','transfer-encoding','content-length')}
            conn.request(self.command,self.path,body,headers);r=conn.getresponse()
            self.send_response(r.status)
            for k,v in r.getheaders():
                if k.lower() not in ('connection','transfer-encoding','content-length','server','date'):self.send_header(k,v)
            self.send_header('Connection','close');self.end_headers();self.close_connection=True
            while chunk:=r.read1(65536):self.wfile.write(chunk);self.wfile.flush()
        except RequestError as exc:self.refuse(400,str(exc),'invalid_request_error')
        except (BrokenPipeError,ConnectionResetError):pass
        except (OSError,http.client.HTTPException):self.close_connection=True
        finally:conn.close()
    do_GET=forward
    do_POST=forward
class Server(http.server.ThreadingHTTPServer):
    daemon_threads=True
if __name__=='__main__':
    KEY=pathlib.Path(os.environ.get('API_KEY_FILE','/run/secrets/api-key')).read_text().strip()
    if not KEY: raise SystemExit('API key file is empty')
    Server((os.environ.get('API_HOST','0.0.0.0'),int(os.environ.get('API_PORT','8888'))),Handler).serve_forever()
