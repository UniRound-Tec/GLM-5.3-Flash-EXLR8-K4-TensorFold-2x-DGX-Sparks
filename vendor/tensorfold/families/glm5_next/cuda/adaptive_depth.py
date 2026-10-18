"""Current-round survival / measured cost, with request-local conditional calibration.

No target sampling changes. DFlash's full block cost is charged even when K=0
is selected after drafting. Rank 0 fits costs; the existing ROUND message carries
its quantized curve. Both ranks learn confidence from the same verified prefix.
"""
import math
import time


class AdaptiveDepth:
    empirical = True
    adaptive = True

    def __init__(self, most, costs, mode=1):
        if not 1 <= most <= 7 or mode not in (0,1):
            raise ValueError('adaptive depth supports 1..7 and modes 0/1')
        self.most,self.mode,self.current=most,mode,most
        self.beta=2.0;self.confidence=0.0;self.shared=False
        self.prior=[float(v)+float(costs['block'])+float(costs.get('taps_row',0)) for v in costs['verify'][:most+1]]
        if len(self.prior)!=most+1 or any(not math.isfinite(v) or v<=0 for v in self.prior):
            raise ValueError('invalid adaptive cost prior')
        self.curve=list(self.prior);self.bias=[0.0]*most;self.n=[0]*most
        self.fit=[0.0]*5;self.offset=0.;self.slope=0.
        self.rounds=0;self.hist={};self.cost_samples={};self.rejected_costs=0;self.shared_fallbacks=0
        self.selected_zero=0;self.early_stops=0;self.selector_ns=0;self.selector_rounds=0
        self.last=None;self.qs=[];self.pred=[];self.chosen=None

    @property
    def chain_rule(self):
        return {'beta':self.beta} if self.shared else {'beta':self.beta,'controller':self}

    def set_shared(self, shared):
        self.shared=bool(shared);self.confidence=.3 if self.shared else 0.
        self.qs=[];self.pred=[];self.chosen=None;self.last=None

    def choose(self, remaining):
        self.current=min(self.most,max(0,remaining))
        if self.shared:self.shared_fallbacks+=1
        return self.current

    def next(self, drafted, accepted):
        return self.most

    def wire(self):
        curve=[]
        for k,p in enumerate(self.prior):
            v=max(1.,p+self.offset+self.slope*k)
            if curve:v=max(v,curve[-1]/1000+.25)
            curve.append(round(v*1000))
        self.receive(curve)
        return curve

    def receive(self, curve):
        if len(curve)!=self.most+1 or any(type(v) is not int or not 0<v<2000000 for v in curve):
            raise ValueError('invalid adaptive cost payload')
        if any(b<a for a,b in zip(curve,curve[1:])):raise ValueError('nonmonotone adaptive costs')
        self.curve=[v/1000 for v in curve]

    def start(self, depth):
        self.t0=time.perf_counter_ns();self.depth=depth
        self.qs=[];self.pred=[];self.expected=[1.];self.reach=1.;self.raw_reach=1.
        self.best=0;self.best_score=1/self.curve[0];self.fnc=None

    def push(self, q):
        q=min(1.,max(0.,float(q)));j=len(self.qs);self.qs.append(q)
        # A bounded log-odds offset preserves endpoint certainty; no extra head.
        e=math.exp(self.bias[j]) if self.mode else 1.
        p=q*e/(1-q+q*e) if 0<q<1 else q
        self.pred.append(p);self.reach*=p;self.raw_reach*=q
        self.expected.append(self.expected[-1]+self.reach);k=j+1
        if k>1 and self.fnc is None and self.raw_reach<.3:self.fnc=k-1
        score=self.expected[k]/self.curve[k]
        if score>self.best_score:self.best,self.best_score=k,score
        # Safe upper bound: every unvisited confidence could be 1. Only prune
        # once the FNC fallback is known, so its 2% decision margin is exact.
        if self.fnc is not None and k<self.depth and all(
            (self.expected[k]+(h-k)*self.reach)/self.curve[h]<=self.best_score
            for h in range(k+1,self.depth+1)):
            self.early_stops+=1;return False
        return True

    def finish(self):
        fallback=self.fnc if self.fnc is not None else len(self.qs)
        base=self.expected[fallback]/self.curve[fallback]
        chosen=self.best if self.best_score>1.02*base else fallback
        assert 0<=chosen<=len(self.qs)<=self.most
        self.chosen=chosen;self.selected_zero+=int(chosen==0)
        self.selector_rounds+=1;self.selector_ns+=time.perf_counter_ns()-self.t0
        return chosen

    def record(self, drafted, accepted, usable=True):
        self.last=None
        if not usable or self.shared or self.chosen is None:return
        if not 0<=accepted<=drafted<=self.chosen:raise ValueError('invalid adaptive observation')
        self.rounds+=1;self.hist[drafted]=self.hist.get(drafted,0)+1
        # After the first rejection later target rows are under a wrong prefix.
        # Unverified suffixes are missing observations, never negative labels.
        for j in range(min(drafted,accepted+1)):
            self.n[j]+=1
            if self.mode:
                y=float(j<accepted)
                self.bias[j]=max(-1.5,min(1.5,self.bias[j]+.12*(y-self.pred[j]-.04*self.bias[j])))
        self.last=drafted

    def observe_ms(self, ms):
        k,self.last=self.last,None
        if k is None or not math.isfinite(ms) or not 1<ms<2000:return
        prediction=self.prior[k]+self.offset+self.slope*k
        if abs(ms-prediction)>max(12.,.3*prediction):self.rejected_costs+=1;return
        # EW ridge regression of runtime residual vs K; O(1), no GPU timing sync.
        n,x,xx,y,xy=[v*.97 for v in self.fit];res=ms-self.prior[k]
        self.fit=[n+1,x+k,xx+k*k,y+res,xy+k*res]
        n,x,xx,y,xy=self.fit;aa=n+2.;bb=xx+8.;det=aa*bb-x*x
        self.offset=max(-3.,min(20.,(y*bb-xy*x)/det))
        self.slope=max(-1.5,min(4.,(xy*aa-y*x)/det))
        rec=self.cost_samples.get(k,[0,0.]);self.cost_samples[k]=[rec[0]+1,rec[1]+ms]

    def describe(self):
        return {'algorithm':'fadapt-v1','mode':self.mode,'rounds':self.rounds,'k_histogram':self.hist,
          'postdraft_k0':self.selected_zero,'shared_fallbacks':self.shared_fallbacks,
          'confidence_logodds_bias':self.bias,'position_observations':self.n,
          'prior_ms':self.prior,'curve_ms':self.curve,'cost_offset_ms':self.offset,'cost_slope_ms':self.slope,
          'observed_cost_ms':{str(k):s/n for k,(n,s) in self.cost_samples.items()},
          'cost_rejections':self.rejected_costs,'early_bound_stops':self.early_stops,
          'chain_selection_mean_us':self.selector_ns/max(1,self.selector_rounds)/1000}


def pack_costs(lanes):
    adaptive=[l for l in lanes if getattr(l.policy,'adaptive',False)]
    if not adaptive:return []
    out=[903,len(adaptive)]
    for l in adaptive:
        curve=l.policy.wire();out.extend([l.sid,len(curve),*curve])
    return out


def receive_costs(lanes,payload):
    wanted={l.sid:l.policy for l in lanes if getattr(l.policy,'adaptive',False)}
    if not payload:
        if wanted:raise ValueError('missing adaptive cost payload')
        return
    if payload[:2]!=[903,len(wanted)]:raise ValueError('invalid adaptive payload header')
    pos=2;seen=set()
    for _ in wanted:
        sid,n=payload[pos:pos+2];pos+=2
        if sid not in wanted or sid in seen or n!=wanted[sid].most+1:raise ValueError('invalid adaptive sid/length')
        wanted[sid].receive(payload[pos:pos+n]);pos+=n;seen.add(sid)
    if pos!=len(payload):raise ValueError('trailing adaptive payload')
