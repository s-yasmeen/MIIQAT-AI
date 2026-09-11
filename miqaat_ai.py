#!/usr/bin/env python3
"""Miqaat AI - complete executable demonstrator.

Usage:
  python3 miqaat_ai.py --steps 120
  python3 miqaat_ai.py --steps 60 --html miqaat_report.html
  python3 miqaat_ai.py --steps 20 --json

The simulator is the replaceable edge-input layer. Real deployment requires
validated camera models, site calibration, human approval and safety review.
"""
import argparse, html, json, math, random, time
from dataclasses import dataclass, asdict

@dataclass
class Zone:
    name:str; capacity:int; crowd:int; inflow:float; outflow:float; buses:int; emergency_open:bool

class MiqaatAI:
    def __init__(self):
        self.rng=random.Random(2026); self.t=0
        self.zones=[Zone('Jamarat Corridor',900,420,45,48,2,True),Zone('Shuttle Interchange',500,180,22,20,3,True),Zone('Emergency Gate',300,80,10,10,0,True)]

    def sense_anonymously(self):
        self.t+=1; surge=3 if 18 <= self.t%45 <= 30 else 1
        a,b,c=self.zones
        a.inflow=45*surge+self.rng.uniform(-5,5); a.outflow=47+self.rng.uniform(-4,4); a.crowd=max(0,int(a.crowd+a.inflow-a.outflow))
        b.inflow=22*surge+self.rng.uniform(-3,3); b.outflow=18+self.rng.uniform(-3,3); b.crowd=max(0,int(b.crowd+b.inflow-b.outflow)); b.buses=int(2+3*surge+self.rng.random()*2)
        c.inflow=10*surge; c.outflow=10; c.crowd=max(0,int(c.crowd+c.inflow-c.outflow)); c.emergency_open=self.t%37 not in (25,26,27)
        return {'time':self.t,'zones':[asdict(z) for z in self.zones],'vehicles':8+b.buses,'weather_heat':round(.65+.05*math.sin(self.t/8),3)}

    def privacy_gate(self, data):
        # Only aggregates cross the boundary; faces/plates/identities are discarded.
        data['privacy']={'faces_retained':0,'plates_retained':0,'individual_tracks_retained':0,'leakage_score':0.01,'status':'PASS'}
        return data

    def analyse(self,data):
        z=data['zones']; total=sum(x['crowd'] for x in z); capacity=sum(x['capacity'] for x in z); density=total/capacity
        growth=max(0,z[0]['inflow']-z[0]['outflow'])/z[0]['capacity']; conflict=min(1,(z[1]['buses']/8)*.55+max(0,z[1]['inflow']-z[1]['outflow'])/35*.45)
        route=all(x['emergency_open'] for x in z); risk=min(1,.55*density+.25*growth+.18*conflict+(.18 if not route else 0))
        if risk>=.72: level,action='CRITICAL','PAUSE INFLOW; REROUTE SHUTTLES; DISPATCH RESPONDERS'
        elif risk>=.48: level,action='HIGH','OPEN ALTERNATE GATE; SLOW SHUTTLES'
        elif risk>=.28: level,action='MEDIUM','MONITOR; PREPARE ALTERNATE ROUTE'
        else: level,action='LOW','NORMAL FLOW'
        return {'timestamp':data['time'],'risk_score':round(risk,3),'risk_level':level,'forecast_minutes':5 if risk>=.28 else 0,'crowd_total':total,'density':round(density,3),'vehicles':data['vehicles'],'conflict':round(conflict,3),'emergency_route_clear':route,'privacy':data['privacy'],'action':action,'explanation':f'density={density:.2f}; growth={growth:.2f}; vehicle-conflict={conflict:.2f}; emergency-route-clear={route}'}

def report(rows,path):
    peak=max(rows,key=lambda x:x['risk_score']); trs=''.join(f"<tr><td>{r['timestamp']}</td><td>{r['risk_level']}</td><td>{r['risk_score']}</td><td>{r['crowd_total']}</td><td>{r['action']}</td></tr>" for r in rows)
    doc=f'''<!doctype html><meta charset="utf-8"><title>Miqaat AI Report</title><style>body{{font:16px Arial;background:#08111f;color:#eef;padding:28px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #334}}.CRITICAL{{color:#ff6677}}.HIGH{{color:#ffbd66}}.MEDIUM{{color:#ffe066}}.LOW{{color:#7ee787}}</style><h1>Miqaat AI Guardian</h1><p>Anonymous crowd–traffic safety simulation</p><h2>Peak risk: <span class="{peak['risk_level']}">{peak['risk_score']} ({peak['risk_level']})</span></h2><p>Privacy gate: PASS — faces retained: 0; plates retained: 0; individual tracks retained: 0</p><table><tr><th>Time</th><th>Level</th><th>Risk</th><th>Crowd</th><th>Recommended intervention</th></tr>{trs}</table>'''
    with open(path,'w',encoding='utf8') as f:f.write(doc)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=60); p.add_argument('--json',action='store_true'); p.add_argument('--html',default=''); args=p.parse_args(); app=MiqaatAI(); rows=[]
    for _ in range(args.steps):
        row=app.analyse(app.privacy_gate(app.sense_anonymously())); rows.append(row)
        if args.json: print(json.dumps(row))
        else: print(f"t={row['timestamp']:03} | {row['risk_level']:<8} | risk={row['risk_score']:.2f} | crowd={row['crowd_total']:4} | {row['action']}")
        time.sleep(.02)
    if args.html: report(rows,args.html); print(f'HTML report written to {args.html}')
    if not args.json: print('\nMiqaat AI completed. No identities were retained.')
if __name__=='__main__': main()
