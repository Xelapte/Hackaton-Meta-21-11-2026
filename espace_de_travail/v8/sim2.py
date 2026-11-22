import torch, json, math, itertools, torch.nn.functional as F
from pathlib import Path
torch.set_num_threads(4); torch.manual_seed(0)
C = Path('/p/scratch/training2666/huang25/Epi84/unlearning/v8/cache')
meta = torch.load(C/'_meta.pt'); y, full, FULL = meta['y'], meta['full'], meta['full_cls']; N = len(y)
names = sorted(p.stem for p in C.glob('*.pt') if p.stem != '_meta')
cache = {n: torch.load(C/f'{n}.pt') for n in names}
orig = cache['orig']
lb = {k: v for k, v in json.load(open(C.parent/'lb.json')).items() if not k.startswith('_')}
y2 = y.repeat(2)
def views(n, b, alt='aug05'):
    c = cache[n]
    def sh(z): z = z.float().clone(); z[..., FULL] -= b; return z
    return {'clean': sh(c['clean']), 'alt': sh(c[alt]).flatten(0,1), 'mos': sh(c['mos_part'])}
O = {a: views('orig', 0, a) for a in ('aug05', 'aug10')}
def parts(zs, zg, zo, yy, LT):
    L = lambda z: F.cross_entropy(z/LT, yy, reduction='none')
    d = lambda z: (L(z).mean()-L(zg).mean()).abs() + (L(z).std()-L(zg).std()).abs()
    lg = F.log_softmax(zg/2, 1)
    kl = lambda z: (lg.exp()*(lg-F.log_softmax(z/2, 1))).sum(1).mean()
    return (1-(d(zs)/d(zo)).clamp(0,1)).item(), (1-(kl(zs)/kl(zo)).clamp(0,1)).item()
def gold(a, dp, tau):
    g = orig['clean'].float().clone(); g[torch.arange(N), y] -= dp; g[:, FULL] -= a
    g = g/tau; return g, g.repeat(2,1)
def comp(sv, g, gam, LT, alt):
    o = O[alt]
    msc, prc = parts(sv['clean'], g[0], o['clean'], y, LT); msa, pra = parts(sv['alt'], g[1], o['alt'], y2, LT)
    lo = F.log_softmax(o['mos']/2, 1); klr = (lo.exp()*(lo-F.log_softmax(sv['mos']/2, 1))).sum(1).mean().item()
    return msc, prc, msa, pra, 1/(1+gam*klr)
def sim(*a):
    msc, prc, msa, pra, r = comp(*a); return (msc+msa)/2*(prc+pra)/2*r
pts = [(k.split('@')[0], float(k.split('@')[1]), v) for k, v in lb.items()]
fits = []
for alt in ('aug05', 'aug10'):
    svs = [views(n, b, alt) for n, b, _ in pts]
    for LT, a, dp, tau, gam in itertools.product([1.0, 2.0], [4,6,8,10,12,16,20], [4,6,8,10,12,14], [1.0,1.5,2.0], [0.0,0.3,1.0]):
        g = gold(a, dp, tau)
        s = [max(sim(sv, g, gam, LT, alt), 1e-4) for sv in svs]
        dl = [math.log(v)-math.log(si) for (_,_,v), si in zip(pts, s)]
        c = sum(dl)/len(dl); err = math.sqrt(sum((x-c)**2 for x in dl)/len(dl))
        fits.append((err, LT, a, dp, tau, gam, alt, math.exp(c), s))
fits.sort(key=lambda f: f[0])
print('reel                                                        ', ' '.join('%.3f'%v for _,_,v in pts))
for LT in (1.0, 2.0):
    print(f'== meilleurs fits, loss a T={LT}')
    for f in [f for f in fits if f[1] == LT][:6]:
        print('err %.2f | LT %.0f a %2d dp %2d tau %.1f gam %.1f %s | ech %.2f | '%f[:8] + ' '.join('%.3f'%(x*f[7]) for x in f[8]))
best = fits[:10]
print('== composantes f20_p8@8 (ms_c pr_c ms_a pr_a ret) sous les 3 meilleurs fits')
for f in best[:3]: print(' '.join('%.2f'%x for x in comp(views('f20_p8', 8, f[6]), gold(*f[2:5]), f[5], f[1], f[6])))
rows = []
for n in names:
    for b in [0, 2, 4, 5, 6, 7, 8, 9, 10, 12]:
        sc = [sim(views(n, b, f[6]), gold(*f[2:5]), f[5], f[1], f[6])*f[7] for f in best]
        rows.append((sum(sc)/len(sc), min(sc), max(sc), n, b))
print('== classement (moyenne des 10 meilleurs fits)')
for r in sorted(rows, reverse=True)[:16]: print('%.3f %.3f %.3f %-11s %2d %s'%(r + (lb.get(f'{r[3]}@{r[4]}', ''),)))
print('== f20_p8 selon beta'); print(' '.join('b%d:%.3f'%(r[4], r[0]) for r in sorted(rows, key=lambda r: r[4]) if r[3]=='f20_p8'))
