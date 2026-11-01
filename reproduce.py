"""Regenerate the quadrature tables and figures: python reproduce.py."""
from fractions import Fraction as F
from math import isqrt, comb
from pathlib import Path
import json
import io
from certify import certify, uniform, write_csv

ROOT = Path(__file__).resolve().parent


def inv_floor(x, beta):
    y = 1/x
    return y.numerator//y.denominator if beta == 1 else isqrt(y.numerator//y.denominator)


def sequence_exceptional(a, b, beta):
    if a == 0:
        return True
    floor_b = inv_floor(b, beta)
    ceil_b = floor_b if F(1, floor_b**beta) == b else floor_b+1
    return ceil_b <= inv_floor(a, beta)


def sequence_value(x, beta):
    return F(0) if x == 0 else F(inv_floor(x, beta) % 2)


def sequence_oracle(beta, background=False):
    def oracle(a, b):
        exceptional = sequence_exceptional(a, b, beta)
        lo, hi = (F(0), F(1)) if exceptional else (sequence_value((a+b)/2, beta),)*2
        if background:
            lo += a*a/8
            hi += b*b/8
        return lo, hi
    return oracle


def cantor_oracle(a, b):
    generation = 1
    while True:
        if a > F(1,3) and b < F(2,3):
            return (F(generation % 2),)*2
        if b < F(1,3):
            a, b = 3*a, 3*b
        elif a > F(2,3):
            a, b = 3*a-2, 3*b-2
        else:
            return F(0), F(1)
        generation += 1


def jump(a, b):
    if b <= F(1,3):
        return F(0), F(0)
    if a > F(1,3):
        return F(1), F(1)
    return F(0), F(1)


def pulse(a, b):
    left, right = F(3699,10000), F(3701,10000)
    if b <= left or a >= right:
        return F(0), F(0)
    if a > left and b < right:
        return F(1), F(1)
    return F(0), F(1)


def reference(beta, M=100):
    low = F(0)
    for n in range(M):
        delta = sum(((-1)**j)*F(comb(n,j), (j+1)**beta) for j in range(n+1))
        assert delta >= 0
        low += delta/F(2**(n+1))
    return low, low+F(1,2**M)


def models():
    r1, r2 = reference(1), reference(2)
    return [
        ('One jump', jump, (F(2,3),)*2, [4,8,12,16,20], None, False),
        ('Narrow pulse', pulse, (F(1,5000),)*2, [4,8,12,16,20], None, False),
        ('f1', sequence_oracle(1), r1, [2,4,6,8,10], None, False),
        ('f2', sequence_oracle(2), r2, [3,5,7,10,13], None, False),
        ('f1+x2/8', sequence_oracle(1,True), tuple(x+F(1,24) for x in r1),
         [2,4,6,8,10], lambda a,b:not sequence_exceptional(a,b,1), True),
        ('Cantor gaps', cantor_oracle, (F(3,5),)*2, [2,3,4,5,6], None, False)]


def validate():
    checked = dict(sequence_cells=0, cantor_cells=0, uniform_comparisons=0, point_ranges=0)
    for beta in (1,2):
        for m in range(1,10):
            N = 2**m
            points = [F(1,k**beta) for k in range(1,N+2)]
            for j in range(N):
                a,b = F(j,N), F(j+1,N)
                explicit = a == 0 or any(a <= p <= b for p in points)
                assert sequence_exceptional(a,b,beta) == explicit
                checked['sequence_cells'] += 1
    # Independent retained-interval cover for the Cantor test.
    for m in range(1,10):
        N = 2**m
        intervals = [(F(0),F(1))]
        while intervals[0][1]-intervals[0][0] >= F(1,N):
            intervals = [(x,y) for a,b in intervals
                         for x,y in [(a,a+(b-a)/3),(b-(b-a)/3,b)]]
        for j in range(N):
            a,b=F(j,N),F(j+1,N)
            explicit=any(a<=v and u<=b for u,v in intervals)
            assert (cantor_oracle(a,b)==(0,1)) == explicit
            checked['cantor_cells'] += 1
    for name,oracle,ref,powers,regular,bg in models():
        for m in range(1,8):
            agg=uniform(oracle,m,regular,bg)
            vals=[oracle(F(j,2**m),F(j+1,2**m)) for j in range(2**m)]
            assert agg['lower']==sum(x[0] for x in vals)/2**m
            assert agg['upper']==sum(x[1] for x in vals)/2**m
            checked['uniform_comparisons'] += 1
    for beta in (1,2):
        oracle=sequence_oracle(beta)
        for m in range(1,8):
            for j in range(2**m):
                a,b=F(j,2**m),F(j+1,2**m)
                l,u=oracle(a,b)
                for x in (a,(3*a+b)/4,(a+b)/2,(a+3*b)/4,b):
                    assert l<=sequence_value(x,beta)<=u
                    checked['point_ranges']+=1
    return checked


def save_figure(fig, path):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf')
    data = buffer.getvalue()
    assert data.startswith(b'%PDF') and b'%%EOF' in data[-64:]
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    ROOT.joinpath('figures').mkdir(exist_ok=True)
    ROOT.joinpath('data').mkdir(exist_ok=True)
    rows, curves, selected = [], [], []
    for name,oracle,ref,powers,regular,bg in models():
        cache={}
        for power in powers:
            eps=F(1,2**power)
            r=certify(oracle,eps)
            assert r['lower'] <= ref[0] <= ref[1] <= r['upper']
            for m in range(25):
                if m not in cache:
                    cache[m]=uniform(oracle,m,regular,bg)
                if cache[m]['radius'] <= eps:
                    break
            else:
                raise RuntimeError('Uniform depth limit exceeded')
            row=dict(model=name,tolerance=eps,**r,uniform_cells=2**m,
                     uniform_lower=cache[m]['lower'],uniform_upper=cache[m]['upper'],
                     uniform_accelerator_queries=cache[m]['queries'])
            rows.append(row)
            if power==powers[-1]:
                selected.append(row)
        for m,r in sorted(cache.items()):
            curves.append(dict(model=name,depth=m,**r))
        print(name, 'selected:', selected[-1]['cells'], selected[-1]['queries'],
              selected[-1]['uniform_cells'], flush=True)
    checks=validate()
    checks['certificates']=len(rows)
    write_csv(ROOT/'data/certificates.csv',rows)
    write_csv(ROOT/'data/uniform_curves.csv',curves)
    (ROOT/'data/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7,4.7),layout='constrained')
    for ax,name,title in zip(axes.flat,['f1','f2','f1+x2/8','Cantor gaps'],
                             [r'$\beta=1$',r'$\beta=2$',r'$f_1+x^2/8$','Cantor gaps']):
        rr=[r for r in rows if r['model']==name]
        uu=[r for r in curves if r['model']==name]
        ax.loglog([r['cells'] for r in rr],[float(r['radius']) for r in rr],'o-',
                  color='#245775',markersize=3,label='Adaptive')
        ax.loglog([r['cells'] for r in uu],[float(r['radius']) for r in uu],'--',
                  color='#9a4e2f',label='Uniform')
        ax.set(title=title,xlabel='Partition cells',ylabel='Certificate radius')
        ax.legend(frameon=False,fontsize=8)
    save_figure(fig, ROOT/'figures/convergence.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7,3.2),layout='constrained',sharex=True,
                          gridspec_kw={'height_ratios':[1,1.5]})
    for k in range(1,65):
        axes[0].plot([1/(k+1),1/k],[k%2,k%2],color='#245775',lw=1)
    axes[0].axvspan(0,1/65,color='#245775',alpha=.15)
    axes[0].set(ylabel=r'$f_1(x)$',yticks=[0,1])
    for level,power in enumerate([3,5,7],1):
        result=certify(sequence_oracle(1),F(1,2**power),keep_leaves=True)
        intervals=[(float(a),float(b-a)) for a,(b,l,u,d) in result['leaves'].items() if u>l]
        axes[1].broken_barh(intervals,(level-.23,.46),facecolors='#245775')
        axes[1].text(.99,level+.26,f'$\\varepsilon=2^{{-{power}}}$; {result["cells"]} cells',
                     ha='right',va='bottom',fontsize=8)
    axes[1].set(xlabel='Input',ylabel='Refinement stage',yticks=[1,2,3],ylim=(.5,3.9),xlim=(0,1))
    save_figure(fig, ROOT/'figures/covers.pdf')
    plt.close(fig)
    print('Validation:',checks)


if __name__=='__main__':
    main()
