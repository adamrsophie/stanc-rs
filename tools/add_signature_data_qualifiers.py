import re,sys
gen=open(sys.argv[1]).read()
dump=open(sys.argv[2]).read().splitlines()
def split_top(s,sep):
    out=[];d=0;cur=''
    for ch in s:
        if ch in '([': d+=1
        if ch in ')]': d-=1
        if ch==sep and d==0: out.append(cur);cur=''
        else: cur+=ch
    if cur.strip(): out.append(cur)
    return [x.strip() for x in out]
def ty(t):
    t=t.strip().replace('UnsizedType.','')
    while t.startswith('(') and t.endswith(')') and balanced(t[1:-1]): t=t[1:-1].strip()
    if t.startswith('UFun'): return 'FUN'
    m=re.match(r'UArray\s*(.*)$',t,re.S)
    if m:
        inner=ty(m.group(1)); n=1
        if inner.startswith('array['):
            k=inner.index(']'); n+=inner[6:k].count(',')+1; inner=inner[k+2:]
        return 'array['+','*(n-1)+'] '+inner
    return {'UReal':'real','UInt':'int','UVector':'vector','URowVector':'row_vector','UMatrix':'matrix','UComplex':'complex'}[t]
def balanced(s):
    d=0
    for c in s:
        d+= c=='(' ; d-= c==')'
        if d<0: return False
    return d==0
entries=[]
for m in re.finditer(r'add_qualified\s*\(\s*"(\w+)"\s*,\s*ReturnType[^,]*?,\s*\[(.*?)\]\s*,\s*(SoA|AoS)\s*\)',gen,re.S):
    name,args=m.group(1),m.group(2)
    flags=[];tys=[]
    for a in split_top(args,';'):
        a=a.strip()
        while a.startswith('(') and a.endswith(')') and balanced(a[1:-1]): a=a[1:-1].strip()
        ad,t=a.split(',',1)
        flags.append('DataOnly' in ad); tys.append(ty(t))
    entries.append((name,flags,tys))
# to_int loop and wiener
for i in range(8):
    for t in ['real','int']:
        entries.append(('to_int',[True],[('array['+','*(i-1)+'] ' if i else '')+t]))
for name,lens in [('wiener_lpdf',[6,8]),('wiener_lcdf_unnorm',[5,8]),('wiener_lccdf_unnorm',[5,8])]:
    for t in ['real','vector']:
        for n in lens: entries.append((name,[False]*n+[True],[t]*n+['real']))
print(len(entries),"entries",file=sys.stderr)
def parse_args(line):
    o=line.index('(');d=0
    for i in range(o,len(line)):
        d+= line[i]=='(' ; d-= line[i]==')'
        if d==0: return line[:o],split_top(line[o+1:i],','),line[i+1:]
used=0;out=[]
for line in dump:
    name,args,rest=parse_args(line)
    for en,fl,tys in entries:
        if en!=name or len(fl)!=len(args): continue
        ok=all((t=='FUN' and a.startswith('(')) or a==t for a,t in zip(args,tys))
        if ok:
            args=[('data '+a if f and not a.startswith('(') else a) for a,f in zip(args,fl)]; used+=1; break
    out.append(name+'('+', '.join(args)+')'+rest)
print(used,"lines updated",file=sys.stderr)
print('\n'.join(out))
