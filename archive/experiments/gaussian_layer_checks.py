#!/usr/bin/env python3
"""Dependency-free 2D algebra diagnostics. Not a learning benchmark.
Run: python3 experiments/gaussian_layer_checks.py
Checks fixed before implementation in outputs/.plans/gaussian-gnn-plan.md.
"""
import math, random, json, sys, platform
from pathlib import Path
I = [[1.,0.],[0.,1.]]
def tr(A): return [list(x) for x in zip(*A)]
def add(A,B): return [[A[i][j]+B[i][j] for j in range(2)] for i in range(2)]
def sc(a,A): return [[a*x for x in r] for r in A]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def mv(A,x): return [sum(a*b for a,b in zip(r,x)) for r in A]
def cg(A,C): return mm(mm(A,C),tr(A))
def outer(x,y): return [[a*b for b in y] for a in x]
def dot(x,y): return sum(a*b for a,b in zip(x,y))
def va(x,y): return [a+b for a,b in zip(x,y)]
def vs(x,y): return [a-b for a,b in zip(x,y)]
def norm(A): return math.sqrt(sum(x*x for r in A for x in r))
def err(A,B): return norm(add(A,sc(-1,B)))/max(1.,norm(B))
def eig(A):
    a,b,c=A[0][0],(A[0][1]+A[1][0])/2,A[1][1]
    m=(a+c)/2; r=math.hypot((a-c)/2,b)
    return m-r,m+r

def fun(A,f):
    lo,hi=eig(A)
    if abs(hi-lo)<1e-13: return sc(f((hi+lo)/2),I)
    slope=(f(hi)-f(lo))/(hi-lo)
    return add(sc(slope,A),sc(f(lo)-slope*lo,I))
def root(A): return fun(A,math.sqrt)
def invroot(A): return fun(A,lambda x:1/math.sqrt(x))
def exp(A): return fun(A,math.exp)
def log(A): return fun(A,math.log)
def transport(C,D):
    H=root(C); Hinv=invroot(C)
    return mm(mm(Hinv,root(cg(H,D))),Hinv)
def rot(t): return [[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]]
def spd(rng): return cg(rot(rng.uniform(-3,3)),[[rng.uniform(.2,2),0],[0,rng.uniform(.2,2)]])

def layer(mu,C,edges):
    outmu=[]; outC=[]
    for i in range(len(mu)):
        v=[0.,0.]; S=[[0.,0.],[0.,0.]]
        for j in edges[i]:
            r=vs(mu[j],mu[i]); r2=dot(r,r)
            a=math.tanh(r2+.1*sum(C[j][k][k] for k in range(2)))
            v=va(v,[a*x/(1+r2) for x in r])
            tensor=add(sc(1/(1+r2),outer(r,r)),sc(.2,add(C[j],sc(-1,C[i]))))
            S=add(S,sc(a,tensor))
        n=max(1,len(edges[i])); v=[.1*x/n for x in v]; S=sc(1/n,S)
        S=sc(.2/(1+norm(S)),S) # invariant scalar norm bound
        outmu.append(va(mu[i],v)); outC.append(cg(exp(S),C[i]))
    return outmu,outC

def main():
    rng=random.Random(71026); residuals=[]; roundtrip=[]; bound_slack=[]
    for _ in range(100):
        C,D=spd(rng),spd(rng); A=transport(C,D)
        residuals.append(err(cg(A,C),D)); roundtrip.append(err(exp(log(A)),A))
        S=[[rng.uniform(-.2,.2),rng.uniform(-.2,.2)],[0,rng.uniform(-.2,.2)]]; S[1][0]=S[0][1]
        D=cg(exp(S),C); c=max(abs(x) for x in eig(S))
        bound_slack.append(eig(D)[0]-math.exp(-2*c)*eig(C)[0])
    n=9; mu=[[rng.uniform(-1,1),rng.uniform(-1,1)] for _ in range(n)]; C=[spd(rng) for _ in mu]
    edges=[[(i-1)%n,(i+1)%n] for i in range(n)]
    out,co=layer(mu,C,edges); Q=rot(.713); t=[1.2,-.3]
    outg,cog=layer([va(mv(Q,x),t) for x in mu],[cg(Q,x) for x in C],edges)
    e_mu=max(math.dist(y,va(mv(Q,x),t)) for x,y in zip(out,outg)); e_C=max(err(y,cg(Q,x)) for x,y in zip(co,cog))
    p=list(range(n)); rng.shuffle(p); pinv={old:new for new,old in enumerate(p)}
    outp,cop=layer([mu[i] for i in p],[C[i] for i in p],[[pinv[j] for j in edges[i]] for i in p])
    e_perm=max(max(math.dist(outp[k],out[i]),err(cop[k],co[i])) for k,i in enumerate(p))
    fixedS=[[.2,0],[0,-.1]]
    fail=err(cg(exp(fixedS),cg(Q,I)),cg(Q,cg(exp(fixedS),I)))
    # Unit-circle cycle neighbor averaging is exactly multiplication by cos(2*pi/n).
    ncycle=16; circle=[[math.cos(2*math.pi*i/ncycle),math.sin(2*math.pi*i/ncycle)] for i in range(ncycle)]
    avg=[[sum(circle[j][k] for j in [(i-1)%ncycle,(i+1)%ncycle])/2 for k in range(2)] for i in range(ncycle)]
    observed=math.sqrt(dot(avg[0],avg[0])); expected=math.cos(2*math.pi/ncycle)
    # Noncommuting BW fixed-point solve; a diagnostic, not exact symbolic verification.
    Cs=[spd(rng) for _ in range(4)]; w=[.1,.2,.3,.4]; X=sum_mats(Cs,w)
    for iteration in range(1000):
        T=sum_mats([transport(X,D) for D in Cs],w); Y=cg(T,X)
        if err(Y,X)<1e-12: X=Y; break
        X=Y
    residual=norm(add(sum_mats([transport(X,D) for D in Cs],w),sc(-1,I)))
    bw_le_arith=eig(add(sum_mats(Cs,w),sc(-1,X)))[0]
    # SPD congruence can match a rotated ellipse but cannot reproduce the material rotation.
    C0=[[2.,0],[0,.4]]; target=cg(Q,C0); A=transport(C0,target)
    material_gap=math.dist(mv(A,[1.,0]),mv(Q,[1.,0]))
    result={"status":"algebra checks only; no training", "seed":71026,"python":sys.version,"platform":platform.platform(),
      "reachability_trials":100,"max_reachability_relative_error":max(residuals),"max_log_exp_relative_error":max(roundtrip),
      "min_eigenvalue_lower_bound_slack":min(bound_slack),"euclidean_mean_max_abs_error":e_mu,"euclidean_cov_max_relative_error":e_C,
      "permutation_max_error":e_perm,"fixed_coordinate_S_rotation_defect":fail,
      "cycle_nodes":ncycle,"cycle_radius_after_1":observed,"cycle_radius_expected":expected,"cycle_radius_after_32_expected":expected**32,
      "BW_iterations":iteration+1,"BW_stationarity_residual":residual,"min_eig_arithmetic_minus_BW":bw_le_arith,
      "rotated_covariance_relative_error":err(cg(A,C0),target),"material_rotation_point_gap":material_gap,
      "scalar_variance_example":{"arithmetic":5.,"BW":4.,"log_euclidean":3.,"precision_mean":1.8}}
    assert max(residuals)<1e-10 and max(roundtrip)<1e-10
    assert min(bound_slack)>-1e-10 and max(e_mu,e_C,e_perm)<1e-10
    assert fail>1e-3 and abs(observed-expected)<1e-12
    assert residual<1e-8 and bw_le_arith>-1e-8 and material_gap>1e-2
    result['assertions']='PASS'
    path=Path('experiments/gaussian_layer_checks.json'); path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
def sum_mats(Cs,w):
    X=[[0.,0.],[0.,0.]]
    for a,C in zip(w,Cs): X=add(X,sc(a,C))
    return X
if __name__=='__main__': main()
