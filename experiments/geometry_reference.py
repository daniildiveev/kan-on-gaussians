"""Executable benchmark definitions extracted from gaussian_kan_geometry_vs_mlp_v2.ipynb.

Sources are cells 3, 5, 9, 15, 17, and 19 (zero-based). Leave the
notebook's demonstrations, display calls and training triggers in the notebook.
Call the extracted definitions in the benchmark's float64 context.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn

DEVICE = torch.device("cpu")
DTYPE = torch.float64
INTERVALS, DEGREE = 8, 3
WIDTHS = (2, 3, 1)


@contextmanager
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)



def sym(M):
    return (M + M.transpose(-1, -2)) / 2

def eye_like(M):
    return torch.eye(2, device=M.device, dtype=M.dtype)

def trace(M):
    return M.diagonal(dim1=-2, dim2=-1).sum(-1)

def det2(M):
    return M[..., 0, 0]*M[..., 1, 1] - M[..., 0, 1]*M[..., 1, 0]

# Явные 2×2 формулы вместо torch.linalg (inv2, expm_sym2, eigmin_sym2, Холецкий в encode_gaussian).
# На CUDA linalg.inv/cholesky/eigvalsh проверяют ошибки и синхронизируют GPU с CPU на каждом вызове;
# matrix_exp для 2×2 избыточен (Паде + scaling-squaring). Здесь только поэлементные операции.
# Совпадение с torch.linalg проверяется в §3.
def inv2(M):
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    adj = torch.stack((d, -b, -c, a), -1).reshape(*M.shape[:-2], 2, 2)
    return adj / det2(M)[..., None, None]

def expm_sym2(S):
    # S = m·I + N, N бесследовая симметричная, N² = d²·I ⇒ exp S = e^m (cosh d · I + sinh(d)/d · N).
    # Около d = 0 — ряд Тейлора по d², чтобы значение и градиент оставались конечными.
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    m, h = (a + c)/2, (a - c)/2
    d2 = h*h + b*b
    small = d2 < 1e-4
    d = torch.where(small, torch.ones_like(d2), d2).sqrt()
    ch = torch.where(small, 1 + d2/2 + d2*d2/24 + d2**3/720, d.cosh())
    sc = torch.where(small, 1 + d2/6 + d2*d2/120 + d2**3/5040, d.sinh()/d)
    E = torch.stack((ch + sc*h, sc*b, sc*b, ch - sc*h), -1).reshape(*S.shape[:-2], 2, 2)
    return m.exp()[..., None, None]*E

def eigmin_sym2(M):
    M = sym(M)
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]
    return (a + c)/2 - (((a - c)/2).square() + b.square()).sqrt()

def sqrt_spd(M):
    M = sym(M)
    det = M[..., 0, 0]*M[..., 1, 1] - M[..., 0, 1]*M[..., 1, 0]
    tiny = torch.finfo(M.dtype).tiny
    root_det = det.clamp_min(tiny).sqrt()
    denom = (trace(M) + 2*root_det).clamp_min(tiny).sqrt()
    return (M + root_det[..., None, None]*eye_like(M)) / denom[..., None, None]

def invsqrt_spd(M):
    return inv2(sqrt_spd(M))

def w2_parts(G, H):
    mu, C = G; nu, D = H
    root = sqrt_spd(C)
    mean = (mu-nu).square().sum(-1)
    cov = (trace(C)+trace(D)-2*trace(sqrt_spd(root @ D @ root))).clamp_min(0)
    return mean, cov

def w2sq(G, H):
    mean, cov = w2_parts(G, H)
    return mean + cov

def cov_from_raw(raw):
    # raw = (log L11, L21, log L22); диагональ положительна.
    a, b, c = raw.unbind(-1)
    zero = torch.zeros_like(a)
    L = torch.stack((a.exp(), zero, b, c.exp()), -1).reshape(*raw.shape[:-1], 2, 2)
    return L @ L.transpose(-1, -2)

def encode_gaussian(mu, cov):
    # Холецкий 2×2 явно, теми же шагами, что LAPACK. Не-SPD вход даёт NaN, а не исключение:
    # такой прогон ловит проверка конечности в обучении и помечает как failed.
    C = sym(cov)
    l11 = C[..., 0, 0].sqrt()
    l21 = C[..., 1, 0]/l11
    l22 = (C[..., 1, 1] - l21*l21).sqrt()
    return torch.cat((mu, l11.log()[..., None], l21[..., None], l22.log()[..., None]), -1)

def decode_gaussian(z):
    return z[..., :2], cov_from_raw(z[..., 2:])

def transport_matrix(C, D):
    root, invroot = sqrt_spd(C), invsqrt_spd(C)
    return sym(invroot @ sqrt_spd(root @ D @ root) @ invroot)

def geodesic(G, H, t):
    # t: скаляр или тензор с batch-размерами без координатных осей.
    mu, C = G; nu, D = H
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=C.dtype, device=C.device)
    T = transport_matrix(C, D)
    A = (1-t)[..., None, None]*eye_like(C) + t[..., None, None]*T
    return ((1-t)[..., None]*mu + t[..., None]*nu,
            sym(A @ C @ A.transpose(-1, -2)))

def random_gaussians(n, nodes=2, seed=0, ood=False, shift_mean=None, shift_cov=None, device=None):
    """ood=True — исходный составной сдвиг (средние + анизотропия). shift_mean / shift_cov включают
    части по отдельности. При одном seed базовые случайные величины общие: варианты отличаются
    только контролируемым преобразованием."""
    shift_mean = ood if shift_mean is None else shift_mean
    shift_cov = ood if shift_cov is None else shift_cov
    rng = torch.Generator(device='cpu').manual_seed(seed)
    mu = 0.8*torch.randn(n, nodes, 2, generator=rng)
    angle = 2*math.pi*torch.rand(n, nodes, generator=rng)
    ca, sa = angle.cos(), angle.sin()
    R = torch.stack((ca, -sa, sa, ca), -1).reshape(n, nodes, 2, 2)
    log_std = 0.35*torch.randn(n, nodes, 2, generator=rng)
    if shift_mean:
        mu = 1.8*mu + torch.tensor([1.0, -0.6])
    if shift_cov:
        log_std = log_std + torch.tensor([0.5, -0.5])
    cov = R @ torch.diag_embed((2*log_std).exp()) @ R.transpose(-1, -2)
    device = DEVICE if device is None else device
    return mu.to(device), sym(cov).to(device)


def barycenter(G, weights=None, iters=12):
    mu, cov = G
    n = cov.shape[-3]
    if weights is None:
        weights = torch.full((n,), 1/n, device=cov.device, dtype=cov.dtype)
    mean = (mu*weights[..., None]).sum(-2)
    S = (cov*weights[..., None, None]).sum(-3)
    for _ in range(iters):
        root, invroot = sqrt_spd(S), invsqrt_spd(S)
        Q = (sqrt_spd(root.unsqueeze(-3) @ cov @ root.unsqueeze(-3))
             * weights[..., None, None]).sum(-3)
        S = sym(invroot @ Q @ Q @ invroot)
    return mean, S

def bary_residual(S, cov):
    root = sqrt_spd(S)
    Q = sqrt_spd(root.unsqueeze(-3) @ cov @ root.unsqueeze(-3)).mean(-3)
    return (Q-S).norm(dim=(-2,-1)) / S.norm(dim=(-2,-1)).clamp_min(1e-12)

def aggregate(msg, mode, iters):
    # iters задаётся конфигурацией прогона и автоматически не меняется ни в одной модели.
    if mode == 'mean':
        return msg[0].mean(-2), msg[1].mean(-3)
    if mode == 'bw':
        return barycenter(msg, iters=iters)
    raise ValueError(mode)


class BSplineBasis(nn.Module):
    """Открытый равномерный B-сплайн-базис; никаких обучаемых узлов."""
    def __init__(self, intervals=8, degree=3, left=-1., right=1.):
        super().__init__()
        self.degree, self.left, self.right = degree, left, right
        knots = torch.cat((torch.full((degree+1,),left),
                           torch.linspace(left,right,intervals+1)[1:-1],
                           torch.full((degree+1,),right)))
        self.register_buffer('knots',knots)
        self.n_basis = len(knots)-degree-1

    def forward(self,x):
        # Численное насыщение на границе; тест градиента проводится внутри интервала.
        eps = torch.finfo(x.dtype).eps*8
        x = x.clamp(self.left, self.right-eps)
        t = self.knots
        N = ((x[...,None]>=t[:-1]) & (x[...,None]<t[1:])).to(x.dtype)
        for p in range(1,self.degree+1):
            size = len(t)-p-1
            den1 = t[p:p+size]-t[:size]
            den2 = t[p+1:p+1+size]-t[1:1+size]
            d1 = torch.where(den1>0,den1,torch.ones_like(den1))
            d2 = torch.where(den2>0,den2,torch.ones_like(den2))
            a = (x[...,None]-t[:size])/d1*(den1>0)
            b = (t[p+1:p+1+size]-x[...,None])/d2*(den2>0)
            N = a*N[...,:size]+b*N[...,1:size+1]
        return N

class GaussianCoordinateLayer(nn.Module):
    def __init__(self,n_in,n_out,kind='spline',aggregation='mean',hidden=25,bary_iters=12):
        super().__init__()
        self.n_in,self.n_out = n_in,n_out
        self.kind,self.aggregation,self.bary_iters = kind,aggregation,bary_iters
        self.register_buffer('coord_scale',torch.tensor([1.5,1.5,.7,.7,.7]))
        self.s_limit = .6
        if kind=='spline':
            self.basis = BSplineBasis(INTERVALS,DEGREE)
            self.coeff = nn.Parameter(.015*torch.randn(n_out,n_in,5,5,self.basis.n_basis))
            self.bias = nn.Parameter(torch.zeros(n_out,n_in,5))
        elif kind=='mlp':
            # Независимый MLP 5→25→5 на каждом ребре, в векторизованной форме.
            self.W1=nn.Parameter(torch.randn(n_out,n_in,hidden,5)/math.sqrt(5))
            self.b1=nn.Parameter(torch.zeros(n_out,n_in,hidden))
            self.W2=nn.Parameter(.015*torch.randn(n_out,n_in,5,hidden))
            self.b2=nn.Parameter(torch.zeros(n_out,n_in,5))
        else:
            raise ValueError(kind)

    def forward(self,mu,cov,return_messages=False):
        x=torch.tanh(encode_gaussian(mu,cov)/self.coord_scale)
        if self.kind=='spline':
            q=torch.einsum('biqk,oirqk->boir',self.basis(x),self.coeff)+self.bias
        else:
            h=torch.tanh(torch.einsum('biq,oihq->boih',x,self.W1)+self.b1)
            q=torch.einsum('boih,oirh->boir',h,self.W2)+self.b2
        b=q[...,:2]
        s=self.s_limit*torch.tanh(q[...,2:]/self.s_limit)
        S=torch.stack((s[...,0],s[...,1],s[...,1],s[...,2]),-1).reshape(*s.shape[:-1],2,2)
        A=expm_sym2(S)
        msg=(mu[:,None]+b, sym(A @ cov[:,None] @ A.transpose(-1,-2)))
        result=aggregate(msg,self.aggregation,self.bary_iters)
        return (result,msg,(b,S,A)) if return_messages else result

class AnchorGeodesicLayer(nn.Module):
    def __init__(self,n_in,n_out,aggregation='bw',bary_iters=12):
        super().__init__()
        self.aggregation,self.bary_iters=aggregation,bary_iters
        self.anchor_mu=nn.Parameter(.5*torch.randn(n_out,n_in,2))
        self.anchor_raw=nn.Parameter(.1*torch.randn(n_out,n_in,3))
        self.log_tau=nn.Parameter(torch.zeros(n_out,n_in))
        self.basis=BSplineBasis(INTERVALS,DEGREE,0.,1.)
        self.coeff=nn.Parameter(torch.zeros(n_out,n_in,self.basis.n_basis))
        self.bias=nn.Parameter(torch.full((n_out,n_in),-2.2))

    def forward(self,mu,cov,return_messages=False):
        H=(self.anchor_mu,cov_from_raw(self.anchor_raw))
        incoming=(mu[:,None],cov[:,None])
        r=w2sq(incoming,H)/self.log_tau.exp().square()
        u=r/(1+r)
        t=torch.sigmoid((self.basis(u)*self.coeff).sum(-1)+self.bias)
        msg=geodesic(incoming,H,t)
        result=aggregate(msg,self.aggregation,self.bary_iters)
        return (result,msg,t) if return_messages else result

class GaussianNetwork(nn.Module):
    def __init__(self,kind='spline',aggregation='mean',widths=WIDTHS,bary_iters=12):
        super().__init__()
        self.layers=nn.ModuleList([
            AnchorGeodesicLayer(a,b,aggregation,bary_iters) if kind=='anchor'
            else GaussianCoordinateLayer(a,b,kind,aggregation,bary_iters=bary_iters)
            for a,b in zip(widths[:-1],widths[1:])])

    def forward(self,mu,cov):
        for layer in self.layers:
            mu,cov=layer(mu,cov)
        return mu,cov

class FlatMLP(nn.Module):
    def __init__(self,hidden=42):
        super().__init__()
        self.register_buffer('coord_scale',torch.tensor([1.5,1.5,.7,.7,.7]))
        self.net=nn.Sequential(nn.Linear(10,hidden),nn.Tanh(),
                               nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,5))
        # Предсказание остатка к среднему состоянию: старт тоже близок к mean-baseline.
        nn.init.normal_(self.net[-1].weight,std=.015)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,mu,cov):
        x=torch.tanh(encode_gaussian(mu,cov)/self.coord_scale).flatten(1)
        base=encode_gaussian(mu.mean(1),cov.mean(1))
        out=decode_gaussian(base+self.net(x))
        return out[0][:,None],out[1][:,None]

MODEL_SPECS={'Spline / mean':('spline','mean'), 'MLP edges / mean':('mlp','mean'), 'Flat MLP':('flat',None),
             'Spline / BW':('spline','bw'), 'MLP edges / BW':('mlp','bw'), 'Anchor / BW':('anchor','bw')}

MODELS=list(MODEL_SPECS)

MODEL_SLUG={'Spline / mean':'spline-mean','MLP edges / mean':'mlp-mean','Flat MLP':'flat',
            'Spline / BW':'spline-bw','MLP edges / BW':'mlp-bw','Anchor / BW':'anchor-bw'}

def build_model(name,bary_iters=12):
    kind,agg=MODEL_SPECS[name]
    return FlatMLP() if kind=='flat' else GaussianNetwork(kind,agg,bary_iters=bary_iters)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


TASKS=['affine','geodesic','nonlinear']      # основные задачи

DIAG_TASKS=['geo_position']                   # диагностика положения, этап C

ALL_TASKS=TASKS+DIAG_TASKS

DATA_SEED_ORIGINAL=2000                       # seed 2001/2002/2003/2004 — как в исходной тетрадке

OOD_VARIANTS={'ood_id':(False,False),'ood_mean':(True,False),'ood_cov':(False,True),'ood':(True,True)}

def task_splits(task):
    ood=['ood_id','ood_mean'] if task=='geo_position' else list(OOD_VARIANTS)
    return ['train','val','test']+ood

def target_fn(G,task):
    mu,cov=G
    m,C=mu.mean(1),cov.mean(1)
    if task=='affine':
        A=torch.tensor([[1.1,.15],[.15,.85]],device=mu.device)
        out=(m @ A.T + torch.tensor([.25,-.15],device=mu.device),sym(A @ C @ A.T))
    elif task=='geodesic':
        out=geodesic((mu[:,0],cov[:,0]),(mu[:,1],cov[:,1]),.35)
    elif task=='nonlinear':
        u,v=mu[:,0],mu[:,1]
        shift=torch.stack((.35*torch.sin(u[:,0]*v[:,1]),
                           .30*torch.tanh(u[:,1]+v[:,0])), -1)
        a=.18*torch.sin(u[:,0]+v[:,0])
        b=.12*torch.tanh(u[:,1]*v[:,1])
        c=-.15*torch.cos(u[:,1]-v[:,0])
        S=torch.stack((a,b,b,c),-1).reshape(-1,2,2)
        A=torch.linalg.matrix_exp(S)
        out=(m+shift,sym(A @ C @ A.transpose(-1,-2)))
    elif task=='geo_position':
        out=(.65*mu[:,0]+.35*mu[:,1], torch.eye(2,device=mu.device).expand_as(cov[:,0]).clone())
    else:
        raise ValueError(task)
    return out[0][:,None],out[1][:,None]

def make_inputs(data_seed,n_train,n_val,n_test,identity_cov=False):
    cpu=torch.device('cpu')
    G={'train':random_gaussians(n_train,2,data_seed+1,device=cpu),
       'val':random_gaussians(n_val,2,data_seed+2,device=cpu),
       'test':random_gaussians(n_test,2,data_seed+3,device=cpu)}
    for name,(shift_mean,shift_cov) in OOD_VARIANTS.items():
        G[name]=random_gaussians(n_test,2,data_seed+4,shift_mean=shift_mean,shift_cov=shift_cov,device=cpu)
    if identity_cov:
        G={k:(mu,torch.eye(2).expand_as(cov).clone()) for k,(mu,cov) in G.items()
           if k in ('train','val','test','ood_id','ood_mean')}
    return G

_DATA_CACHE={}

def make_dataset(task,data_seed,n_train,n_val,n_test):
    """{split: (G, Y)} на CPU. Зависит только от задачи, seed данных и размеров."""
    key=(task,data_seed,n_train,n_val,n_test)
    if key not in _DATA_CACHE:
        G=make_inputs(data_seed,n_train,n_val,n_test,identity_cov=(task=='geo_position'))
        _DATA_CACHE[key]={s:(G[s],target_fn(G[s],task)) for s in task_splits(task)}
    return _DATA_CACHE[key]

SIZES0=(256,128,256)


@dataclass(frozen=True)
class RunCfg:
    """Всё, что определяет результат прогона. Хэш полей — идентификатор."""
    model: str
    task: str
    data_seed: int
    train_seed: int
    lr: float
    steps: int
    batch: int
    n_train: int
    n_val: int
    n_test: int
    bary_iters: int
    dtype: str = 'float64'
    eval_every: int = 10
    early_window: int = 80
    clip_norm: float = 5.0
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    protocol: str = 'gkan-v2'   # версия кода данных/обучения/оценки; менять при изменении протокола

    @property
    def config_id(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @property
    def run_id(self):
        return (f'{MODEL_SLUG[self.model]}__{self.task}__d{self.data_seed}__s{self.train_seed}'
                f'__lr{self.lr:g}__{self.config_id}')

    def group_key(self):
        """Поля, которые должны совпадать у прогонов одного батча (vmap)."""
        d = asdict(self)
        for k in ('task', 'data_seed', 'train_seed', 'lr'):
            d.pop(k)
        return json.dumps(d, sort_keys=True)

RUN_FIELDS = ('steps','batch','n_train','n_val','n_test','bary_iters','dtype','eval_every','early_window')

BASE = dict(batch=64, n_train=256, n_val=128, n_test=256, bary_iters=12, dtype='float64',
            eval_every=10, early_window=80, device=str(DEVICE), save_every=200)

STAGES = {
    'smoke':         dict(BASE, steps=80, save_every=20, tasks=ALL_TASKS, models=MODELS,
                          data_seeds=[DATA_SEED_ORIGINAL], train_seeds=[11], lrs=[0.003]),
    'long_training': dict(BASE, steps=2000, tasks=TASKS, models=MODELS,
                          data_seeds=[DATA_SEED_ORIGINAL], train_seeds=[11, 29], lrs=[0.003]),
    'lr_search':     dict(BASE, steps=2000, tasks=TASKS, models=MODELS,
                          data_seeds=[DATA_SEED_ORIGINAL], train_seeds=[11, 29], lrs=[0.001, 0.003, 0.01]),
    'diagnostics':   dict(BASE, steps=2000, tasks=DIAG_TASKS, models=MODELS,
                          data_seeds=[DATA_SEED_ORIGINAL], train_seeds=[11, 29], lrs=[0.003]),
    'robustness':    dict(BASE, steps=2000, tasks=TASKS, models=MODELS,
                          data_seeds=[3000, 4000, 5000], train_seeds=[11, 29, 47], lrs='selected'),
}


CURVE_COLS = ['train_w2', 'train_mean_error', 'train_cov_error', 'val_w2', 'val_mean_error', 'val_cov_error']

METRIC_NAMES = ['w2', 'mean_error', 'cov_error', 'min_eig', 'symmetry_error', 'finite']

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

@torch.no_grad()
def batched_metrics(pred, Y):
    """pred, Y с осью прогонов 0: (R, N, 1, …) → (R, 6) в порядке METRIC_NAMES."""
    em, ec = w2_parts(pred, Y); C = pred[1]
    finite = torch.isfinite(pred[0]).flatten(1).all(1) & torch.isfinite(C).flatten(1).all(1)
    return torch.stack(((em + ec).flatten(1).mean(1), em.flatten(1).mean(1), ec.flatten(1).mean(1),
                        eigmin_sym2(C).flatten(1).amin(1), (C - C.transpose(-1, -2)).abs().flatten(1).amax(1),
                        finite.to(C.dtype)), 1)

@torch.no_grad()
def evaluate(model, pair):
    dev = next(model.parameters()).device
    G, Y = pair
    pred = model(*(g.to(dev) for g in G))
    row = batched_metrics(tuple(p[None] for p in pred), tuple(y.to(dev)[None] for y in Y))[0]
    return dict(zip(METRIC_NAMES, row.tolist()))


def batch_indices(n_train, batch, count, generator):
    """Draw with replacement in the order of train_group's CPU generator."""
    return torch.stack([torch.randint(n_train, (batch,), generator=generator) for _ in range(count)])
