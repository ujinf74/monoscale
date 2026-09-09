"""The ground projection's error Jacobian, analytic, checked against the
perturbation columns measured on straight120_v2.

  p_g = t_BC - (h / d_z) d,   d = R_BC pi^-1(u,v;K),   h = t_BC . e_z

A hop moves the vehicle by (s, psi). The photometric fit finds the (s, psi,
pitch, roll) whose induced image flow best matches the observed flow. If a
calibration parameter c is wrong by dc, the flow the model predicts is wrong,
and the fit absorbs it into its four freedoms. The step it then reports is

  ds/s = -(J_s^T J_s)^-1 J_s^T J_c dc     restricted to the step row

which is the same first-order shift the fit's own normal equations give. Here
J_s is the flow's derivative with respect to the four motion parameters and J_c
its derivative with respect to the calibration parameter.
"""
import numpy as np

def Rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])

class Cam:
    def __init__(self, R_BC, t_BC, f, cx, cy, w, h):
        self.R=np.array(R_BC,float).reshape(3,3); self.t=np.array(t_BC,float)
        self.f=f; self.cx=cx; self.cy=cy; self.w=w; self.h=h
    def bearing(self, u, v, f=None):
        f = self.f if f is None else f
        xn=(u-self.cx)/f; yn=(v-self.cy)/f
        th=np.hypot(xn,yn)
        s=np.where(th>1e-12, np.sin(th)/np.maximum(th,1e-12), 1.0)
        return np.stack([xn*s, yn*s, np.cos(th)], -1)
    def pixel(self, b, f=None):
        f = self.f if f is None else f
        r=np.hypot(b[...,0],b[...,1])
        th=np.arctan2(r,b[...,2])
        k=np.where(r>1e-12, th/np.maximum(r,1e-12), 1.0)
        return np.stack([b[...,0]*k*f+self.cx, b[...,1]*k*f+self.cy], -1)

def homography(cam, step, turn, pitch=0.0, roll=0.0, R_BC=None, t_BC=None):
    R = cam.R if R_BC is None else R_BC
    t = cam.t if t_BC is None else t_BC
    r_cb = R.T
    r_b = Rz(turn)
    rot = r_cb @ r_b.T @ R
    hop = np.array([step,0.0,0.0])
    tt = r_cb @ (r_b.T @ (t - hop) - t)
    n = r_cb @ np.array([0.,0.,1.])
    H = rot - np.outer(tt, n)/t[2]
    if pitch or roll:
        cp,sp,cr,sr=np.cos(pitch),np.sin(pitch),np.cos(roll),np.sin(roll)
        tilt=np.array([[cp,sp*sr,sp*cr],[0,cr,-sr],[-sp,cp*sr,cp*cr]])
        H = r_cb @ tilt.T @ R @ H
    return H

def flow(cam, pix, step, turn, pitch=0.0, roll=0.0, R_BC=None, t_BC=None, f=None):
    """Where each pixel's ground point was one hop ago, in pixels."""
    b = cam.bearing(pix[...,0], pix[...,1], f)
    if R_BC is not None or t_BC is not None or f is not None:
        cam2 = Cam(R_BC if R_BC is not None else cam.R,
                   t_BC if t_BC is not None else cam.t,
                   f if f is not None else cam.f, cam.cx, cam.cy, cam.w, cam.h)
    else:
        cam2 = cam
    H = homography(cam2, step, turn, pitch, roll)
    q = b @ np.linalg.inv(H).T
    return cam2.pixel(q, f)

def step_leak(cam, band, step, turn, perturb, dof=4):
    """How far the fit's step moves when a calibration parameter is perturbed.

    Least squares over the region: find the motion change that best cancels the
    flow change the perturbation caused. The step row of that is the leak.
    """
    u = np.linspace(band[0]*cam.w, band[2]*cam.w, 41)
    v = np.linspace(band[1]*cam.h, band[3]*cam.h, 41)
    U,V = np.meshgrid(u,v)
    pix = np.stack([U,V],-1)
    base = flow(cam, pix, step, turn)
    # motion Jacobian, central differences
    eps = [1e-5, 1e-6, 1e-6, 1e-6]
    cols=[]
    for k in range(dof):
        p=[step,turn,0.0,0.0]; m=[step,turn,0.0,0.0]
        p[k]+=eps[k]; m[k]-=eps[k]
        d=(flow(cam,pix,*p)-flow(cam,pix,*m))/(2*eps[k])
        cols.append(d.reshape(-1))
    J = np.stack(cols,1)
    # perturbed flow
    pert = perturb(cam, pix, step, turn)
    r = (pert - base).reshape(-1)
    sol,*_ = np.linalg.lstsq(J, r, rcond=None)
    return -sol[0]/step   # fractional step error per unit of the perturbation

if __name__=="__main__":
    W,H=640,360
    F=1051.81*W/2560.0
    front=Cam([0,-0.5,0.8660254,-1,0,0,0,-0.8660254,-0.5],[3.694,0,0.89],F,(W-1)/2,(H-1)/2,W,H)
    rear =Cam([0,0.5,-0.8660254,1,0,0,0,-0.8660254,-0.5],[-0.82,0,1.26],F,(W-1)/2,(H-1)/2,W,H)
    band=(0.25,0.60,0.75,1.00)
    S,T=0.0615,0.0
    def mk_height(dh):
        def p(cam,pix,s,t):
            return flow(cam,pix,s,t,t_BC=cam.t+np.array([0,0,dh]))
        return p
    def mk_pitch(da):
        def p(cam,pix,s,t):
            return flow(cam,pix,s,t,R_BC=Ry(da)@cam.R)
        return p
    def mk_roll(da):
        def p(cam,pix,s,t):
            return flow(cam,pix,s,t,R_BC=Rx(da)@cam.R)
        return p
    def mk_focal(rel):
        def p(cam,pix,s,t):
            return flow(cam,pix,s,t,f=cam.f*(1+rel))
        return p
    import math
    tests=[("높이 +10mm", mk_height(0.010), 1.0),
           ("pitch +0.5deg", mk_pitch(math.radians(0.5)), 1.0),
           ("roll +0.5deg",  mk_roll(math.radians(0.5)), 1.0),
           ("초점 +1%",      mk_focal(0.01), 1.0)]
    meas={"높이 +10mm":(1.123,0.794),"pitch +0.5deg":(0.520,-0.534),
          "roll +0.5deg":(0.082,-0.159),"초점 +1%":(-0.234,-0.319)}
    print(f"{'섭동':16s} {'해석 front':>11s} {'실측 front':>11s} {'해석 rear':>11s} {'실측 rear':>11s}")
    for name,p,_ in tests:
        gf=step_leak(front,band,S,T,p)
        gr=step_leak(rear,band,S,T,p)
        mf,mr=meas[name]
        print(f"{name:16s} {100*gf:10.3f}% {mf:10.3f}% {100*gr:10.3f}% {mr:10.3f}%")
