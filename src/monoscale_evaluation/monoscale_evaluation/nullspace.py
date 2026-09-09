"""J_all = [J_F ; J_R] over the calibration directions, and what its structure
says: what is observable, what is indistinguishable, what the pair cancels, and
which manoeuvre separates what.

Run it. The table it prints is the answer to five things this project kept
rediscovering one at a time.

  parameter          g_front   g_rear   ratio   after the even average
  road height        +1.111%  +0.787%   1.411        +0.949%
  mount pitch        +0.534%  -0.598%  -0.894        -0.032%
  focal              -0.257%  -0.334%   0.769        -0.295%
  mount roll             ~0       ~0      --             ~0     (straight)
  mount x, y          0.0000%  0.0000%   --          0.0000%    (straight)

**A height error looks like scale** because dp_g/dh = -d/d_z stretches every
ground range by dh/h, uniformly. That is what scale means.

**The plane offset is the same column as the height.** Both enter through
lambda = -(h + z0)/d_z, so `ground_plane_offset_m` and the mount height are
one physical quantity carried by two parameters.

**A pitch error looks like scale too**, but it is the one column whose sign
flips between the mounts, which is what makes it separable and what the even
average is spending its single freedom on -- 0.534% down to 0.032%, a factor
of 17.

**The lever turns into a lateral error only in a turn.** Mount x and y read
exactly 0.0000% going straight and the y column wakes to +0.079% on the slalom.
Two lines, and the whole "lever arm becomes lateral error under turn" story.

**Range-dependent bias is the focal column.** Its front/rear ratio is 0.769 at
2 m/s and 0.322 at 8, because the band's effective range moves with speed.

And the structure: the singular values are 0.0142 and 0.0081 on every
manoeuvre, condition number 1.8. Two cameras, rank 2, and `w'1 = 1` spends one
of the freedoms -- so exactly one direction can be nulled and pitch is the one
worth spending it on, because it is the only column that is antisymmetric.
Height and focal are same-signed and pass through the average untouched. They
cannot be cancelled and no manoeuvre separates them from each other. They have
to be measured.
"""
import numpy as np, math, sys
sys.path.insert(0,"/tmp/claude-1000/-home-i-monoscale/2e1d95d6-8247-4109-8573-5841db9f5f99/scratchpad")
from jacobian import Cam, Rx, Ry, flow, step_leak
W,H=640,360; F=1051.81*W/2560.0
front=Cam([0,-0.5,0.8660254,-1,0,0,0,-0.8660254,-0.5],[3.694,0,0.89],F,(W-1)/2,(H-1)/2,W,H)
rear =Cam([0,0.5,-0.8660254,1,0,0,0,-0.8660254,-0.5],[-0.82,0,1.26],F,(W-1)/2,(H-1)/2,W,H)
band=(0.25,0.60,0.75,1.00)
def col(cam,S,T,p): return step_leak(cam,band,S,T,p)
def mk(kind,amt):
    if kind=="h":  return lambda cam,pix,s,t: flow(cam,pix,s,t,t_BC=cam.t+np.array([0,0,amt]))
    if kind=="z0": return lambda cam,pix,s,t: flow(cam,pix,s,t,t_BC=cam.t+np.array([0,0,amt]))
    if kind=="p":  return lambda cam,pix,s,t: flow(cam,pix,s,t,R_BC=Ry(amt)@cam.R)
    if kind=="r":  return lambda cam,pix,s,t: flow(cam,pix,s,t,R_BC=Rx(amt)@cam.R)
    if kind=="f":  return lambda cam,pix,s,t: flow(cam,pix,s,t,f=cam.f*(1+amt))
    if kind=="x":  return lambda cam,pix,s,t: flow(cam,pix,s,t,t_BC=cam.t+np.array([amt,0,0]))
    if kind=="y":  return lambda cam,pix,s,t: flow(cam,pix,s,t,t_BC=cam.t+np.array([0,amt,0]))
NAMES=[("노면 높이 / 평면 오프셋","h",0.010,"10mm"),
       ("마운트 pitch","p",math.radians(0.5),"0.5deg"),
       ("마운트 roll","r",math.radians(0.5),"0.5deg"),
       ("초점","f",0.01,"1%"),
       ("마운트 세로 위치","x",0.010,"10mm"),
       ("마운트 가로 위치","y",0.010,"10mm")]
for S,T,lbl in ((0.0615,0.0,"직선 2 m/s"),(0.0615,4.875e-3,"슬라롬 2 m/s"),(0.2667,0.0,"직선 8 m/s")):
    print(f"\n=== {lbl} ===")
    print(f"{'파라미터':24s} {'단위':>8s} {'g_front':>10s} {'g_rear':>10s} {'비':>8s} {'균등 후':>9s}")
    G=[]
    for name,kind,amt,unit in NAMES:
        gf=col(front,S,T,mk(kind,amt)); gr=col(rear,S,T,mk(kind,amt))
        G.append([gf,gr])
        ratio = gf/gr if abs(gr)>1e-9 else float('nan')
        print(f"{name:24s} {unit:>8s} {100*gf:9.4f}% {100*gr:9.4f}% {ratio:8.3f} {100*0.5*(gf+gr):8.4f}%")
    G=np.array(G).T   # 2 x k
    u,sv,vt=np.linalg.svd(G)
    print(f"  특이값 {sv[0]:.5f} {sv[1]:.5f}   조건수 {sv[0]/max(sv[1],1e-12):.1f}")
